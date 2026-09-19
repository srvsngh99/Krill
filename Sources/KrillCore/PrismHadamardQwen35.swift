import Foundation
import MLX
import MLXNN

// Native Swift+MLX support for Prism ML's `prism_hadamard_qwen35` checkpoints
// (e.g. Ternary-Bonsai-2-27B): the SAME Qwen3.5-class hybrid decoder
// `Qwen35ForCausalLM` already implements (GatedDeltaNet linear-attention
// layers interleaved with full attention every `full_attention_interval`),
// but 402 of its weight matrices carry a BLOCKWISE HADAMARD ROTATION folded
// into their affine 2-bit/group-128 quantization (the "Prism" pack). An
// ordinary affine load produces garbage: the matching transform must be
// applied to ACTIVATIONS at inference, which is what `PrismPackedLinear` /
// `PrismPackedEmbedding` do below. Ported from the pack's own
// `runtime/runtime.py` (`fwht()` + class `Packed`) - see PACK-RUNTIME.md.
//
// TEXT-ONLY: the pack's safetensors also carry a `vision_tower.*` weight set
// (333 tensors), but NONE of them appear in the 402-entry Hadamard manifest,
// and PACK-RUNTIME.md states the bundled runtime is text-only. Loading is
// wired up in `loadPrismHadamardQwen35` (ModelLoader.swift), which drops the
// vision tower exactly as `loadQwen35` does.

// MARK: - fwht (blockwise Walsh-Hadamard activation transform)

/// Port of `runtime.py`'s `fwht()`, EXACTLY:
///
/// ```python
/// def fwht(x, block, signs, inverse=False):
///     shape, dtype = x.shape, x.dtype
///     if shape[-1] % block: raise ValueError(...)
///     x = x.astype(mx.float32)
///     if not inverse: x = x * signs
///     x = mx.hadamard_transform(x.reshape(-1, block), scale=1/sqrt(block)).reshape(shape)
///     if inverse: x = x * signs
///     return x.astype(dtype)
/// ```
///
/// `block` must divide `x`'s last dimension and `signs` must have exactly
/// `block`-aligned length equal to that last dimension (both are guaranteed
/// by construction here - every `PrismPackedLinear`/`PrismPackedEmbedding`
/// is built from a manifest entry already validated against `hadamard.json`,
/// so a mismatch here would mean a loader bug, not a bad checkpoint - hence
/// `precondition` rather than `throws`, matching `UnaryLayer.callAsFunction`
/// not being a throwing signature).
func fwht(_ x: MLXArray, block: Int, signs: MLXArray, inverse: Bool = false) -> MLXArray {
    let shape = x.shape
    let dtype = x.dtype
    precondition(
        shape.last! % block == 0,
        "Hadamard block (\(block)) does not divide activation width (\(shape.last!))")
    var xf = x.asType(.float32)
    if !inverse {
        xf = xf * signs
    }
    xf = MLX.hadamardTransform(xf.reshaped([-1, block]), scale: 1 / Float(block).squareRoot())
        .reshaped(shape)
    if inverse {
        xf = xf * signs
    }
    return xf.asType(dtype)
}

// MARK: - Packed modules

/// A folded-Hadamard affine 2-bit/group-128 Linear: rotates the input
/// activation (`fwht`, forward direction) THEN runs the quantized matmul.
/// Mirrors `runtime.py`'s `Packed.__call__` non-embedding branch exactly.
/// Substituting an ordinary `QuantizedLinear` here would skip the rotation
/// and emit confidently wrong (fluent but garbage) logits - see
/// `loadPrismHadamardQwen35`'s doc comment for why this can't be reached via
/// MLX's generic `quantize(model:...)`.
///
/// `weight`/`scales`/`biases` are plain (non-`@ModuleInfo`) stored
/// properties, matching upstream MLX's own `QuantizedLinear`/`Linear`: this
/// module is always constructed FULLY FORMED from the checkpoint's actual
/// tensors (see `makePrismPackedLinear`) and never touched by a later
/// generic `model.update(parameters:)` pass, so there is no need for the
/// reassignable-property machinery `@ModuleInfo`/`@ParameterInfo` provide.
public final class PrismPackedLinear: Module, UnaryLayer {
    public let weight: MLXArray // uint32, [rows, width/16]
    public let scales: MLXArray // [rows, width/128]
    public let biases: MLXArray // [rows, width/128]
    let block: Int
    /// Sign vector for THIS module's input width, picked once at
    /// construction from `PrismHadamardConfig.signs[width]`.
    let signs: MLXArray

    public init(weight: MLXArray, scales: MLXArray, biases: MLXArray, block: Int, signs: MLXArray) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.block = block
        self.signs = signs
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rotated = fwht(x, block: block, signs: signs, inverse: false)
        return MLX.quantizedMM(
            rotated, weight, scales: scales, biases: biases,
            transpose: true, groupSize: 128, bits: 2, mode: .affine)
    }
}

/// A folded-Hadamard affine 2-bit/group-128 Embedding: dequantizes the
/// gathered rows THEN applies the INVERSE `fwht` (mirrors `runtime.py`'s
/// `Packed.__call__` embedding branch). The Prism manifest's inverse set is
/// exactly `language_model.model.embed_tokens.weight` - the only embedding
/// this pack folds.
public final class PrismPackedEmbedding: Module, UnaryLayer {
    public let weight: MLXArray // uint32, [vocab, width/16]
    public let scales: MLXArray // [vocab, width/128]
    public let biases: MLXArray // [vocab, width/128]
    let block: Int
    let signs: MLXArray

    public init(weight: MLXArray, scales: MLXArray, biases: MLXArray, block: Int, signs: MLXArray) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.block = block
        self.signs = signs
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let indices = x.flattened()
        let dequant = MLX.dequantized(
            weight[indices], scales: scales[indices], biases: biases[indices],
            groupSize: 128, bits: 2, mode: .affine)
        let out = dequant.reshaped(shape + [-1]).asType(.float16)
        return fwht(out, block: block, signs: signs, inverse: true)
    }
}

// MARK: - Packed-module manifest (config.json's `modules` array)

/// One entry of config.json's `modules` array: a packed module's dotted
/// path (relative to `Qwen35ForCausalLM`, e.g. `"lm_head"`,
/// `"model.embed_tokens"`, `"model.layers.3.mlp.gate_proj"`), its Hadamard
/// block size, and whether it is the (inverse-transform) embedding.
struct PrismModuleRecord: Decodable {
    let path: String
    let block: Int
    let embedding: Bool
    let dtype: String
}

// MARK: - hadamard.json contract

/// Decoded `hadamard.json`: the version-1 Prism Hadamard contract. Field
/// names are the literal (dotted) top-level JSON keys - NOT a nested
/// `prism.hadamard` object.
struct PrismHadamardManifest: Decodable {
    let version: Int
    let blockSize: Int
    let transform: String
    let axis: String
    let signMode: String
    /// 401 entries: the forward-transform (folded Linear) manifest, full
    /// safetensors tensor names (`language_model.` prefix, `.weight` suffix).
    let weightNames: [String]
    /// Exactly 1 entry: the inverse-transform (embedding) manifest.
    let inverseWeightNames: [String]
    let signWidths: [Int]
    let signValues: [Float]
    let gdnVGrouped: Bool

    enum CodingKeys: String, CodingKey {
        case version = "prism.hadamard.version"
        case blockSize = "prism.hadamard.block_size"
        case transform = "prism.hadamard.transform"
        case axis = "prism.hadamard.axis"
        case signMode = "prism.hadamard.sign_mode"
        case weightNames = "prism.hadamard.weight_names"
        case inverseWeightNames = "prism.hadamard.inverse_weight_names"
        case signWidths = "prism.hadamard.sign_widths"
        case signValues = "prism.hadamard.sign_values"
        case gdnVGrouped = "prism.hadamard.gdn_v_grouped"
    }
}

/// Valid Hadamard block sizes - shared by the top-level `block_size` and
/// every per-module `block` in config.json's `modules` array.
private let validHadamardBlockSizes: Set<Int> = [512, 1024, 2048, 4096]

/// Slice `values` sequentially into one vector per `widths` entry (in
/// order), keyed by width, validating every element is exactly `+-1` and
/// that the slices consume `values` exactly (no trailing values). Ports
/// `runtime.py`'s `load()` sign-table construction:
///
/// ```python
/// signs, offset = {}, 0
/// for width in widths:
///     a = values[offset:offset+width]
///     if len(a) != width or not isin(a, [-1, 1]).all(): raise ValueError(...)
///     signs[width] = mx.array(a); offset += width
/// if offset != len(values): raise ValueError("Trailing sign values")
/// ```
func buildPrismSignTable(widths: [Int], values: [Float]) throws -> [Int: MLXArray] {
    var table: [Int: MLXArray] = [:]
    var offset = 0
    for width in widths {
        guard width > 0, offset + width <= values.count else {
            throw ModelLoadError.invalidConfig(
                "Hadamard sign_values too short for sign_widths (width \(width) at offset \(offset))")
        }
        let slice = Array(values[offset ..< (offset + width)])
        guard slice.allSatisfy({ $0 == 1 || $0 == -1 }) else {
            throw ModelLoadError.invalidConfig(
                "Invalid Hadamard sign vector for width \(width): every value must be +-1")
        }
        table[width] = MLXArray(slice)
        offset += width
    }
    guard offset == values.count else {
        throw ModelLoadError.invalidConfig(
            "Trailing Hadamard sign values: \(values.count - offset) unconsumed")
    }
    return table
}

// MARK: - PrismHadamardConfig

/// Full decoded + validated `prism_hadamard_qwen35` contract: the Qwen3.5
/// text hyperparameters (reuses `Qwen35Config` - identical architecture),
/// the 402-entry packed-module manifest, and the width-keyed Hadamard sign
/// table. Every validation `runtime.py`'s `load()` and `hadamard.json`'s own
/// contract require happens here - a config that fails to construct must
/// not be loaded, not loaded wrong.
struct PrismHadamardConfig {
    let textConfig: Qwen35Config
    let modules: [PrismModuleRecord]
    let quantization: QuantizationConfig
    let blockSize: Int
    /// Sign vector keyed by a packed module's INPUT width (this checkpoint
    /// has three: `hidden_size` 5120, `num_heads*head_dim` / GDN value width
    /// 6144, and `intermediate_size` 17408). A module picks `signs[width]`
    /// once at construction (`makePrismPackedLinear`/`makePrismPackedEmbedding`).
    let signs: [Int: MLXArray]

    /// Top-level config.json wrapper. `hadamardConfigFile` names the
    /// sibling contract file (`hadamard_config`, defaults to
    /// `"hadamard.json"` when absent).
    private struct Wrapper: Decodable {
        let modelType: String
        let textConfig: Qwen35Config
        let modules: [PrismModuleRecord]
        let quantization: QuantizationConfig
        let tensorNamespace: String
        let gdnActivationLayout: String
        let tieWordEmbeddings: Bool
        let hadamardConfigFile: String?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case textConfig = "text_config"
            case modules
            case quantization
            case tensorNamespace = "tensor_namespace"
            case gdnActivationLayout = "gdn_activation_layout"
            case tieWordEmbeddings = "tie_word_embeddings"
            case hadamardConfigFile = "hadamard_config"
        }
    }

    init(configData: Data, directory: URL) throws {
        let wrapper = try JSONDecoder().decode(Wrapper.self, from: configData)

        // `tensor_namespace`/`gdn_activation_layout` are the two load-bearing
        // invariants PACK-RUNTIME.md calls out explicitly: this loader
        // assumes mlx-vlm key naming (so `loadWeights`'s `language_model.`
        // prefix strip lands on `Qwen35ForCausalLM`'s tree) and that GDN v is
        // ALREADY GROUPED (so applying no extra permutation, exactly like
        // `loadQwen35`, is correct rather than an oversight).
        guard wrapper.tensorNamespace == "mlx-vlm-qwen3_5" else {
            throw ModelLoadError.invalidConfig(
                "Unsupported tensor_namespace \(wrapper.tensorNamespace); expected mlx-vlm-qwen3_5")
        }
        guard wrapper.gdnActivationLayout == "grouped" else {
            throw ModelLoadError.invalidConfig(
                "Unsupported gdn_activation_layout \(wrapper.gdnActivationLayout); "
                + "this loader applies no GDN-v permutation and requires \"grouped\"")
        }
        guard wrapper.tieWordEmbeddings == false else {
            throw ModelLoadError.invalidConfig(
                "prism_hadamard_qwen35 requires tie_word_embeddings=false (explicit lm_head)")
        }
        guard wrapper.quantization.bits == 2, wrapper.quantization.groupSize == 128,
            wrapper.quantization.mode == "affine"
        else {
            throw ModelLoadError.invalidConfig(
                "Unsupported quantization \(wrapper.quantization.bits)-bit/group"
                + "\(wrapper.quantization.groupSize)/\(wrapper.quantization.mode); "
                + "PrismPackedLinear/Embedding hardcode 2-bit/group-128/affine")
        }

        let hadamardURL = directory.appendingPathComponent(
            wrapper.hadamardConfigFile ?? "hadamard.json")
        let hadamardData = try Data(contentsOf: hadamardURL)
        let hadamard = try JSONDecoder().decode(PrismHadamardManifest.self, from: hadamardData)

        guard hadamard.version == 1 else {
            throw ModelLoadError.invalidConfig(
                "Unsupported Hadamard contract version \(hadamard.version); only version 1 is implemented")
        }
        guard validHadamardBlockSizes.contains(hadamard.blockSize) else {
            throw ModelLoadError.invalidConfig("Unvalidated Hadamard block size \(hadamard.blockSize)")
        }
        guard hadamard.signMode == "explicit" else {
            throw ModelLoadError.invalidConfig("Explicit Hadamard signs required, got sign_mode=\(hadamard.signMode)")
        }
        guard hadamard.gdnVGrouped else {
            throw ModelLoadError.invalidConfig(
                "hadamard.json declares gdn_v_grouped=false; this loader applies no GDN-v permutation")
        }

        let signTable = try buildPrismSignTable(widths: hadamard.signWidths, values: hadamard.signValues)

        // Forward/inverse manifests: validate hadamard.json's contract
        // against itself...
        let forward = Set(hadamard.weightNames)
        let inverse = Set(hadamard.inverseWeightNames)
        guard inverse == ["language_model.model.embed_tokens.weight"] else {
            throw ModelLoadError.invalidConfig("Unexpected inverse-transform manifest: \(inverse.sorted())")
        }
        guard forward.isDisjoint(with: inverse) else {
            throw ModelLoadError.invalidConfig("Forward and inverse Hadamard transform manifests overlap")
        }

        // ...then cross-validate it against config.json's `modules` array,
        // which is what this loader actually drives module construction
        // from (per the task's ground truth: "prefer driving module
        // construction off config.json's modules array ... and use
        // hadamard.json to validate and to source the sign vectors").
        let expectedForward = Set(
            wrapper.modules.filter { !$0.embedding }.map { "language_model." + $0.path + ".weight" })
        let expectedInverse = Set(
            wrapper.modules.filter { $0.embedding }.map { "language_model." + $0.path + ".weight" })
        guard expectedForward == forward else {
            throw ModelLoadError.invalidConfig(
                "config.json modules manifest does not match hadamard.json weight_names")
        }
        guard expectedInverse == inverse else {
            throw ModelLoadError.invalidConfig(
                "config.json modules manifest does not match hadamard.json inverse_weight_names")
        }
        for m in wrapper.modules {
            guard validHadamardBlockSizes.contains(m.block) else {
                throw ModelLoadError.invalidConfig("Module \(m.path): unvalidated Hadamard block size \(m.block)")
            }
            guard m.block == hadamard.blockSize else {
                throw ModelLoadError.invalidConfig(
                    "Module \(m.path): block \(m.block) does not match hadamard.json block_size \(hadamard.blockSize)")
            }
        }

        self.textConfig = wrapper.textConfig
        self.modules = wrapper.modules
        self.quantization = wrapper.quantization
        self.blockSize = hadamard.blockSize
        self.signs = signTable
    }
}

// MARK: - Packed-module construction

/// Build one `PrismPackedLinear`, replicating `artifact.py`'s
/// `validate_record` shape contract exactly: a packed module of dense shape
/// `(rows, width)` stores `weight (rows, width/16) uint32`,
/// `scales/biases (rows, width/128)`, with `width % 128 == 0` and
/// `width % block == 0`. `width` (the module's input dimension - what
/// `fwht`/`signs` key on) is recovered from the packed weight's own shape,
/// exactly like `runtime.py`'s `signs[shape[1]]`.
func makePrismPackedLinear(
    path: String, block: Int, weight: MLXArray, scales: MLXArray, biases: MLXArray,
    signs: [Int: MLXArray]
) throws -> PrismPackedLinear {
    guard weight.dtype == .uint32, weight.ndim == 2 else {
        throw ModelLoadError.invalidConfig("\(path): packed weight must be a 2D uint32 array")
    }
    let rows = weight.dim(0)
    let width = weight.dim(1) * 16
    guard width % 128 == 0 else {
        throw ModelLoadError.invalidConfig("\(path): invalid packed width \(width)")
    }
    let groupCols = width / 128
    guard scales.shape == [rows, groupCols], biases.shape == [rows, groupCols] else {
        throw ModelLoadError.invalidConfig(
            "\(path): scales/biases shape \(scales.shape)/\(biases.shape) does not match packed width \(width)")
    }
    guard width % block == 0 else {
        throw ModelLoadError.invalidConfig("\(path): Hadamard block \(block) does not divide width \(width)")
    }
    guard let signVector = signs[width] else {
        throw ModelLoadError.invalidConfig("\(path): no Hadamard sign vector for width \(width)")
    }
    return PrismPackedLinear(weight: weight, scales: scales, biases: biases, block: block, signs: signVector)
}

/// Build the single `PrismPackedEmbedding` (`model.embed_tokens`). Same
/// packed-shape contract as `makePrismPackedLinear`, except `width` is the
/// embedding's OUTPUT dimension (`hidden_size` - the dequantized row width),
/// matching `runtime.py`'s embedding branch (`signs[shape[1]]` where `shape`
/// is the dense `(vocab, hidden)` embedding shape).
func makePrismPackedEmbedding(
    path: String, block: Int, weight: MLXArray, scales: MLXArray, biases: MLXArray,
    signs: [Int: MLXArray]
) throws -> PrismPackedEmbedding {
    guard weight.dtype == .uint32, weight.ndim == 2 else {
        throw ModelLoadError.invalidConfig("\(path): packed weight must be a 2D uint32 array")
    }
    let rows = weight.dim(0)
    let width = weight.dim(1) * 16
    guard width % 128 == 0 else {
        throw ModelLoadError.invalidConfig("\(path): invalid packed width \(width)")
    }
    let groupCols = width / 128
    guard scales.shape == [rows, groupCols], biases.shape == [rows, groupCols] else {
        throw ModelLoadError.invalidConfig(
            "\(path): scales/biases shape \(scales.shape)/\(biases.shape) does not match packed width \(width)")
    }
    guard width % block == 0 else {
        throw ModelLoadError.invalidConfig("\(path): Hadamard block \(block) does not divide width \(width)")
    }
    guard let signVector = signs[width] else {
        throw ModelLoadError.invalidConfig("\(path): no Hadamard sign vector for width \(width)")
    }
    return PrismPackedEmbedding(weight: weight, scales: scales, biases: biases, block: block, signs: signVector)
}
