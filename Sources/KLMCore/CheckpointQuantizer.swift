import Foundation
import MLX

/// Native Swift+MLX checkpoint quantizer: converts a dense (bf16/fp16) HuggingFace
/// / MLX-format checkpoint to a k-bit MLX-format checkpoint that KrillLM's own
/// loader reads back, with no Python / mlx-lm shell-out.
///
/// It quantizes exactly the tensor set the loader reconstructs for a dense family:
/// every 2-D `*.weight` whose inner dimension is group-divisible (the Linear and
/// Embedding leaves), emitting the packed `weight` + `scales` (+ `biases`) triple
/// `MLX.quantized` produces, and passing every other tensor (norms, 1-D biases,
/// non-divisible weights) through unchanged. This mirrors `mlx_lm.convert`'s
/// effective behavior; on a dense checkpoint the produced packed tensors are
/// byte-identical to it (same MLX op, same group/bits/mode, same layer set).
///
/// MoE (stacked 3-D experts), vision/multimodal (towers kept fp + Conv2d layouts),
/// and Gemma 4 (PLE / tied-head specifics) checkpoints need per-family handling
/// the loader applies but this shape-driven pass does not, so they are rejected
/// up front rather than silently mis-quantized.
public enum CheckpointQuantizer {

    public enum QuantizeError: Error, CustomStringConvertible {
        case missingConfig(URL)
        case invalidConfig(URL)
        case unsupportedFamily(String)
        case noWeights(URL)

        public var description: String {
            switch self {
            case .missingConfig(let u): return "no config.json in \(u.path)"
            case .invalidConfig(let u): return "could not parse config.json in \(u.path)"
            case .unsupportedFamily(let why):
                return "native quantize supports dense text models only: \(why). "
                    + "Use a pre-quantized build or the Python mlx_lm path for this one."
            case .noWeights(let u): return "no .safetensors weights in \(u.path)"
            }
        }
    }

    /// Map a storage dtype name to an MLX float type. Non-quantized tensors and
    /// the scales/biases are stored at this precision (mlx_lm.convert defaults to
    /// fp16, which is the mlx-community convention; bf16 preserves source range).
    private static func storageDType(_ name: String) -> DType {
        switch name.lowercased() {
        case "bf16", "bfloat16": return .bfloat16
        case "fp32", "float32", "f32": return .float32
        default: return .float16
        }
    }

    /// Quantize `sourceDir` into `outputDir`. Returns the number of tensors quantized.
    @discardableResult
    public static func quantize(
        sourceDir: URL, outputDir: URL,
        bits: Int, groupSize: Int, mode: String = "affine", dtype: String = "fp16",
        log: (String) -> Void = { _ in }
    ) throws -> Int {
        let fm = FileManager.default

        // 1. Read + validate the source config; reject families this pass cannot
        //    handle structurally.
        let configURL = sourceDir.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: configURL) else {
            throw QuantizeError.missingConfig(configURL)
        }
        guard var config = (try? JSONSerialization.jsonObject(with: configData))
            as? [String: Any] else {
            throw QuantizeError.invalidConfig(configURL)
        }
        try assertSupportedDense(config)

        // 2. Load the flat source weight dict (merges shards; HF names verbatim).
        var weights = try loadWeightArrays(from: sourceDir)
        guard !weights.isEmpty else { throw QuantizeError.noWeights(sourceDir) }

        // 3. Quantize 2-D, group-divisible `*.weight` tensors; pass the rest through.
        //    Evaluate + drop each source tensor as we go so peak memory stays near
        //    (one source tensor + the growing quantized set), not (all bf16 + all
        //    quantized) at once.
        let qmode = mlxQuantizationMode(mode)
        let storeDType = storageDType(dtype)
        var out: [String: MLXArray] = [:]
        var quantizedCount = 0
        for name in weights.keys.sorted() {
            guard let w = weights[name] else { continue }
            if name.hasSuffix(".weight"), w.ndim == 2, w.dim(1) % groupSize == 0 {
                // Quantize AT the storage precision (mlx_lm.convert casts the model
                // to the target dtype first), so the derived scales/biases are
                // byte-identical to an mlx-community build rather than off by a
                // post-hoc-cast fp16 ULP.
                let wq0 = w.asType(storeDType)
                let (wq, scales, biases) = MLX.quantized(
                    wq0, groupSize: groupSize, bits: bits, mode: qmode)
                let stem = String(name.dropLast("weight".count))   // keeps trailing "."
                out[name] = wq
                out[stem + "scales"] = scales
                var realized: [MLXArray] = [wq, scales]
                if let biases {
                    out[stem + "biases"] = biases
                    realized.append(biases)
                }
                MLX.eval(realized)        // realize now so the source input can be freed
                quantizedCount += 1
            } else {
                // Pass-through (norms, 1-D biases, non-divisible weights): store float
                // tensors at the target precision; leave non-float tensors as-is.
                let isFloat = [DType.float16, .bfloat16, .float32].contains(w.dtype)
                out[name] = isFloat ? w.asType(storeDType) : w
            }
            weights[name] = nil           // release the source reference
        }
        log("quantized \(quantizedCount) tensors (\(bits)-bit, group \(groupSize), \(mode), \(dtype))")

        // 4. Write a single safetensors shard (the loader globs *.safetensors; an
        //    index.json is not required and not read).
        try fm.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let weightsURL = outputDir.appendingPathComponent("model.safetensors")
        try save(arrays: out, metadata: ["format": "mlx"], url: weightsURL)

        // 5. Emit config.json with the quantization block the loader reads
        //    (`quantization`), plus `quantization_config` for HF-tool parity.
        var qblock: [String: Any] = ["group_size": groupSize, "bits": bits]
        if qmode != .affine { qblock["mode"] = mode.lowercased() }
        config["quantization"] = qblock
        config["quantization_config"] = qblock
        let outConfig = try JSONSerialization.data(
            withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        try outConfig.write(to: outputDir.appendingPathComponent("config.json"))

        // 6. Copy tokenizer + auxiliary files the loader/tokenizer need.
        try copyAuxiliaryFiles(from: sourceDir, to: outputDir)

        return quantizedCount
    }

    /// Reject checkpoints whose quantized-layer set is NOT "all 2-D divisible
    /// weights": Mixture-of-Experts (stacked 3-D expert tensors), any model with a
    /// vision/multimodal tower (kept fp + Conv2d transpose), and Gemma (PLE / tied
    /// head specifics). Everything else is treated as a dense text model.
    private static func assertSupportedDense(_ config: [String: Any]) throws {
        if config["vision_config"] is [String: Any] {
            throw QuantizeError.unsupportedFamily("has a vision_config (multimodal/VL)")
        }
        for key in ["num_experts", "num_local_experts", "n_routed_experts",
                    "num_experts_per_tok", "moe_intermediate_size"] {
            if let v = config[key], !(v is NSNull) {
                throw QuantizeError.unsupportedFamily("Mixture-of-Experts (\(key))")
            }
        }
        let modelType = (config["model_type"] as? String ?? "").lowercased()
        let arch = ((config["architectures"] as? [Any])?.first as? String ?? "").lowercased()
        for bad in ["gemma", "vl", "vision", "llava", "mllama"] {
            if modelType.contains(bad) || arch.contains(bad) {
                throw QuantizeError.unsupportedFamily("family '\(modelType)' needs per-family handling")
            }
        }
    }

    /// Copy tokenizer + small auxiliary JSON/model files from the source, skipping
    /// the weights and the config (we write our own).
    private static func copyAuxiliaryFiles(from sourceDir: URL, to outputDir: URL) throws {
        let fm = FileManager.default
        let names = [
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "tokenizer.model", "chat_template.jinja", "merges.txt", "vocab.json",
            "added_tokens.json", "generation_config.json", "preprocessor_config.json",
        ]
        for name in names {
            let src = sourceDir.appendingPathComponent(name)
            guard fm.fileExists(atPath: src.path) else { continue }
            // HF cache entries are symlinks into a sibling `blobs/` dir; copying the
            // link verbatim would dangle. Resolve to the real file and copy that.
            let realSrc = src.resolvingSymlinksInPath()
            let dst = outputDir.appendingPathComponent(name)
            try? fm.removeItem(at: dst)
            try fm.copyItem(at: realSrc, to: dst)
        }
    }
}
