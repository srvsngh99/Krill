import Foundation
import MLX

/// Native Swift+MLX checkpoint quantizer: converts a dense (bf16/fp16) HuggingFace
/// / MLX-format checkpoint to a k-bit MLX-format checkpoint that KrillLM's own
/// loader reads back, with no Python / mlx-lm shell-out.
///
/// It quantizes every 2-D `*.weight` (the `Linear`/`Embedding` leaves), emitting
/// the packed `weight` + `scales` (+ `biases` for affine) `MLX.quantized`
/// produces, and passes every other tensor (norms, 1-D biases) through unchanged.
/// This is the same set `mlx_lm.convert` quantizes for a dense model, so the
/// produced tensors are byte-identical to the canonical MLX op - verified
/// 1007/1007 on GLM-4-9B-0414 affine 4-bit vs the mlx-community build
/// (`tools/verify_native_quantize_parity.sh`) and 765/765 for nvfp4 vs
/// `mx.quantize(mode:"nvfp4")` (`tools/verify_native_quantize_nvfp4.sh`).
///
/// IMPORTANT vs the loader: KrillLM's loader reconstructs quantized layers with
/// mlx-swift `quantize(model:)`, which quantizes EVERY `Linear`/`Embedding` leaf
/// uniformly (no per-layer divisibility skip), so a dense checkpoint is only
/// uniformly loadable if every such weight's inner dim is group-divisible.
/// `mlx_lm.convert` instead silently leaves a non-divisible layer dense, which
/// KrillLM's uniform load could not honor - so this quantizer THROWS on a
/// non-divisible 2-D weight rather than emit a checkpoint that would mis-load.
/// In practice every supported dense family has group-divisible dims.
///
/// MoE (stacked 3-D experts), vision/multimodal (towers kept fp + Conv2d layouts),
/// and Gemma (PLE / tied-head specifics) checkpoints need per-family handling
/// this shape-driven pass does not do, so they are rejected up front. affine and
/// nvfp4 are gated end-to-end; mxfp4/mxfp8 share the same path (auto group 32) but
/// are not separately gated.
public enum CheckpointQuantizer {

    public enum QuantizeError: Error, CustomStringConvertible {
        case missingConfig(URL)
        case invalidConfig(URL)
        case unsupportedFamily(String)
        case noWeights(URL)
        case nonDivisibleWeight(name: String, dim: Int, groupSize: Int)

        public var description: String {
            switch self {
            case .missingConfig(let u): return "no config.json in \(u.path)"
            case .invalidConfig(let u): return "could not parse config.json in \(u.path)"
            case .unsupportedFamily(let why):
                return "native quantize supports dense text models only: \(why). "
                    + "Use a pre-quantized build or the Python mlx_lm path for this one."
            case .noWeights(let u): return "no .safetensors weights in \(u.path)"
            case .nonDivisibleWeight(let name, let dim, let groupSize):
                return "weight '\(name)' inner dim \(dim) is not divisible by group size "
                    + "\(groupSize); KrillLM's loader quantizes every Linear uniformly, so "
                    + "it cannot load a checkpoint with this layer left dense. Use a group "
                    + "size that divides \(dim)."
            }
        }
    }

    /// Map a storage dtype name to an MLX float type. Non-quantized tensors are
    /// stored at this precision (mlx_lm.convert defaults to fp16, the
    /// mlx-community convention; bf16 preserves source range).
    private static func storageDType(_ name: String) -> DType {
        switch name.lowercased() {
        case "bf16", "bfloat16": return .bfloat16
        case "fp32", "float32", "f32": return .float32
        default: return .float16
        }
    }

    /// The MLX-required (bits, groupSize) for the float quantization formats, so
    /// `--mode nvfp4` works without the caller also knowing it needs group 16. The
    /// micro-scaled formats only support one shape each (nvfp4: 4-bit / group 16;
    /// mxfp4: 4-bit / group 32; mxfp8: 8-bit / group 32). `affine` is flexible, so
    /// the caller's bits/groupSize are honored as-is.
    private static func effectiveParams(mode: String, bits: Int, groupSize: Int)
        -> (bits: Int, groupSize: Int) {
        switch mode.lowercased() {
        case "nvfp4": return (4, 16)
        case "mxfp4": return (4, 32)
        case "mxfp8": return (8, 32)
        default: return (bits, groupSize)
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
        // Float formats (nvfp4/mxfp4/mxfp8) only support one (bits, groupSize) each;
        // honor that regardless of the passed flags so `--mode nvfp4` just works.
        let (qBits, qGroup) = effectiveParams(mode: mode, bits: bits, groupSize: groupSize)
        if qBits != bits || qGroup != groupSize {
            log("mode \(mode) requires \(qBits)-bit / group \(qGroup); overriding the passed bits/group")
        }
        var out: [String: MLXArray] = [:]
        var quantizedCount = 0
        for name in weights.keys.sorted() {
            guard let w = weights[name] else { continue }
            // A 2-D `.weight` is a Linear/Embedding the loader WILL quantize; if its
            // inner dim is not group-divisible we cannot produce a uniformly-loadable
            // checkpoint, so fail loudly instead of silently leaving it dense.
            if name.hasSuffix(".weight"), w.ndim == 2, w.dim(1) % qGroup != 0 {
                throw QuantizeError.nonDivisibleWeight(
                    name: name, dim: w.dim(1), groupSize: qGroup)
            }
            if name.hasSuffix(".weight"), w.ndim == 2, w.dim(1) % qGroup == 0 {
                // affine: quantize AT the storage precision (mlx_lm.convert casts the
                // model to fp16 first) so scales/biases are byte-identical to an
                // mlx-community build. Float formats (nvfp4/...) quantize from the
                // SOURCE dtype, matching tools/requant_gemma4_nvfp4.py, and carry
                // their own scale encoding (e.g. uint8 block scales, no biases).
                let wIn = qmode == .affine ? w.asType(storeDType) : w
                let (wq, scales, biases) = MLX.quantized(
                    wIn, groupSize: qGroup, bits: qBits, mode: qmode)
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
        log("quantized \(quantizedCount) tensors (\(qBits)-bit, group \(qGroup), \(mode), \(dtype))")

        // 4. Write a single safetensors shard (the loader globs *.safetensors; an
        //    index.json is not required and not read).
        try fm.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let weightsURL = outputDir.appendingPathComponent("model.safetensors")
        try save(arrays: out, metadata: ["format": "mlx"], url: weightsURL)

        // 5. Emit config.json with the quantization block the loader reads
        //    (`quantization`), plus `quantization_config` for HF-tool parity.
        var qblock: [String: Any] = ["group_size": qGroup, "bits": qBits]
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
        // Vision/VL towers are already caught by `vision_config` above; here we
        // reject the remaining families that need per-family handling, matching
        // unambiguous model_type / architecture markers (not bare substrings like
        // "vl" that could clip an unrelated name).
        if arch.contains("forconditionalgeneration") {
            throw QuantizeError.unsupportedFamily("multimodal architecture '\(arch)'")
        }
        for bad in ["gemma", "llava", "mllama"] {
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
