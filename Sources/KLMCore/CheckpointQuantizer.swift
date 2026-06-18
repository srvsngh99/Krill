import Foundation
import MLX

/// Native Swift+MLX checkpoint quantizer: converts a dense (bf16/fp16) HuggingFace
/// / MLX-format checkpoint to a k-bit MLX-format checkpoint that KrillLM's own
/// loader reads back, with no Python / mlx-lm shell-out.
///
/// Two modes of operation, picked by whether a `referenceDir` is supplied:
///
/// 1. **Dense (no reference).** Quantizes every 2-D `*.weight` (the
///    `Linear`/`Embedding` leaves) and passes every other tensor through. This is
///    the set `mlx_lm.convert` quantizes for a dense model, so the produced
///    tensors are byte-identical to the canonical MLX op - verified 1007/1007 on
///    GLM-4-9B-0414 affine 4-bit and 765/765 for nvfp4. Families that need
///    per-module handling (MoE / vision / Gemma) are rejected up front.
///
/// 2. **Reference-set (a 4-bit checkpoint passed as `referenceDir`).** Learns the
///    EXACT quantized-module set from the reference checkpoint's `.scales` tensors
///    (every `<module>.scales` => quantize `<module>.weight`) and quantizes only
///    those, passing the rest through. This is the generalization of
///    `tools/requant_gemma4_nvfp4.py` to every family: it reproduces the proven
///    mlx-community coverage exactly, so it handles MoE (stacked 3-D experts,
///    DeepSeek's float router gate), vision/multimodal (vision tower kept float on
///    Qwen2.5-VL, quantized on LLaVA), and Gemma (PLE / tied-head specifics)
///    without reverse-engineering each loader predicate. A `--protect` list raises
///    chosen modules to a higher precision (the Gemma vision/audio projectors are
///    auto-protected at 8-bit affine, matching the recipe) and is recorded as
///    per-module overrides in `config.quantization` - the exact `q.effective(path)`
///    keys KrillLM's loader resolves. Stacked 3-D expert weights are always
///    quantized affine at the config's TOP-LEVEL group (the MoE runtime
///    reconstructs them affine from the top-level group and reads no override).
///
/// IMPORTANT vs the loader: KrillLM's loader reconstructs quantized layers with
/// mlx-swift `quantize(model:)`, which only turns `Linear`/`Embedding` leaves into
/// quantized leaves; `strictVerify` then crashes on any extra/missing `.scales`.
/// The checkpoint must therefore ship `weight`+`scales`(+`biases`) for EXACTLY the
/// set the loader will quantize. The reference set is the safest way to match that
/// (it is the set a proven build already shipped); the dense pass matches it for a
/// plain dense model. A non-divisible quantized weight is rejected loudly rather
/// than emitted, since the uniform load could not honor a layer left dense.
public enum CheckpointQuantizer {

    public enum QuantizeError: Error, CustomStringConvertible {
        case missingConfig(URL)
        case invalidConfig(URL)
        case unsupportedFamily(String)
        case noWeights(URL)
        case noReferenceScales(URL)
        case nonDivisibleWeight(name: String, dim: Int, groupSize: Int)
        case expertGroupUnsupported(group: Int)

        public var description: String {
            switch self {
            case .missingConfig(let u): return "no config.json in \(u.path)"
            case .invalidConfig(let u): return "could not parse config.json in \(u.path)"
            case .unsupportedFamily(let why):
                return "native quantize supports dense text models without a reference: \(why). "
                    + "Pass --reference <a 4-bit build of this model> to learn the per-module "
                    + "quantized set (MoE / vision / Gemma), or use a pre-quantized build."
            case .noWeights(let u): return "no .safetensors weights in \(u.path)"
            case .noReferenceScales(let u):
                return "reference checkpoint \(u.path) has no `.scales` tensors - it is not a "
                    + "quantized build, so there is no module set to learn."
            case .nonDivisibleWeight(let name, let dim, let groupSize):
                return "weight '\(name)' inner dim \(dim) is not divisible by group size "
                    + "\(groupSize); KrillLM's loader quantizes every Linear uniformly, so "
                    + "it cannot load a checkpoint with this layer left dense. Use a group "
                    + "size that divides \(dim)."
            case .expertGroupUnsupported(let group):
                return "this checkpoint has Mixture-of-Experts (stacked 3-D experts), which the "
                    + "MoE runtime reconstructs as AFFINE at the top-level group \(group); affine "
                    + "only supports group 32/64/128, so a float top-level whose group is \(group) "
                    + "(e.g. nvfp4 = group 16) cannot produce a loadable MoE checkpoint. Quantize "
                    + "MoE with --mode affine (group 64), or --mode mxfp4/mxfp8 (group 32)."
            }
        }
    }

    /// Vision/audio projector module substrings auto-protected at higher precision
    /// in reference mode (matches `tools/requant_gemma4_nvfp4.py`): nvfp4 on the
    /// patch / media embedding attenuates the red channel enough to misread
    /// red-heavy colors (red->brown, yellow->olive, magenta->purple) while text is
    /// unaffected. These tensors are tiny, so 8-bit costs almost nothing.
    public static let visionProtectSubstrings = [
        "vision_embedder.patch_dense",
        "embed_vision.embedding_projection",
        "embed_audio.embedding_projection",
    ]

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

    /// Per-module quantization parameters, resolved against the protect list and
    /// the tensor's rank. Protected modules win; stacked 3-D experts are forced to
    /// affine (the MoE runtime is affine-only); everything else uses the top-level
    /// format.
    private struct PerModule { let bits: Int; let groupSize: Int; let mode: String }

    /// Quantize `sourceDir` into `outputDir`. Returns the number of tensors quantized.
    ///
    /// - Parameters:
    ///   - referenceDir: a 4-bit checkpoint of the same model whose `.scales`
    ///     tensors define exactly which modules to quantize. `nil` => dense pass.
    ///   - protect: module-path substrings to quantize at the protect precision.
    ///   - protectBits/protectGroupSize/protectMode: the protect precision.
    ///   - autoProtectVision: in reference mode, also protect the known vision/
    ///     audio projector modules (the color-fidelity fix). No-op if absent.
    @discardableResult
    public static func quantize(
        sourceDir: URL, outputDir: URL,
        bits: Int, groupSize: Int, mode: String = "affine", dtype: String = "fp16",
        referenceDir: URL? = nil,
        protect: [String] = [],
        protectBits: Int = 8, protectGroupSize: Int = 64, protectMode: String = "affine",
        autoProtectVision: Bool = true,
        log: (String) -> Void = { _ in }
    ) throws -> Int {
        let fm = FileManager.default

        // 1. Read + validate the source config.
        let configURL = sourceDir.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: configURL) else {
            throw QuantizeError.missingConfig(configURL)
        }
        guard var config = (try? JSONSerialization.jsonObject(with: configData))
            as? [String: Any] else {
            throw QuantizeError.invalidConfig(configURL)
        }

        // 2. Decide the quantized-module set. With a reference we trust its proven
        //    coverage and skip the family gate; without one we run the dense pass
        //    and reject families that need per-module handling.
        let quantSet: Set<String>?
        if let referenceDir {
            let mods = try referenceQuantModules(referenceDir)
            guard !mods.isEmpty else { throw QuantizeError.noReferenceScales(referenceDir) }
            log("learned \(mods.count) quantized modules from reference \(referenceDir.lastPathComponent)")
            quantSet = mods
        } else {
            try assertSupportedDense(config)
            quantSet = nil
        }

        // Build the effective protect list (vision/audio projectors auto-added in
        // reference mode; harmless substrings that just won't match if absent).
        var protectList = protect
        if quantSet != nil, autoProtectVision {
            for s in visionProtectSubstrings where !protectList.contains(s) {
                protectList.append(s)
            }
            log("auto-protecting vision/audio projectors @ \(protectBits)b \(protectMode) "
                + "(disable with --no-protect-vision)")
        }

        // 3. Load the flat source weight dict (lazy mmap; merges shards).
        var weights = try loadWeightArrays(from: sourceDir)
        guard !weights.isEmpty else { throw QuantizeError.noWeights(sourceDir) }

        // Top-level format. Float formats only support one (bits, groupSize) each;
        // honor that regardless of the passed flags so `--mode nvfp4` just works.
        let storeDType = storageDType(dtype)
        let (topBits, topGroup) = effectiveParams(mode: mode, bits: bits, groupSize: groupSize)
        let topMode = mode.lowercased()
        if topBits != bits || topGroup != groupSize {
            log("mode \(mode) requires \(topBits)-bit / group \(topGroup); overriding the passed bits/group")
        }

        // Resolve per-module precision. The 3-D expert rule is checked FIRST and
        // wins over --protect: the MoE runtime reconstructs born-quantized experts
        // at the top-level group/bits affine no matter what, so it cannot honor a
        // protect precision on an expert - a `--protect down_proj` that matched the
        // stacked `switch_mlp.down_proj` expert would otherwise emit an unloadable
        // checkpoint.
        func perModule(_ module: String, ndim: Int) -> PerModule {
            if ndim == 3 {
                // Stacked experts are born-quantized in the MoE runtime
                // (MoEQuantizedSwitchedLinear): it reconstructs them as AFFINE at
                // the config's TOP-LEVEL (bits, groupSize) and never consults a
                // per-module override. Mirror that exactly so the emitted scales
                // shape [E, O, I/topGroup] matches what the loader pre-allocates -
                // including under a float top-level (nvfp4 => affine group 16),
                // where the experts cannot be a float format anyway.
                return PerModule(bits: topBits, groupSize: topGroup, mode: "affine")
            }
            if protectList.contains(where: { module.contains($0) }) {
                // Normalize the protect format the same way the top level is: the
                // float formats only support one (bits, groupSize) each, so
                // `--protect-mode mxfp8` works without the caller also passing
                // group 32.
                let (pb, pg) = effectiveParams(
                    mode: protectMode, bits: protectBits, groupSize: protectGroupSize)
                return PerModule(bits: pb, groupSize: pg, mode: protectMode.lowercased())
            }
            return PerModule(bits: topBits, groupSize: topGroup, mode: topMode)
        }

        // Does this `.weight` get quantized? Reference mode: membership in the set.
        // Dense mode: a 2-D weight (Linear/Embedding leaf the loader would quantize).
        func shouldQuantize(module: String, ndim: Int) -> Bool {
            if let quantSet { return quantSet.contains(module) }
            return ndim == 2
        }

        // 4. Quantize the selected `*.weight` tensors; pass the rest through.
        //    Evaluate + drop each source tensor as we go so peak memory stays near
        //    (one source tensor + the growing quantized set), not (everything) at
        //    once.
        var out: [String: MLXArray] = [:]
        var overrides: [String: [String: Any]] = [:]
        var quantizedCount = 0
        for name in weights.keys.sorted() {
            guard let w = weights[name] else { continue }
            let isWeight = name.hasSuffix(".weight")
            let module = isWeight ? String(name.dropLast(".weight".count)) : name

            if isWeight, shouldQuantize(module: module, ndim: w.ndim) {
                // Stacked experts are reconstructed affine at the top-level group;
                // affine supports only 32/64/128, so a float top-level whose group
                // is smaller (nvfp4 = 16) cannot yield a loadable MoE checkpoint.
                if w.ndim == 3, ![32, 64, 128].contains(topGroup) {
                    throw QuantizeError.expertGroupUnsupported(group: topGroup)
                }
                let pm = perModule(module, ndim: w.ndim)
                // Inner (last) dim must be group-divisible or the uniform load cannot
                // honor it; fail loudly instead of emitting a mis-loadable checkpoint.
                let innerDim = w.dim(w.ndim - 1)
                if innerDim % pm.groupSize != 0 {
                    throw QuantizeError.nonDivisibleWeight(
                        name: name, dim: innerDim, groupSize: pm.groupSize)
                }
                let qmode = mlxQuantizationMode(pm.mode)
                // affine: quantize AT the storage precision (mlx_lm.convert casts to
                // fp16 first) so scales/biases are byte-identical to an mlx-community
                // build. Float formats (nvfp4/...) quantize from the SOURCE dtype.
                let wIn = qmode == .affine ? w.asType(storeDType) : w
                let (wq, scales, biases) = MLX.quantized(
                    wIn, groupSize: pm.groupSize, bits: pm.bits, mode: qmode)
                out[name] = wq
                out[module + ".scales"] = scales
                var realized: [MLXArray] = [wq, scales]
                if let biases {
                    out[module + ".biases"] = biases
                    realized.append(biases)
                }
                MLX.eval(realized)        // realize now so the source input can be freed

                // Record a per-module override whenever a regular (2-D) module's
                // precision differs from the top-level block - these the loader
                // resolves via QuantizationConfig.effective(for:) (protected
                // projectors, an 8-bit router gate, ...). Stacked 3-D experts are
                // born-quantized affine at the top-level group and the loader does
                // NOT read an override for them, so emitting one would be dead
                // config; skip it.
                if w.ndim != 3,
                   pm.bits != topBits || pm.groupSize != topGroup || pm.mode != topMode {
                    overrides[module] = ["group_size": pm.groupSize, "bits": pm.bits, "mode": pm.mode]
                }
                quantizedCount += 1
            } else {
                // Pass-through (norms, 1-D biases, the float-kept vision tower, etc.):
                // store float tensors at the target precision; leave non-float as-is.
                let isFloat = [DType.float16, .bfloat16, .float32].contains(w.dtype)
                out[name] = isFloat ? w.asType(storeDType) : w
            }
            weights[name] = nil           // release the source reference
        }
        log("quantized \(quantizedCount) tensors (\(topBits)-bit, group \(topGroup), \(mode), \(dtype))"
            + (overrides.isEmpty ? "" : "; \(overrides.count) per-module overrides"))

        // 5. Write a single safetensors shard (the loader globs *.safetensors).
        try fm.createDirectory(at: outputDir, withIntermediateDirectories: true)
        let weightsURL = outputDir.appendingPathComponent("model.safetensors")
        try save(arrays: out, metadata: ["format": "mlx"], url: weightsURL)

        // 6. Emit config.json with the quantization block the loader reads
        //    (`quantization`), plus `quantization_config` for HF-tool parity. The
        //    per-module overrides are keyed by module path - the exact form
        //    QuantizationConfig.effective(for:) resolves.
        var qblock: [String: Any] = ["group_size": topGroup, "bits": topBits]
        if topMode != "affine" { qblock["mode"] = topMode }
        for (path, mq) in overrides { qblock[path] = mq }
        config["quantization"] = qblock
        config["quantization_config"] = qblock
        let outConfig = try JSONSerialization.data(
            withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        try outConfig.write(to: outputDir.appendingPathComponent("config.json"))

        // 7. Copy tokenizer + auxiliary files the loader/tokenizer need.
        try copyAuxiliaryFiles(from: sourceDir, to: outputDir)

        return quantizedCount
    }

    /// Learn the quantized-module set from a reference 4-bit checkpoint: every
    /// `<module>.scales` tensor means `<module>.weight` is quantized. Reads the
    /// safetensors index (no tensor load) when present, else lazily opens each
    /// shard and reads its keys (mmap, nothing materialized).
    private static func referenceQuantModules(_ dir: URL) throws -> Set<String> {
        let fm = FileManager.default
        var keys: [String] = []
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: indexURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let map = json["weight_map"] as? [String: Any] {
            keys = Array(map.keys)
        } else {
            let contents = try fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            for f in contents where f.pathExtension == "safetensors" {
                keys.append(contentsOf: try loadArrays(url: f).keys)
            }
        }
        let suffix = ".scales"
        var mods = Set<String>()
        for k in keys where k.hasSuffix(suffix) { mods.insert(String(k.dropLast(suffix.count))) }
        return mods
    }

    /// Reject checkpoints whose quantized-layer set is NOT "all 2-D divisible
    /// weights" when no reference is supplied: Mixture-of-Experts (stacked 3-D
    /// expert tensors), any model with a vision/multimodal tower, and Gemma. With a
    /// reference these are all supported, so this only runs on the dense pass.
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
            "processor_config.json",
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
