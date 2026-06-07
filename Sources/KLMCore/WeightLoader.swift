import Foundation
import MLX
import MLXNN

/// Map a config quantization `mode` string to MLX's `QuantizationMode`.
///
/// `"affine"` (the historical default) and any unrecognized value fall back
/// to `.affine`, so existing checkpoints are unaffected. `"nvfp4"` /
/// `"mxfp4"` / `"mxfp8"` select the 4-bit-float (and 8-bit-float) formats.
/// nvfp4 in particular preserves outlier weights that affine int4 clips, at
/// the same 4-bit speed (verified end-to-end on Gemma 4 12B MMLU).
func mlxQuantizationMode(_ mode: String) -> QuantizationMode {
    switch mode.lowercased() {
    case "nvfp4": return .nvfp4
    case "mxfp4": return .mxfp4
    case "mxfp8": return .mxfp8
    default: return .affine
    }
}

// MARK: - Weight Loading

/// Load model weights from a directory containing safetensors files.
///
/// Handles both single-file and sharded (multi-file) weight formats:
///   - `model.safetensors` (single file)
///   - `model-00001-of-00004.safetensors` etc. (sharded)
///
/// Returns a flat dictionary of weight name -> MLXArray.
public func loadWeightArrays(from directory: URL) throws -> [String: MLXArray] {
    let fm = FileManager.default
    let contents = try fm.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: nil)
    let safetensorsFiles = contents
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    guard !safetensorsFiles.isEmpty else {
        throw WeightLoadError.noSafetensorsFiles(directory)
    }

    var allWeights: [String: MLXArray] = [:]
    for file in safetensorsFiles {
        let fileWeights = try loadArrays(url: file)
        for (key, value) in fileWeights {
            allWeights[key] = value
        }
    }
    return allWeights
}

/// True when a checkpoint exports a base model (backbone tensors named
/// `embed_tokens.*`/`layers.N.*`/`norm.*` with no `model.` prefix and no
/// lm_head), as sentence-embedding decoders like intfloat/e5-mistral-7b-instruct
/// and Salesforce/SFR-Embedding-Mistral do (`MistralModel`, not
/// `MistralForCausalLM`). The causal modules used as embedder backbones nest
/// the backbone under `model.`, so such keys must be prefixed. Normal
/// `*ForCausalLM` checkpoints already carry the `model.` prefix and are left
/// untouched.
public func baseModelNeedsModelPrefix<C: Collection>(_ keys: C) -> Bool
where C.Element == String {
    let hasNestedBackbone = keys.contains { $0.hasPrefix("model.") }
    let hasBareBackbone = keys.contains { $0.hasPrefix("layers.") }
    return hasBareBackbone && !hasNestedBackbone
}

/// Load weights from a directory and apply them to a model.
///
/// Handles quantization: if the config specifies quantization parameters,
/// the model's Linear layers are quantized before weight loading so that
/// the QuantizedLinear modules accept the packed weight format.
///
/// Also handles tied embeddings: if `lm_head.weight` is missing but
/// `model.embed_tokens.weight` exists, copies it as the lm_head weight.
///
/// - Parameter keyPrefix: Optional prefix to strip from weight keys (e.g., "language_model."
///   for Gemma 4 which nests text weights under that prefix).
/// - Parameter tieWordEmbeddings: When true, the consumer model has NO
///   separate `lm_head` module (it reuses `embed_tokens` via
///   `asLinear`). The loader will skip the embed_tokens -> lm_head
///   key duplication that historically handled Llama 3.2 1B and
///   similar tied checkpoints, so we do not try to assign weights
///   into a property that does not exist. Defaults to false to keep
///   prior callers unchanged.
/// - Parameter keyRewrite: Optional caller-defined hook to mutate the
///   flat `[key: MLXArray]` weights dict AFTER prefix stripping +
///   tied-embedding duplication and BEFORE the `model.update` call.
///   Used by `loadQwen3MoE` to unpack mlx-community's stacked
///   `mlp.switch_mlp.{proj}` expert tensors into the per-expert keys
///   our `Qwen3MoESparseMLP` allocates. Defaults to nil (no rewrite).
/// - Parameter strictVerify: When true, the `model.update` call uses
///   `.allModelKeysSet, .shapeMismatch, .noUnusedKeys` so loader bugs
///   in this class (silent-dropped checkpoint keys, missing model
///   params, shape mismatches) crash at load time instead of producing
///   garbage tokens at decode time. The historical default
///   (`verify: []`) is preserved for callers that haven't audited their
///   key conventions; new families should adopt strict verify.
public func loadWeights(
    into model: Module,
    from directory: URL,
    quantization: QuantizationConfig? = nil,
    keyPrefix: String? = nil,
    tieWordEmbeddings: Bool = false,
    keyRewrite: ((inout [String: MLXArray]) -> Void)? = nil,
    strictVerify: Bool = false
) throws {
    // Quantize the model if needed (converts Linear -> QuantizedLinear).
    // The mixed-precision branch routes through MLX's per-layer filter
    // overload so per-module (groupSize, bits) overrides land on the
    // right layers - Qwen3-Coder ships 4-bit base + 8-bit MoE gates,
    // and loading the gates as 4-bit crashes `quantized_matmul` with a
    // scales-shape mismatch.
    if let q = quantization {
        if q.moduleOverrides.isEmpty {
            quantize(model: model, groupSize: q.groupSize, bits: q.bits,
                     mode: mlxQuantizationMode(q.mode))
        } else {
            quantize(model: model) { path, _ in
                let eff = q.effective(for: path)
                return (eff.groupSize, eff.bits, mlxQuantizationMode(eff.mode))
            }
        }
    }

    var flatWeights = try loadWeightArrays(from: directory)

    // Strip key prefix if specified (e.g., "language_model." for
    // Gemma 4, "roberta." for BGE Reranker). Keys that DO start
    // with the prefix are renamed (prefix removed); keys that do
    // NOT start with the prefix are preserved unchanged. The
    // preservation matters for cross-encoder rerankers: the
    // backbone weights are under `roberta.*` but the
    // classification head lives at top-level `classifier.*` and
    // would otherwise be silently dropped, leaving the classifier
    // at its random initialization (cause of past bug where
    // reranker scores converged near zero).
    if let prefix = keyPrefix {
        var rebuilt: [String: MLXArray] = [:]
        var anyStripped = false
        for (key, value) in flatWeights {
            if key.hasPrefix(prefix) {
                rebuilt[String(key.dropFirst(prefix.count))] = value
                anyStripped = true
            } else {
                rebuilt[key] = value
            }
        }
        if anyStripped {
            flatWeights = rebuilt
        }
    }

    // Base-model checkpoints name the backbone with no `model.` prefix and
    // ship no lm_head: a sentence-embedding decoder like
    // intfloat/e5-mistral-7b-instruct exports `MistralModel` (tensors
    // `embed_tokens.*`, `layers.N.*`, `norm.*`), not `MistralForCausalLM`
    // (which nests them under `model.` and adds `lm_head.*`). The causal
    // modules used as embedder backbones expect the nested layout, so a bare
    // checkpoint otherwise drops every backbone weight (lax verify) or fails
    // (strict) - the cause of e5-mistral's earlier garbage embeddings. Prefix
    // the bare backbone keys; the lm_head tie below then fills the unused head
    // so strict verify still passes. Detected by bare `layers.` keys with no
    // `model.` keys, so normal ForCausalLM checkpoints are untouched.
    if baseModelNeedsModelPrefix(flatWeights.keys) {
        var rebuilt: [String: MLXArray] = [:]
        for (key, value) in flatWeights { rebuilt["model." + key] = value }
        flatWeights = rebuilt
    }

    // Handle tied embeddings: copy embed_tokens weights to lm_head if
    // missing. Skipped when the consumer model declared
    // `tieWordEmbeddings` -> it has no `lm_head` property, and
    // assigning the duplicated weights into a nonexistent Optional
    // module errors with "none not compatible with ...".
    let hasLmHead = flatWeights.keys.contains { $0.hasPrefix("lm_head.") }
    if !hasLmHead && !tieWordEmbeddings {
        // Copy all embed_tokens keys to lm_head (handles weight, scales, biases)
        let embedKeys = flatWeights.keys.filter { $0.hasPrefix("model.embed_tokens.") }
        for key in embedKeys {
            let lmHeadKey = key.replacingOccurrences(of: "model.embed_tokens.", with: "lm_head.")
            flatWeights[lmHeadKey] = flatWeights[key]
        }
    }

    // Caller hook: rewrite keys after prefix-strip and tied-embed
    // duplication, before update. Used by Qwen3-MoE to slice
    // mlx-community's stacked switch_mlp tensors into the per-expert
    // keys the model allocates.
    keyRewrite?(&flatWeights)

    // Use mlx-swift's built-in unflattened() to convert flat dict -> ModuleParameters
    let tuples = flatWeights.map { ($0.key, $0.value) }
    let nested = ModuleParameters.unflattened(tuples)

    // Strict mode catches three classes of loader bug at load time:
    //   .allModelKeysSet -> a module param the checkpoint did not cover
    //                       (would leave that weight at random init)
    //   .shapeMismatch   -> a key matched but the shape disagrees
    //   .noUnusedKeys    -> a checkpoint key with no matching param
    //                       (this is what hid the original #75: the
    //                       silent-drop of mlx-community's stacked
    //                       switch_mlp keys was invisible until decode)
    // Lax mode preserves the historical `verify: []` for callers that
    // have not been audited.
    let verify: Module.VerifyUpdate = strictVerify
        ? [.allModelKeysSet, .shapeMismatch, .noUnusedKeys]
        : []
    try model.update(parameters: nested, verify: verify)
}

// MARK: - Errors

public enum WeightLoadError: Error, CustomStringConvertible {
    case noSafetensorsFiles(URL)

    public var description: String {
        switch self {
        case .noSafetensorsFiles(let dir):
            return "No .safetensors files found in \(dir.path)"
        }
    }
}
