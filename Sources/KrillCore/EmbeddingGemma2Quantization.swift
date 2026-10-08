import Foundation
import MLX
import MLXNN

// MARK: - EmbeddingGemma 2: quantized checkpoints
//
// `krill quantize google/embeddinggemma-2 --mode mxfp8|nvfp4 ...` writes, for
// every quantized Linear / Embedding `<module>`, three tensors in place of
// `<module>.weight`:
//
//   <module>.weight   packed uint32
//   <module>.scales   uint8 (mxfp8 / nvfp4) or float (affine)
//   <module>.biases   only for affine
//
// plus a `quantization` block in config.json (top-level `group_size` / `bits` /
// `mode`, and a per-module override keyed by the FULL checkpoint path for any
// module whose precision differs from the top level). The checkpoint, not a
// hand-kept list, decides which modules are quantized: a module is quantized
// exactly when `<module>.scales` exists, and its precision is whatever
// `config.quantization` says for that path. Everything is cross-checked, so a
// checkpoint and a config that disagree fail the load instead of producing
// plausible garbage:
//
//   * `.scales` tensors but no `quantization` block        -> throw
//   * a `quantization` block but no `.scales` tensor at all -> throw
//   * a `.scales` without its `.weight`                     -> throw
//   * a per-module override that names no quantized module  -> throw
//   * a `.scales` on a module that is not a Linear/Embedding, an unexpected
//     `.biases`, or any shape that does not match the declared (group, bits,
//     mode)                                                 -> caught by the
//     towers' / text model's strict `update(parameters:verify:)`

public enum EG2QuantizationError: Error, CustomStringConvertible, Equatable {
    case scalesWithoutConfig(example: String)
    case configWithoutScales
    case scalesWithoutWeight(String)
    case staleOverride(String)
    case invalidConfig(String)

    public var description: String {
        switch self {
        case .scalesWithoutConfig(let e):
            return "EmbeddingGemma2 checkpoint has quantized tensors (e.g. '\(e)') but config.json has no "
                + "`quantization` block, so their group size / bits / mode are unknown"
        case .configWithoutScales:
            return "config.json declares `quantization` but the checkpoint has no `.scales` tensor: the "
                + "weights are not quantized (wrong config or wrong weights)"
        case .scalesWithoutWeight(let m):
            return "EmbeddingGemma2 checkpoint has '\(m).scales' but no '\(m).weight'"
        case .staleOverride(let m):
            return "config.json `quantization` has a per-module entry for '\(m)' but the checkpoint has no "
                + "'\(m).scales' (a stale or mis-keyed override; keys are full checkpoint paths)"
        case .invalidConfig(let why):
            return "config.json `quantization` block is invalid: \(why)"
        }
    }
}

/// The `quantization` block of an EmbeddingGemma 2 `config.json`, or nil when
/// the checkpoint is dense.
func eg2QuantizationConfig(configData: Data) throws -> QuantizationConfig? {
    guard let root = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
          let block = root["quantization"], !(block is NSNull) else { return nil }
    guard block is [String: Any] else { throw EG2QuantizationError.invalidConfig("not an object") }
    do {
        return try JSONDecoder().decode(
            QuantizationConfig.self, from: JSONSerialization.data(withJSONObject: block))
    } catch {
        throw EG2QuantizationError.invalidConfig("\(error)")
    }
}

/// Module paths (checkpoint keys minus `.scales`) of every quantized module in
/// `keys`. Throws if one has no `.weight` next to it.
func eg2QuantizedModules<S: Sequence>(in keys: S) throws -> Set<String> where S.Element == String {
    let all = Set(keys)
    var mods = Set<String>()
    for k in all where k.hasSuffix(".scales") {
        let m = String(k.dropLast(".scales".count))
        guard all.contains(m + ".weight") else { throw EG2QuantizationError.scalesWithoutWeight(m) }
        mods.insert(m)
    }
    return mods
}

/// Whole-checkpoint consistency between the quantized module set and the
/// config (see the header). Call once, with EVERY checkpoint key.
func eg2ValidateQuantization(modules: Set<String>, config: QuantizationConfig?) throws {
    guard let config else {
        if let e = modules.sorted().first { throw EG2QuantizationError.scalesWithoutConfig(example: e + ".scales") }
        return
    }
    if modules.isEmpty { throw EG2QuantizationError.configWithoutScales }
    for path in config.moduleOverrides.keys.sorted() where !modules.contains(path) {
        throw EG2QuantizationError.staleOverride(path)
    }
}

/// Swap the Linear / Embedding leaves of `module` that the checkpoint ships
/// quantized for their quantized counterparts, at the precision the config
/// declares for that path. `prefix` maps the module's own paths onto checkpoint
/// paths (`"language_model."` for the text model, `""` for the towers, whose
/// parameter keys already equal the checkpoint's). Returns the number swapped.
@discardableResult
func eg2ApplyQuantization(
    to module: Module, prefix: String, modules: Set<String>, config: QuantizationConfig?
) -> Int {
    guard let config, !modules.isEmpty else { return 0 }
    var n = 0
    quantize(model: module) { path, _ in
        let full = prefix + path
        guard modules.contains(full) else { return nil }
        let e = config.effective(for: full)
        n += 1
        return (e.groupSize, e.bits, mlxQuantizationMode(e.mode))
    }
    return n
}

/// Cast every FLOATING parameter to `dtype`. Quantized weights (uint32) and
/// float-format scales (uint8) must keep their dtype; affine scales / biases
/// are float and follow the compute dtype.
func eg2CastFloatParameters(_ module: Module, to dtype: DType) {
    module.update(parameters: module.parameters().mapValues { $0.dtype.isFloatingPoint ? $0.asType(dtype) : $0 })
}
