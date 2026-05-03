import Foundation
import MLX
import MLXNN

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

/// Load weights from a directory and apply them to a model.
///
/// Handles quantization: if the config specifies quantization parameters,
/// the model's Linear layers are quantized before weight loading so that
/// the QuantizedLinear modules accept the packed weight format.
public func loadWeights(
    into model: Module,
    from directory: URL,
    quantization: QuantizationConfig? = nil
) throws {
    // Quantize the model if needed (converts Linear -> QuantizedLinear)
    if let q = quantization {
        quantize(
            model: model,
            groupSize: q.groupSize,
            bits: q.bits
        )
    }

    let flatWeights = try loadWeightArrays(from: directory)

    // Use mlx-swift's built-in unflattened() to convert flat dict -> ModuleParameters
    let tuples = flatWeights.map { ($0.key, $0.value) }
    let nested = ModuleParameters.unflattened(tuples)
    try model.update(parameters: nested, verify: .noUnusedKeys)
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
