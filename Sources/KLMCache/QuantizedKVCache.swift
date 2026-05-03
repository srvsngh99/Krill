import MLX

/// Quantized KV cache storing keys and values in int8 with per-head scale factors.
///
/// Provides ~2x memory reduction at long context lengths compared to fp16.
/// Dequantizes on the fly during attention computation.
///
/// Quantization scheme: asymmetric per-head min-max scaling.
///   quantized = round((value - min) / (max - min) * 255) - 128
///   scale = (max - min) / 255
///   zero_point = min
public final class QuantizedKVCache: @unchecked Sendable {
    private var _keys: MLXArray?       // int8: [B, numKVHeads, seqLen, headDim]
    private var _values: MLXArray?     // int8: [B, numKVHeads, seqLen, headDim]
    private var _keyScales: MLXArray?  // fp16: [B, numKVHeads, seqLen, 1]
    private var _keyZeros: MLXArray?   // fp16: [B, numKVHeads, seqLen, 1]
    private var _valScales: MLXArray?  // fp16: [B, numKVHeads, seqLen, 1]
    private var _valZeros: MLXArray?   // fp16: [B, numKVHeads, seqLen, 1]

    public init() {}

    /// Number of tokens currently cached.
    public var sequenceLength: Int {
        _keys?.dim(2) ?? 0
    }

    /// Approximate memory usage in bytes.
    public var memorySizeBytes: Int {
        guard let keys = _keys else { return 0 }
        let seqLen = keys.dim(2)
        let numHeads = keys.dim(1)
        let headDim = keys.dim(3)
        // int8 KV + fp16 scales/zeros
        let kvBytes = 2 * seqLen * numHeads * headDim * 1  // int8 = 1 byte
        let metaBytes = 4 * seqLen * numHeads * 1 * 2       // fp16 = 2 bytes, 4 arrays
        return kvBytes + metaBytes
    }

    /// Quantize and append new key/value tensors, return dequantized full K/V for attention.
    ///
    /// Input: fp16 [B, numKVHeads, seqLen, headDim]
    /// Stored: int8 quantized with per-head scale factors
    /// Output: dequantized fp16 full sequence [B, numKVHeads, totalSeqLen, headDim]
    public func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        // Quantize new K/V
        let (qK, kScale, kZero) = quantizePerHead(newK)
        let (qV, vScale, vZero) = quantizePerHead(newV)

        // Append to stored
        if let existingK = _keys {
            _keys = concatenated([existingK, qK], axis: 2)
            _values = concatenated([_values!, qV], axis: 2)
            _keyScales = concatenated([_keyScales!, kScale], axis: 2)
            _keyZeros = concatenated([_keyZeros!, kZero], axis: 2)
            _valScales = concatenated([_valScales!, vScale], axis: 2)
            _valZeros = concatenated([_valZeros!, vZero], axis: 2)
        } else {
            _keys = qK
            _values = qV
            _keyScales = kScale
            _keyZeros = kZero
            _valScales = vScale
            _valZeros = vZero
        }

        // Dequantize full sequence for attention
        let fullK = dequantize(_keys!, scale: _keyScales!, zero: _keyZeros!)
        let fullV = dequantize(_values!, scale: _valScales!, zero: _valZeros!)
        return (fullK, fullV)
    }

    /// Discard all cached state.
    public func reset() {
        _keys = nil
        _values = nil
        _keyScales = nil
        _keyZeros = nil
        _valScales = nil
        _valZeros = nil
    }
}

// MARK: - Quantization Helpers

extension QuantizedKVCache {
    /// Quantize a tensor to int8 with per-head min-max scaling.
    /// Input shape: [B, numHeads, seqLen, headDim]
    /// Returns: (quantized int8, scale fp16, zero fp16)
    private func quantizePerHead(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // Compute per-position min/max across headDim
        let minVal = MLX.min(x, axis: -1, keepDims: true)  // [B, H, S, 1]
        let maxVal = MLX.max(x, axis: -1, keepDims: true)  // [B, H, S, 1]
        let scale = (maxVal - minVal) / MLXArray(Float(255.0))
        // Avoid division by zero
        let safeScale = MLX.maximum(scale, MLXArray(Float(1e-8)))

        // Quantize: (x - min) / scale -> [0, 255] -> [-128, 127]
        let normalized = (x - minVal) / safeScale
        let quantized = normalized.asType(.int8)

        return (quantized, safeScale.asType(.float16), minVal.asType(.float16))
    }

    /// Dequantize int8 back to fp16.
    private func dequantize(_ q: MLXArray, scale: MLXArray, zero: MLXArray) -> MLXArray {
        // q * scale + zero
        let floatQ = q.asType(.float16)
        return floatQ * scale + zero
    }
}

/// Create an array of quantized KV caches, one per transformer layer.
public func makeQuantizedKVCaches(numLayers: Int) -> [QuantizedKVCache] {
    (0 ..< numLayers).map { _ in QuantizedKVCache() }
}
