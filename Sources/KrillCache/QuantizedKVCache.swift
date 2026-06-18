import MLX

/// Quantized KV cache storing keys and values in uint8 with per-head scale factors.
///
/// Provides ~2x memory reduction at long context lengths compared to fp16.
/// Dequantizes on the fly during attention computation.
///
/// Quantization scheme: asymmetric per-head min-max scaling into [0, 255].
///   quantized = clip(round((value - min) / scale), 0, 255)   // uint8
///   scale = (max - min) / 255
///   zero_point = min                                         // dequant offset
///   dequantized = quantized * scale + zero_point
///
/// uint8 storage avoids the int8-cast wraparound bug that produced a value range of
/// [-128, -56) for normalized inputs > 127 (200 -> -56 in 2's complement) and
/// corrupted any KV value above the quarter-range mark.
public final class QuantizedKVCache: KVCacheProtocol, @unchecked Sendable {
    private var _keys: MLXArray?       // uint8: [B, numKVHeads, seqLen, headDim]
    private var _values: MLXArray?     // uint8: [B, numKVHeads, seqLen, headDim]
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

    /// Return dequantized full K/V (fp16) for protocol compatibility.
    ///
    /// Used by Gemma4 KV-sharing: a shared layer reads its donor's accumulated
    /// K/V directly. With int8 storage the donor is quantized, so we dequantize
    /// here to hand back fp16 tensors at attention time.
    public func snapshot() -> (keys: MLXArray, values: MLXArray)? {
        guard let qK = _keys, let qV = _values,
              let kS = _keyScales, let kZ = _keyZeros,
              let vS = _valScales, let vZ = _valZeros else { return nil }
        let k = dequantize(qK, scale: kS, zero: kZ)
        let v = dequantize(qV, scale: vS, zero: vZ)
        return (k, v)
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

    /// Return the raw quantized state for prefix-cache persistence.
    ///
    /// Unlike `snapshot()` (which dequantizes for attention compatibility),
    /// this preserves the uint8 tensors with their fp16 scales/zeros so the
    /// cache can be re-served without an extra dequant→requant round trip.
    public func quantizedSnapshot() -> QuantizedKVSnapshot? {
        guard let qK = _keys, let qV = _values,
              let kS = _keyScales, let kZ = _keyZeros,
              let vS = _valScales, let vZ = _valZeros else { return nil }
        return QuantizedKVSnapshot(
            keys: qK, values: qV,
            keyScales: kS, keyZeros: kZ,
            valueScales: vS, valueZeros: vZ
        )
    }

    /// Restore previously quantized state directly into this cache.
    ///
    /// Used by the prefix-cache replay path; the caller is responsible for
    /// truncating to `prompt.count - 1` and re-forwarding the last token, so
    /// the dependency graph behaves the same as a fresh prefill.
    public func restoreQuantized(_ snap: QuantizedKVSnapshot) {
        _keys = snap.keys
        _values = snap.values
        _keyScales = snap.keyScales
        _keyZeros = snap.keyZeros
        _valScales = snap.valueScales
        _valZeros = snap.valueZeros
    }

    /// Truncate cached state along the sequence axis.
    ///
    /// All six tensors share axis 2 as the sequence dimension, matching the
    /// layout produced by `update`.
    public func truncate(to sequenceLength: Int) {
        guard let qK = _keys else { return }
        let currentLen = qK.dim(2)
        guard sequenceLength < currentLen, sequenceLength >= 0 else { return }
        _keys      = qK[0..., 0..., 0 ..< sequenceLength, 0...]
        _values    = _values?[0..., 0..., 0 ..< sequenceLength, 0...]
        _keyScales = _keyScales?[0..., 0..., 0 ..< sequenceLength, 0...]
        _keyZeros  = _keyZeros?[0..., 0..., 0 ..< sequenceLength, 0...]
        _valScales = _valScales?[0..., 0..., 0 ..< sequenceLength, 0...]
        _valZeros  = _valZeros?[0..., 0..., 0 ..< sequenceLength, 0...]
    }
}

/// Per-layer snapshot of quantized KV state — six tensors that together
/// describe the int8-encoded sequence and its dequantization parameters.
public struct QuantizedKVSnapshot {
    public let keys: MLXArray        // uint8 [B, H, S, D]
    public let values: MLXArray      // uint8 [B, H, S, D]
    public let keyScales: MLXArray   // fp16  [B, H, S, 1]
    public let keyZeros: MLXArray    // fp16  [B, H, S, 1]
    public let valueScales: MLXArray // fp16  [B, H, S, 1]
    public let valueZeros: MLXArray  // fp16  [B, H, S, 1]

    public init(
        keys: MLXArray, values: MLXArray,
        keyScales: MLXArray, keyZeros: MLXArray,
        valueScales: MLXArray, valueZeros: MLXArray
    ) {
        self.keys = keys
        self.values = values
        self.keyScales = keyScales
        self.keyZeros = keyZeros
        self.valueScales = valueScales
        self.valueZeros = valueZeros
    }

    /// Number of cached tokens (sequence axis).
    public var sequenceLength: Int { keys.dim(2) }
}

// MARK: - Quantization Helpers

extension QuantizedKVCache {
    /// Quantize a tensor to uint8 with per-head min-max scaling.
    /// Input shape: [B, numHeads, seqLen, headDim]
    /// Returns: (quantized uint8, scale fp16, zero fp16)
    private func quantizePerHead(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // Compute per-position min/max across headDim
        let minVal = MLX.min(x, axis: -1, keepDims: true)  // [B, H, S, 1]
        let maxVal = MLX.max(x, axis: -1, keepDims: true)  // [B, H, S, 1]
        let scale = (maxVal - minVal) / MLXArray(Float(255.0))
        // Avoid division by zero
        let safeScale = MLX.maximum(scale, MLXArray(Float(1e-8)))

        // Quantize: (x - min) / scale -> [0, 255], stored as uint8.
        // Clip before cast so floating-point round-trip noise can't push values
        // outside the representable range.
        let normalized = (x - minVal) / safeScale
        let clipped = MLX.clip(normalized, min: MLXArray(Float(0.0)), max: MLXArray(Float(255.0)))
        let quantized = clipped.asType(.uint8)

        return (quantized, safeScale.asType(.float16), minVal.asType(.float16))
    }

    /// Dequantize uint8 back to fp16.
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
