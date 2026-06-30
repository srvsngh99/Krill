import Foundation
import MLX
import MLXNN

// MARK: - Shared MoE SwitchGLU (gather_qmm expert dispatch)

/// Activation for the gated-GLU FFN inside a `MoESwitchGLU`. Receives the
/// `gate` and `up` projection outputs and returns the elementwise-gated
/// product fed to `down_proj`.
///
/// The native MoE families split into exactly two activations:
///   - `.swiglu` -- `silu(gate) * up` -- Qwen3-MoE, Qwen2-MoE, Mixtral,
///     OLMoE, DeepSeek-V2/V2-Lite.
///   - `.geglu` -- `geluApproximate(gate) * up` -- Gemma 4 (26B-A4B).
///
/// `silu` and `geluApproximate` resolve to MLXNN's public functions, i.e.
/// the exact same ops the per-family copies called before this module was
/// extracted, so the gated output is bit-identical.
enum MoEActivation {
    case swiglu
    case geglu

    @inline(__always)
    func callAsFunction(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
        switch self {
        case .swiglu: return silu(gate) * up
        case .geglu: return geluApproximate(gate) * up
        }
    }
}

/// One stacked quantized switched-linear projection -- the
/// `[numExperts, outputDims, inputDims_packed]` quantized weight plus
/// per-expert scales and biases -- that dispatches across the chosen
/// top-K experts in a single `gatherQuantizedMM` call instead of a Swift
/// `for` loop over per-expert matmuls.
///
/// This is the shared form of the per-family copies that used to live in
/// `Qwen3MoEModel` / `Gemma4Model` / `MixtralModel` / `Qwen2MoEModel` /
/// `OLMoEModel` / `DeepSeekModel`. The scatter-dispatch path that landed
/// in the first MoE PRs walked the experts in a Swift loop, which forced a
/// per-layer host sync (the loop bounds came from a CPU read of per-expert
/// token counts); decoding one token per step paid that sync once per
/// layer and dominated the FFN math. `gather_qmm` keeps the whole dispatch
/// on the GPU and matches `mlx_lm/models/switch_layers.QuantizedSwitchLinear`
/// bit for bit.
///
/// Parameter layout matches mlx-community's packed MoE format directly, so
/// the loader binds `...switch_mlp.{proj}.{weight,scales,biases}` (or
/// `experts.switch_glu.{proj}.*` for Gemma 4) with no per-expert unpacking:
///   - `weight: [E, O, I/(32/bits)]` int-packed
///   - `scales: [E, O, I/groupSize]`
///   - `biases: [E, O, I/groupSize]`
class MoEQuantizedSwitchedLinear: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    // `biases` is nil for non-affine modes (nvfp4/mxfp4): MLX's `mx.quantize`
    // emits no biases for those, so the checkpoint has no `.biases` key and
    // `gather_qmm` takes `biases: nil`. Optional, like MLXNN's `Linear.bias`,
    // so the missing parameter is simply absent rather than a load mismatch.
    @ParameterInfo(key: "biases") var biases: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode

    init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        groupSize: Int, bits: Int, mode: QuantizationMode = .affine
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        // Pre-allocate the parameter tensors with the SAME shape/dtype the
        // checkpoint ships so the loader's `model.update(parameters:)` binds
        // them by shape match. The fill values are placeholders overwritten at
        // load time. nvfp4 packs scales as uint8 (e4m3 group scales) with no
        // biases; affine ships bfloat16 scales + biases.
        let packedIn = inputDims * bits / 32
        let groupsIn = inputDims / groupSize
        _weight = ParameterInfo(
            wrappedValue: MLXArray.zeros([numExperts, outputDims, packedIn], dtype: .uint32),
            key: "weight")
        let scalesDtype: DType = (mode == .affine) ? .bfloat16 : .uint8
        _scales = ParameterInfo(
            wrappedValue: MLXArray.zeros([numExperts, outputDims, groupsIn], dtype: scalesDtype),
            key: "scales")
        if mode == .affine {
            _biases = ParameterInfo(
                wrappedValue: MLXArray.zeros([numExperts, outputDims, groupsIn], dtype: .bfloat16),
                key: "biases")
        } else {
            _biases = ParameterInfo(wrappedValue: nil, key: "biases")
        }
    }

    /// Per-token expert dispatch. `x` is shaped so the last two dims feed
    /// `gather_qmm`'s `[..., M, K]` matmul slot (the `MoESwitchGLU` caller
    /// expands to `[..., 1, 1, I]`); `indices` is `[..., K]` Int32 expert
    /// ids into the weight tensor's leading batch dim.
    /// - Parameter sortedIndices: When true the caller has pre-sorted
    ///   `indices` by expert id so MLX's gather kernel can use the faster
    ///   sorted-indices path (the prefill sort path).
    func callAsFunction(
        _ x: MLXArray, indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        return gatherQuantizedMM(
            x, weight,
            scales: scales, biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize, bits: bits, mode: mode,
            sortedIndices: sortedIndices)
    }
}

/// Experts as three stacked quantized switched linears (`gate_proj`,
/// `up_proj`, `down_proj`) plus a gated-GLU activation. Mirrors mlx-lm's
/// `switch_layers.SwitchGLU`, parameterized by `MoEActivation` so the one
/// module serves both the SwiGLU families (Qwen3/Qwen2-MoE/Mixtral/OLMoE/
/// DeepSeek) and Gemma 4's GeGLU. The in-checkpoint key path
/// `...{proj}.{weight,scales,biases}` lines up with this module hierarchy
/// directly, so no per-family subclass is needed.
///
/// Forward:
///   1. Reshape `[N, H]` to `[N, 1, 1, H]` so each row participates in
///      `topK` expert matmuls (one per chosen expert).
///   2. `gate_proj` / `up_proj` via `gatherQuantizedMM` -> `[N, topK, 1,
///      hiddenDims]` in a single device kernel each.
///   3. The `MoEActivation` gate: `act(gate) * up`.
///   4. `down_proj` back to `[N, topK, 1, H]`.
///   5. Squeeze the M=1 axis to `[N, topK, H]`. The caller does the topK
///      weighted sum.
///
/// At high token counts (`indices.size >= moeSortThreshold`, i.e. prefill)
/// the forward sorts the `(token, expert)` assignments by expert id so each
/// expert's gather slice is contiguous and MLX's `gather_qmm`
/// `sortedIndices` fast path applies, recovering the long-prompt prefill
/// throughput the unsorted dispatch regresses. Decode (`indices.size` below
/// the threshold) stays on the unsorted path so its 2.7x win is untouched.
/// See `MoESortPath.swift` for the sort helpers and the mlx-swift vs Python
/// shape-contract notes.
class MoESwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: MoEQuantizedSwitchedLinear
    @ModuleInfo(key: "up_proj") var upProj: MoEQuantizedSwitchedLinear
    @ModuleInfo(key: "down_proj") var downProj: MoEQuantizedSwitchedLinear

    /// The gated-GLU activation. Stored as a plain value (not a parameter
    /// or submodule) so MLXNN's Mirror walk ignores it.
    let activation: MoEActivation

    init(
        inputDims: Int, hiddenDims: Int, numExperts: Int,
        groupSize: Int, bits: Int, activation: MoEActivation,
        mode: QuantizationMode = .affine
    ) {
        self.activation = activation
        _gateProj = ModuleInfo(
            wrappedValue: MoEQuantizedSwitchedLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, groupSize: groupSize, bits: bits, mode: mode),
            key: "gate_proj")
        _upProj = ModuleInfo(
            wrappedValue: MoEQuantizedSwitchedLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, groupSize: groupSize, bits: bits, mode: mode),
            key: "up_proj")
        _downProj = ModuleInfo(
            wrappedValue: MoEQuantizedSwitchedLinear(
                inputDims: hiddenDims, outputDims: inputDims,
                numExperts: numExperts, groupSize: groupSize, bits: bits, mode: mode),
            key: "down_proj")
    }

    /// - Parameters:
    ///   - x: `[N, H]` flattened token activations.
    ///   - indices: `[N, topK]` Int32 expert ids per token (router score
    ///     order).
    /// - Returns: `[N, topK, H]` per-expert outputs; the caller does the
    ///   topK weighted sum.
    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let N = x.dim(0)
        let H = x.dim(1)
        let topK = indices.dim(indices.ndim - 1)

        // Prefill (many assignments): sort (token, expert) by expert id so
        // each expert's gather slice is contiguous and the gather_qmm
        // `sortedIndices` fast path applies, recovering long-prompt
        // throughput the unsorted dispatch regresses. Output is unsorted
        // back to (token, slot) order so the caller's weighted sum is
        // unchanged. See `MoESortPath.swift`.
        if moeShouldSort(n: N, topK: topK) {
            let (xs, idx, invOrder) = moeGatherSort(x, indices: indices)
            // xs: [N*topK, 1, H] -- one M=1 row per assignment, sorted.
            let xGate = gateProj(xs, indices: idx, sortedIndices: true)
            let xUp = upProj(xs, indices: idx, sortedIndices: true)
            let activated = activation(xGate, xUp)
            let out = downProj(activated, indices: idx, sortedIndices: true)
            // out: [N*topK, 1, H_out] -- unsort to [N, topK, H].
            return moeScatterUnsort(out, invOrder: invOrder, n: N, topK: topK)
        }

        // Decode / short prompts (assignments below the sort threshold):
        // expand to [N, 1, 1, H] so each (token, slot) sees an M=1 row
        // inside the gather. [N, 1] outer batch x [N, topK] indices ->
        // [N, topK, 1, H_out] per projection: no Swift loop, no per-layer
        // host sync, one device kernel per projection.
        let xExp = x.reshaped(N, 1, 1, H)
        let idx = indices.asType(.int32)

        let xGate = gateProj(xExp, indices: idx)
        let xUp = upProj(xExp, indices: idx)
        let activated = activation(xGate, xUp)
        let out = downProj(activated, indices: idx)

        // out: [N, topK, 1, H_out] -- squeeze the M=1 inner axis.
        return out.squeezed(axis: -2)
    }
}
