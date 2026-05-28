import Foundation
import MLX

// MARK: - SwitchGLU sort path (prefill parity)

/// Shared `(token, expert)` sort helpers for the SwitchGLU MoE dispatch,
/// mirroring mlx-lm's `switch_layers._gather_sort` / `_scatter_unsort`.
///
/// ## Why this exists
///
/// Both `Gemma4SwitchGLU` (#82) and `Qwen3SwitchGLU` (#85) dispatch the
/// chosen top-K experts in a single `gatherQuantizedMM` per projection.
/// The unsorted dispatch does an `M = 1` matmul per `(token, expert)`
/// pair with the expert weights gathered in router-score order, so the
/// kernel touches expert slices in an arbitrary, repeating order. That
/// is fine at decode (`N = 1`, a handful of assignments) but regresses
/// long-prompt prefill, where there are `N * topK` assignments and the
/// scattered expert access defeats weight-tile reuse (measured on
/// Qwen3-Coder-30B-A3B: prefill 385 -> 230 tok/s, -40%, on a 256-token
/// prompt while decode went 24 -> 66 tok/s).
///
/// mlx-lm recovers prefill by sorting the flattened `(token, slot)`
/// assignments by expert id once `indices.size >= 64`, so every expert's
/// gather slice is contiguous and the `gather_qmm` `sortedIndices` fast
/// path applies. The output is unsorted back to the original
/// `(token, slot)` order so the caller's router-weighted sum is
/// unchanged. Decode stays on the unsorted path (its `indices.size` is
/// far below the threshold), so its 2.7x win is untouched.
///
/// ## mlx-swift shape contract (differs from the Python reference)
///
/// In the unsorted path the activations are 4-D `[N, 1, 1, H]` and the
/// indices are 2-D `[N, topK]`, so `gather_qmm` broadcasts the `[N, 1]`
/// activation batch against the `[N, topK]` index batch to produce
/// `[N, topK, 1, O]`. The sorted path instead flattens both into one
/// batch axis: activations become 3-D `[N*topK, 1, H]` and indices 1-D
/// `[N*topK]`, yielding `[N*topK, 1, O]`. This 4-D vs 3-D batching is the
/// shape difference the original SwitchGLU notes flagged as a follow-up.

/// Token count at which the SwitchGLU sorts its assignments, matching
/// mlx-lm's `do_sort = indices.size >= 64`. `indices.size` is
/// `N * topK`, so at decode (`N = 1`) the sort is skipped; at prefill it
/// engages.
let moeSortThreshold = 64

/// Whether the SwitchGLU should sort its `(token, expert)` assignments
/// for this routing. Mirrors mlx-lm's `do_sort = indices.size >= 64`.
@inline(__always)
func moeShouldSort(n: Int, topK: Int) -> Bool {
    return n * topK >= moeSortThreshold
}

/// Sort `(token, expert)` assignments by expert id so each expert's
/// gather slice is contiguous. Mirrors mlx-lm's `_gather_sort`.
///
/// - Parameters:
///   - x: `[N, H]` flattened token activations.
///   - indices: `[N, topK]` expert ids per token (router-score order).
/// - Returns:
///   - x: `[N*topK, 1, H]` gathered `M = 1` rows in ascending-expert
///     order -- one row per `(token, slot)` assignment.
///   - idx: `[N*topK]` flat Int32 expert ids, ascending (so the
///     `gather_qmm` `sortedIndices` fast path is valid).
///   - invOrder: `[N*topK]` inverse permutation that restores the
///     original `(token, slot)` order after the expert matmuls.
func moeGatherSort(
    _ x: MLXArray, indices: MLXArray
) -> (x: MLXArray, idx: MLXArray, invOrder: MLXArray) {
    let N = x.dim(0)
    let H = x.dim(1)
    let topK = indices.dim(indices.ndim - 1)

    let flatIdx = indices.flattened()              // [N*topK]
    let order = argSort(flatIdx)                    // [N*topK] sort permutation
    let invOrder = argSort(order)                   // [N*topK] inverse permutation
    let sortedIdx = flatIdx.take(order, axis: 0)    // [N*topK] ascending expert ids
    // `order / topK` maps each sorted slot back to its source token row
    // (mlx-lm's `order // M`).
    let tokenIdx = order.floorDivide(topK)          // [N*topK]
    let gatheredX = x.take(tokenIdx, axis: 0)       // [N*topK, H]
        .reshaped(N * topK, 1, H)                   // [N*topK, 1, H]
    return (gatheredX, sortedIdx.asType(.int32), invOrder)
}

/// Restore the original `(token, slot)` order after the sorted expert
/// matmuls and fold back to `[N, topK, H]`. Mirrors mlx-lm's
/// `_scatter_unsort` followed by the SwitchGLU `squeeze(-2)`.
///
/// - Parameters:
///   - x: `[N*topK, 1, H]` per-assignment expert outputs in
///     ascending-expert (sorted) order.
///   - invOrder: the `[N*topK]` inverse permutation from `moeGatherSort`.
///   - n: original token count `N`.
///   - topK: experts per token.
/// - Returns: `[N, topK, H]` outputs in the original `(token, slot)`
///   order, ready for the caller's router-weighted sum.
func moeScatterUnsort(
    _ x: MLXArray, invOrder: MLXArray, n: Int, topK: Int
) -> MLXArray {
    let H = x.dim(x.ndim - 1)
    let unsorted = x.take(invOrder, axis: 0)        // [N*topK, 1, H]
    return unsorted.reshaped(n, topK, 1, H)         // [N, topK, 1, H]
        .squeezed(axis: -2)                         // [N, topK, H]
}
