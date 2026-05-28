import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import KLMCore

/// PR #87: the SwitchGLU prefill sort path.
///
/// At prefill the SwitchGLU sorts its `(token, expert)` assignments by
/// expert id (`moeGatherSort`), runs the `gather_qmm` `sortedIndices`
/// fast path, then unsorts back to `(token, slot)` order
/// (`moeScatterUnsort`). These tests pin the two invariants that make
/// that safe:
///
///   1. The sort path is *numerically identical* to the unsorted
///      dispatch -- verified with a float `gatherMM` surrogate that has
///      the exact same shape contract and `sortedIndices` semantics as
///      the quantized `gather_qmm` used in production, but over non-zero
///      weights (the no-weight-load synthetic MoE tests run over zero
///      quantized placeholders, so they cannot catch a sort/unsort
///      mismatch).
///   2. The `moeShouldSort` threshold engages at prefill and is skipped
///      at decode, so decode keeps the unsorted fast path untouched.
final class MoESortPathTests: XCTestCase {

    /// Run the unsorted dispatch (the production decode path's shape:
    /// `[N,1,1,H]` activations, `[N,topK]` indices) over a float expert
    /// stack and return `[N, topK, O]`.
    private func unsortedReference(
        x: MLXArray, weightT: MLXArray, indices: MLXArray
    ) -> MLXArray {
        let N = x.dim(0)
        let H = x.dim(1)
        let xExp = x.reshaped(N, 1, 1, H)
        // gatherMM(a=[N,1,1,H], b=[E,H,O], rhsIndices=[N,topK]) -> [N,topK,1,O]
        let out = gatherMM(
            xExp, weightT, rhsIndices: indices.asType(.int32),
            sortedIndices: false)
        return out.squeezed(axis: -2)  // [N, topK, O]
    }

    /// Run the sort path (`moeGatherSort` -> sorted `gatherMM` ->
    /// `moeScatterUnsort`) over the same float expert stack.
    private func sortedPath(
        x: MLXArray, weightT: MLXArray, indices: MLXArray
    ) -> MLXArray {
        let N = x.dim(0)
        let topK = indices.dim(indices.ndim - 1)
        let (xs, idx, invOrder) = moeGatherSort(x, indices: indices)
        // xs: [N*topK, 1, H], idx: [N*topK] (ascending) -> [N*topK, 1, O]
        let out = gatherMM(xs, weightT, rhsIndices: idx, sortedIndices: true)
        return moeScatterUnsort(out, invOrder: invOrder, n: N, topK: topK)
    }

    private func assertAllClose(
        _ a: MLXArray, _ b: MLXArray, tol: Float = 1e-4, _ msg: String = ""
    ) {
        XCTAssertEqual(a.shape, b.shape, "shape mismatch \(msg)")
        eval(a, b)
        let av = a.asArray(Float.self)
        let bv = b.asArray(Float.self)
        var maxDiff: Float = 0
        for i in 0..<av.count { maxDiff = max(maxDiff, abs(av[i] - bv[i])) }
        XCTAssertLessThan(maxDiff, tol,
            "sort path diverged (max|Δ| = \(maxDiff)) \(msg)")
    }

    /// The sort path must reproduce the unsorted dispatch bit-for-bit
    /// (within fp tolerance) across many tokens, repeated experts, and a
    /// non-trivial routing -- this is the core safety property.
    func testSortPathMatchesUnsortedDispatch() {
        MLXRandom.seed(0)
        let N = 40, topK = 4, H = 16, O = 12, E = 6  // N*topK = 160 >= 64 -> sorts
        let x = MLXRandom.normal([N, H]).asType(.float32)
        // Expert weights stacked [E, O, H]; gatherMM wants b = [E, H, O].
        let weight = MLXRandom.normal([E, O, H]).asType(.float32)
        let weightT = weight.swappedAxes(-1, -2)
        let indices = MLXRandom.randInt(0 ..< Int32(E), [N, topK])

        XCTAssertTrue(moeShouldSort(n: N, topK: topK),
            "N*topK must exceed the sort threshold for this test")
        let reference = unsortedReference(x: x, weightT: weightT, indices: indices)
        let sorted = sortedPath(x: x, weightT: weightT, indices: indices)
        assertAllClose(reference, sorted, "many-token routing")
    }

    /// Edge case: a routing where every token picks the same expert (the
    /// sort is a no-op permutation but the unsort must still reconstruct
    /// order) and one where experts are strictly descending (the sort
    /// fully reverses), to stress the permutation algebra.
    func testSortPathMatchesUnderDegenerateRoutings() {
        MLXRandom.seed(1)
        let N = 32, topK = 2, H = 8, O = 8, E = 5  // N*topK = 64 -> sorts
        let x = MLXRandom.normal([N, H]).asType(.float32)
        let weight = MLXRandom.normal([E, O, H]).asType(.float32)
        let weightT = weight.swappedAxes(-1, -2)

        // All assignments to expert 3.
        let allSame = broadcast(MLXArray(Int32(3)).reshaped(1, 1), to: [N, topK])
        assertAllClose(
            unsortedReference(x: x, weightT: weightT, indices: allSame),
            sortedPath(x: x, weightT: weightT, indices: allSame),
            "all-same-expert routing")

        // Descending-by-token expert assignment (forces a full reorder).
        let perToken = (0..<N).map { Int32(($0 + 1) % E) }
        var flat = [Int32]()
        for t in perToken { flat.append(t); flat.append((t + 1) % Int32(E)) }
        let desc = MLXArray(flat).reshaped(N, topK)
        assertAllClose(
            unsortedReference(x: x, weightT: weightT, indices: desc),
            sortedPath(x: x, weightT: weightT, indices: desc),
            "interleaved routing")
    }

    /// `moeGatherSort` must emit ascending expert ids (the precondition
    /// for `gather_qmm`'s `sortedIndices` fast path) and a `(gathered x,
    /// invOrder)` pair that round-trips back to identity.
    func testGatherSortProducesAscendingIndicesAndRoundTrips() {
        MLXRandom.seed(2)
        let N = 20, topK = 3, H = 4, E = 7
        let x = MLXRandom.normal([N, H]).asType(.float32)
        let indices = MLXRandom.randInt(0 ..< Int32(E), [N, topK])

        let (xs, idx, invOrder) = moeGatherSort(x, indices: indices)
        XCTAssertEqual(xs.shape, [N * topK, 1, H])
        XCTAssertEqual(idx.shape, [N * topK])

        // idx is sorted ascending.
        eval(idx)
        let ids = idx.asArray(Int32.self)
        for i in 1..<ids.count {
            XCTAssertLessThanOrEqual(ids[i - 1], ids[i],
                "moeGatherSort must emit ascending expert ids")
        }

        // Unsorting the gathered rows restores the original [N,topK,H]
        // (token, slot) layout: row (t, k) == x[t].
        let restored = moeScatterUnsort(xs, invOrder: invOrder, n: N, topK: topK)
        let expected = broadcast(x.reshaped(N, 1, H), to: [N, topK, H])
        assertAllClose(expected, restored, "round-trip identity")
    }

    /// End-to-end cover for the *production* quantized kernel path: build
    /// a real `Qwen3SwitchGLU` with quantized random weights and assert
    /// that the sorted-batch dispatch (`N*topK >= 64`, the prefill path
    /// through `gatherQuantizedMM(sortedIndices: true)`) produces the
    /// same per-`(token, slot)` output as running each token alone
    /// (`indices.size = topK < 64`, the unsorted decode path). This
    /// exercises the quantized `gather_qmm` sorted-indices kernel that
    /// the float `gatherMM` surrogate above does not, over non-zero
    /// expert weights (the no-weight-load synthetic MoE tests run over
    /// zero quantized placeholders).
    func testQuantizedSwitchGLUSortedMatchesUnsorted() throws {
        MLXRandom.seed(7)
        // group size must be one of {32, 64, 128} for mlx's quantize op;
        // both projection input dims (H and I) must be divisible by it.
        let H = 64, I = 64, E = 8, gs = 32, bits = 4, topK = 8
        let glu = Qwen3SwitchGLU(
            inputDims: H, hiddenDims: I, numExperts: E,
            groupSize: gs, bits: bits)

        // The module is born with zero packed placeholders; fill the
        // three stacked projections with quantized random weights via
        // the same update(parameters:) path the real loader uses.
        var flat: [(String, MLXArray)] = []
        func addProj(_ name: String, outDim: Int, inDim: Int) {
            let wf = MLXRandom.normal([E, outDim, inDim]).asType(.float32) * 0.1
            let (wq, scales, biases) = quantized(wf, groupSize: gs, bits: bits)
            flat.append(("\(name).weight", wq))
            flat.append(("\(name).scales", scales.asType(.bfloat16)))
            flat.append((
                "\(name).biases",
                (biases ?? MLXArray.zeros(scales.shape)).asType(.bfloat16)))
        }
        addProj("gate_proj", outDim: I, inDim: H)
        addProj("up_proj", outDim: I, inDim: H)
        addProj("down_proj", outDim: H, inDim: I)
        try glu.update(
            parameters: ModuleParameters.unflattened(flat), verify: [.all])

        // N=8, topK=8 -> indices.size = 64 -> the sorted path engages.
        let N = 8
        let x = MLXRandom.normal([N, H]).asType(.float32)
        let indices = MLXRandom.randInt(0 ..< Int32(E), [N, topK])
        XCTAssertTrue(moeShouldSort(n: N, topK: topK))

        let sorted = glu(x, indices: indices)  // [N, topK, H] via sort path

        // Per-token unsorted reference: each token alone is below the
        // threshold, so it dispatches through the unsorted kernel.
        var rows = [MLXArray]()
        for i in 0..<N {
            XCTAssertFalse(moeShouldSort(n: 1, topK: topK))
            rows.append(glu(x[i ..< (i + 1)], indices: indices[i ..< (i + 1)]))
        }
        let unsorted = concatenated(rows, axis: 0)  // [N, topK, H]
        assertAllClose(sorted, unsorted, tol: 5e-3,
            "quantized sorted kernel vs unsorted per-token dispatch")
    }

    /// Decode (`N = 1`) stays below the sort threshold so the unsorted
    /// fast path runs; prefill-scale token counts cross it.
    func testSortThresholdEngagesAtPrefillNotDecode() {
        XCTAssertFalse(moeShouldSort(n: 1, topK: 8),
            "decode (1 token, topK 8 -> size 8) must skip the sort")
        XCTAssertFalse(moeShouldSort(n: 7, topK: 8),
            "size 56 (< 64) must skip the sort")
        XCTAssertTrue(moeShouldSort(n: 8, topK: 8),
            "size 64 must engage the sort (mlx-lm boundary)")
        XCTAssertTrue(moeShouldSort(n: 256, topK: 8),
            "a 256-token prefill must sort")
    }
}
