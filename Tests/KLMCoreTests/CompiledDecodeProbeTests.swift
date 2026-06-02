import XCTest
import MLX
import MLXNN
import MLXFast
import MLXRandom
import KLMCache

/// Empirical probe for the "compiled text-decode block" perf lever.
///
/// The default `KVCache` returns a slice that grows one row per decode step, so
/// a block forward built on it changes shape every step and `MLX.compile`
/// re-traces each step (no reuse). `FixedBufferKVCache` keeps a constant-shape
/// `[B, H, capacity, D]` buffer with a dynamic write offset, so a compiled
/// block forward is traced once and replayed -- at the cost of running
/// attention over the full padded `capacity` every step.
///
/// This probe builds transformer-style decode blocks (GQA attention + SwiGLU
/// MLP + two RMSNorms) at Llama-3.2-3B dims with random weights, and:
///   1. asserts the compiled fixed-buffer path is numerically equivalent to the
///      uncompiled growing-cache path at matched offsets (the position mask
///      must make the padded buffer behave like a length-`offset` cache);
///   2. times both over many steps and prints decode tok/s, so the M-series
///      verdict (does compile beat the extra padded-attention compute?) is
///      data, not conjecture.
///
/// Gated on `KLM_DECODE_PROBE` (it allocates real-size tensors and runs timing
/// loops); skipped in the normal suite. It touches no shipped code path.
final class CompiledDecodeProbeTests: XCTestCase {

    // Llama-3.2-3B-ish dims.
    private let B = 1
    private let hidden = 3072
    private let nHeads = 24
    private let nKVHeads = 8
    private let headDim = 128
    private let inter = 8192
    private let dtype = DType.float16

    private struct Weights {
        let wIn: MLXArray, wPost: MLXArray
        let wq: MLXArray, wk: MLXArray, wv: MLXArray, wo: MLXArray
        let wGate: MLXArray, wUp: MLXArray, wDown: MLXArray
    }

    private func makeWeights() -> Weights {
        func lin(_ outD: Int, _ inD: Int) -> MLXArray {
            (MLXRandom.normal([outD, inD]) * 0.02).asType(dtype)
        }
        return Weights(
            wIn: MLXArray.ones([hidden]).asType(dtype),
            wPost: MLXArray.ones([hidden]).asType(dtype),
            wq: lin(nHeads * headDim, hidden),
            wk: lin(nKVHeads * headDim, hidden),
            wv: lin(nKVHeads * headDim, hidden),
            wo: lin(hidden, nHeads * headDim),
            wGate: lin(inter, hidden),
            wUp: lin(inter, hidden),
            wDown: lin(hidden, inter))
    }

    private func rmsNorm(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: w, eps: 1e-5)
    }

    /// Project the new token to per-head q/k/v: q `[B,nHeads,1,hd]`,
    /// k/v `[B,nKVHeads,1,hd]`.
    private func project(_ x: MLXArray, _ w: Weights) -> (MLXArray, MLXArray, MLXArray) {
        let h = rmsNorm(x, w.wIn)                                  // [B,1,hidden]
        let q = matmul(h, w.wq.transposed(1, 0)).reshaped(B, 1, nHeads, headDim).transposed(0, 2, 1, 3)
        let k = matmul(h, w.wk.transposed(1, 0)).reshaped(B, 1, nKVHeads, headDim).transposed(0, 2, 1, 3)
        let v = matmul(h, w.wv.transposed(1, 0)).reshaped(B, 1, nKVHeads, headDim).transposed(0, 2, 1, 3)
        return (q, k, v)
    }

    /// Finish the block from the attention output (`[B,nHeads,1,hd]`) + residual.
    private func finish(_ x: MLXArray, _ attn: MLXArray, _ w: Weights) -> MLXArray {
        let o = matmul(attn.transposed(0, 2, 1, 3).reshaped(B, 1, nHeads * headDim), w.wo.transposed(1, 0))
        let h = x + o
        let hn = rmsNorm(h, w.wPost)
        let ffn = matmul(silu(matmul(hn, w.wGate.transposed(1, 0))) * matmul(hn, w.wUp.transposed(1, 0)),
                         w.wDown.transposed(1, 0))
        return h + ffn
    }

    private let scale: Float = 1.0 / 11.313708  // 1/sqrt(128)

    // MARK: - Correctness: compiled fixed-buffer == uncompiled growing

    func testCompiledFixedBufferMatchesGrowingCache() throws {
        guard ProcessInfo.processInfo.environment["KLM_DECODE_PROBE"] != nil else {
            throw XCTSkip("Set KLM_DECODE_PROBE to run the compiled-decode probe")
        }
        MLXRandom.seed(0)
        let w = makeWeights()
        let capacity = 256
        let prefill = 40

        // Seed identical prefix K/V into both a growing history and the fixed buffer.
        var kHist = MLXArray.zeros([B, nKVHeads, 0, headDim], dtype: dtype)
        var vHist = MLXArray.zeros([B, nKVHeads, 0, headDim], dtype: dtype)
        let cache = FixedBufferKVCache(
            batch: B, heads: nKVHeads, headDim: headDim, valueDim: headDim,
            capacity: capacity, dtype: dtype)
        for _ in 0 ..< prefill {
            let x = (MLXRandom.normal([B, 1, hidden]) * 0.1).asType(dtype)
            let (_, k, v) = project(x, w)
            kHist = concatenated([kHist, k], axis: 2)
            vHist = concatenated([vHist, v], axis: 2)
            cache.writeStep(keys: k, values: v)
        }
        eval(kHist, vHist, cache.keys, cache.values)

        // One more token through both paths; outputs must match.
        let x = (MLXRandom.normal([B, 1, hidden]) * 0.1).asType(dtype)
        let (q, k, v) = project(x, w)

        // Uncompiled growing path.
        let kg = concatenated([kHist, k], axis: 2)
        let vg = concatenated([vHist, v], axis: 2)
        let attnG = MLXFast.scaledDotProductAttention(
            queries: q, keys: kg, values: vg, scale: scale, mask: nil)
        let outG = finish(x, attnG, w)

        // Fixed-buffer path (padded attention + position mask).
        cache.writeStep(keys: k, values: v)
        let mask = cache.positionMask(dtype: dtype)
        let attnF = MLXFast.scaledDotProductAttention(
            queries: q, keys: cache.keys, values: cache.values,
            scale: scale, mask: mask)
        let outF = finish(x, attnF, w)

        eval(outG, outF)
        let g = outG.asType(.float32).asArray(Float.self)
        let f = outF.asType(.float32).asArray(Float.self)
        var maxAbs: Float = 0
        for i in 0 ..< g.count { maxAbs = max(maxAbs, abs(g[i] - f[i])) }
        XCTAssertLessThan(maxAbs, 5e-3,
            "fixed-buffer padded attention must match growing cache (maxAbs=\(maxAbs))")
    }

    // MARK: - Timing: uncompiled growing vs compiled fixed-buffer

    func testDecodeStepTiming() throws {
        guard ProcessInfo.processInfo.environment["KLM_DECODE_PROBE"] != nil else {
            throw XCTSkip("Set KLM_DECODE_PROBE to run the compiled-decode probe")
        }
        MLXRandom.seed(1)
        // Stack `nBlocks` distinct blocks so the per-step lazy graph-build
        // overhead that `MLX.compile` eliminates is realistically represented
        // (it scales with layer count -- a single block understates it).
        let nBlocks = 16
        let steps = 128
        let weights = (0 ..< nBlocks).map { _ in makeWeights() }

        for capacity in [256, 1024] {
            // --- Uncompiled growing cache (production decode pattern) ---
            var kHist = (0 ..< nBlocks).map { _ in MLXArray.zeros([B, nKVHeads, 0, headDim], dtype: dtype) }
            var vHist = kHist
            func growingStep(_ x0: MLXArray) -> MLXArray {
                var x = x0
                for i in 0 ..< nBlocks {
                    let (q, k, v) = project(x, weights[i])
                    kHist[i] = concatenated([kHist[i], k], axis: 2)
                    vHist[i] = concatenated([vHist[i], v], axis: 2)
                    let attn = MLXFast.scaledDotProductAttention(
                        queries: q, keys: kHist[i], values: vHist[i], scale: scale, mask: nil)
                    x = finish(x, attn, weights[i])
                }
                return x
            }
            var x = (MLXRandom.normal([B, 1, hidden]) * 0.1).asType(dtype)
            _ = growingStep(x); eval(kHist + vHist)  // warm
            let tGrow = time {
                for _ in 0 ..< steps { x = growingStep(x); eval([x] + kHist) }
            }

            // --- Compiled fixed buffer: compile the WHOLE nBlocks step once ---
            let caches = (0 ..< nBlocks).map { _ in
                FixedBufferKVCache(
                    batch: B, heads: nKVHeads, headDim: headDim, valueDim: headDim,
                    capacity: capacity, dtype: dtype)
            }
            // Cache buffers are Updatable in/out; new k/v are scattered at the
            // runtime offset so the traced multi-block graph is shape-stable and
            // reused across steps (one trace instead of nBlocks graph-builds/step).
            let compiled = MLX.compile(inputs: caches, outputs: caches) {
                (args: [MLXArray]) -> [MLXArray] in
                var x = args[0]
                let offArr = args[1]
                let pos = MLXArray(Int32(0) ..< Int32(capacity))
                let mask = ((pos .> offArr).asType(self.dtype) * Float(-30000.0)).reshaped(1, 1, 1, capacity)
                for i in 0 ..< nBlocks {
                    let (q, k, v) = self.project(x, weights[i])
                    caches[i].keys[0..., 0..., offArr, 0...] = k.squeezed(axis: 2)
                    caches[i].values[0..., 0..., offArr, 0...] = v.squeezed(axis: 2)
                    let attn = MLXFast.scaledDotProductAttention(
                        queries: q, keys: caches[i].keys, values: caches[i].values,
                        scale: self.scale, mask: mask)
                    x = self.finish(x, attn, weights[i])
                }
                return [x]
            }
            var xc = (MLXRandom.normal([B, 1, hidden]) * 0.1).asType(dtype)
            var off = 0
            func compiledStep() {
                xc = compiled([xc, MLXArray(Int32(off))])[0]
                off += 1
            }
            compiledStep(); eval([xc] + caches.map { $0.keys })  // warm (trace once)
            let tComp = time {
                for _ in 0 ..< steps { compiledStep(); eval([xc] + caches.map { $0.keys }) }
            }

            let growTokS = Double(steps) / tGrow
            let compTokS = Double(steps) / tComp
            print(String(format:
                "[decode-probe %d blocks cap=%d] growing %.1f tok/s (%.2f ms/step)  |  compiled-fixed %.1f tok/s (%.2f ms/step)  |  speedup %.2fx",
                nBlocks, capacity, growTokS, tGrow / Double(steps) * 1000,
                compTokS, tComp / Double(steps) * 1000, compTokS / growTokS))
        }
    }

    private func time(_ body: () -> Void) -> Double {
        let start = Date()
        body()
        return -start.timeIntervalSinceNow
    }
}
