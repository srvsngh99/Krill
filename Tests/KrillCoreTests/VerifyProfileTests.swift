import XCTest
import MLX
import MLXNN
import MLXFast
import MLXRandom
import KrillCache

/// Empirical probe for the **speculative-verify cost ceiling** (the `beta` gate
/// for n-gram / prompt-lookup speculative decoding).
///
/// WS2 (`docs/workstreams/WS2_SPECULATIVE_DECODING.md`) capped speculative decode
/// using a *fitted* per-round overhead `beta ~ 0.78`, never a measured one. The
/// n-gram (draft-model-free) variant has `alpha -> 0`, so its single-stream
/// speedup ceiling collapses to `1 / beta`. Whether that lever is worth shipping
/// depends entirely on the *measured* `beta`, which this probe produces.
///
/// ## What it measures
///
/// For a stack of `nBlocks` transformer-style decode blocks (GQA attention +
/// SwiGLU MLP + two RMSNorms) at Llama-3.2-3B dims with **production-faithful
/// 4-bit quantized weights**, against a fixed-length-`L` KV history:
///   - `t1`  = cost of a **width-1** forward (one plain decode step), and
///   - `tK`  = cost of a **width-K** verify forward (the spec-decode verify shape),
/// over `K in {1,2,4,8,16,32}` and `L in {512, 4096}`, *including the lm_head over
/// all K positions* (the verify path needs an argmax at every position, so it
/// pays a K-wide lm_head — a real, K-scaling cost the profile must capture).
///
/// From those it derives, per K, the directly-measured quantities the decision
/// hinges on:
///   - `r(K)   = tK / t1`              (verify cost in decode-steps)
///   - `beta   = (r(K) + 1)/(K + 1)`   (per-round overhead, WS2's parameter)
///   - `ideal  = (K + 1)/(r(K) + 1)`   (= 1/beta; speedup at 100% acceptance,
///                                       an UPPER bound — real workloads accept < 1)
/// and prints the lm_head-only cost so its K-scaling contribution is visible.
///
/// If `r(K)` stays near 1 (the K extra positions ride along on the one weight
/// stream), `ideal -> K+1` and the lever is huge. If `r(K) ~ K` (no batching
/// benefit), `ideal -> 1` and it is dead. The truth is in between and is the
/// answer.
///
/// ## Decision (see the plan + `docs/CEILINGS_AND_REATTEMPTS.md` entry #1)
///   - `ideal >= 1.5` at a reachable `K*` (8-16)  -> ship n-gram single-stream.
///   - `1.2 <= ideal < 1.5`                       -> ship advisory-gated.
///   - `ideal < 1.2`                              -> stay closed; update the doc.
///
/// Gated on `KRILL_VERIFY_PROFILE` (allocates real-size tensors + timing loops);
/// skipped in the normal suite. Touches no shipped code path.
final class VerifyProfileTests: XCTestCase {

    // Llama-3.2-3B dims (match CompiledDecodeProbeTests).
    private let B = 1
    private let hidden = 3072
    private let nHeads = 24
    private let nKVHeads = 8
    private let headDim = 128
    private let inter = 8192
    private let vocab = 128_256        // Llama-3 vocab — lm_head is a big, K-scaling cost
    private let groupSize = 64
    private let dtype = DType.float16
    private let scale: Float = 1.0 / 11.313708  // 1/sqrt(128)

    /// A 4-bit affine-quantized linear `[outD, inD]` applied as `x @ W^T`.
    private struct QLinear {
        let w: MLXArray, scales: MLXArray, biases: MLXArray
        let groupSize: Int
    }

    private func qlinear(_ outD: Int, _ inD: Int, _ keyOffset: Int) -> QLinear {
        let weight = MLXRandom.normal([outD, inD], key: MLXRandom.key(UInt64(keyOffset))) * 0.02
        let (wq, s, bOpt) = MLX.quantized(weight, groupSize: groupSize, bits: 4, mode: .affine)
        return QLinear(w: wq, scales: s, biases: bOpt!, groupSize: groupSize)
    }

    private func qmm(_ x: MLXArray, _ l: QLinear) -> MLXArray {
        MLX.quantizedMatmul(
            x, l.w, scales: l.scales, biases: l.biases,
            transpose: true, groupSize: l.groupSize, bits: 4, mode: .affine)
    }

    private struct Block {
        let inNorm: MLXArray, postNorm: MLXArray
        let q: QLinear, k: QLinear, v: QLinear, o: QLinear
        let gate: QLinear, up: QLinear, down: QLinear
    }

    private func makeBlock(_ seed: Int) -> Block {
        Block(
            inNorm: MLXArray.ones([hidden]).asType(dtype),
            postNorm: MLXArray.ones([hidden]).asType(dtype),
            q: qlinear(nHeads * headDim, hidden, seed + 1),
            k: qlinear(nKVHeads * headDim, hidden, seed + 2),
            v: qlinear(nKVHeads * headDim, hidden, seed + 3),
            o: qlinear(hidden, nHeads * headDim, seed + 4),
            gate: qlinear(inter, hidden, seed + 5),
            up: qlinear(inter, hidden, seed + 6),
            down: qlinear(hidden, inter, seed + 7))
    }

    private func rmsNorm(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: w, eps: 1e-5)
    }

    /// Additive causal mask `[1,1,T,L+T]`: query `i` is at absolute position
    /// `L+i` and may attend to key `j` iff `j <= L+i`.
    private func causalMask(_ T: Int, _ L: Int) -> MLXArray {
        let queryIdx = MLXArray(Int32(L) ..< Int32(L + T))           // [T]
        let keyIdx = MLXArray(Int32(0) ..< Int32(L + T))             // [L+T]
        let blocked = expandedDimensions(keyIdx, axis: 0) .> expandedDimensions(queryIdx, axis: 1)
        return (blocked.asType(dtype) * Float(-30000.0)).reshaped(1, 1, T, L + T)
    }

    /// One stacked forward over `T` new positions `[B,T,hidden]` against the
    /// fixed length-`L` per-layer histories. Returns the `[B,T,hidden]` output.
    private func stackForward(
        _ x0: MLXArray, blocks: [Block],
        kHist: [MLXArray], vHist: [MLXArray], mask: MLXArray?, T: Int
    ) -> MLXArray {
        var x = x0
        for i in 0 ..< blocks.count {
            let b = blocks[i]
            let h = rmsNorm(x, b.inNorm)
            let q = qmm(h, b.q).reshaped(B, T, nHeads, headDim).transposed(0, 2, 1, 3)
            let k = qmm(h, b.k).reshaped(B, T, nKVHeads, headDim).transposed(0, 2, 1, 3)
            let v = qmm(h, b.v).reshaped(B, T, nKVHeads, headDim).transposed(0, 2, 1, 3)
            let keys = concatenated([kHist[i], k], axis: 2)         // [B,nKVHeads,L+T,hd]
            let vals = concatenated([vHist[i], v], axis: 2)
            let attn = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: vals, scale: scale, mask: mask)
            let o = qmm(attn.transposed(0, 2, 1, 3).reshaped(B, T, nHeads * headDim), b.o)
            let hRes = x + o
            let hn = rmsNorm(hRes, b.postNorm)
            let ffn = qmm(silu(qmm(hn, b.gate)) * qmm(hn, b.up), b.down)
            x = hRes + ffn
        }
        return x
    }

    func testVerifyForwardBetaProfile() throws {
        guard ProcessInfo.processInfo.environment["KRILL_VERIFY_PROFILE"] != nil else {
            throw XCTSkip("Set KRILL_VERIFY_PROFILE to run the verify-forward beta profile")
        }
        MLXRandom.seed(0)
        let nBlocks = 28                       // Llama-3.2-3B layer count
        let ks = [1, 2, 4, 8, 16, 32]
        let iters = 60

        let blocks = (0 ..< nBlocks).map { makeBlock($0 * 100) }
        let lmHead = qlinear(vocab, hidden, 999_000)

        for L in [512, 4096] {
            // Random fixed KV history at length L (values are irrelevant to timing;
            // only the shapes / bandwidth matter). One per layer.
            let kHist = (0 ..< nBlocks).map { _ in
                (MLXRandom.normal([B, nKVHeads, L, headDim]) * 0.1).asType(dtype)
            }
            let vHist = (0 ..< nBlocks).map { _ in
                (MLXRandom.normal([B, nKVHeads, L, headDim]) * 0.1).asType(dtype)
            }
            eval(kHist + vHist)

            print("\n[verify-profile] Llama-3.2-3B dims, \(nBlocks) blocks, 4-bit, L=\(L), \(iters) iters")
            print("  K   t1(us)   tK(us)   r=tK/t1   beta    ideal(=1/beta)   lm_head(us)")

            // Establish t1 once per L (width-1 forward + width-1 lm_head).
            let t1 = timeForward(K: 1, L: L, blocks: blocks, kHist: kHist, vHist: vHist,
                                 lmHead: lmHead, iters: iters)

            for K in ks {
                let tK = timeForward(K: K, L: L, blocks: blocks, kHist: kHist, vHist: vHist,
                                     lmHead: lmHead, iters: iters)
                let tLM = timeLMHead(K: K, lmHead: lmHead, iters: iters)
                let r = tK / t1
                let beta = (r + 1.0) / Double(K + 1)
                let ideal = Double(K + 1) / (r + 1.0)
                print(String(format: "  %-3d %8.1f %8.1f %9.3f %7.3f %12.3f %12.1f",
                             K, t1, tK, r, beta, ideal, tLM))
            }
        }
    }

    /// Time a width-`K` stacked forward (incl. the K-wide lm_head), against a
    /// fixed length-`L` history. The new K positions are NOT persisted, so the
    /// history stays length `L` across iterations -> a clean, repeatable
    /// fixed-context measurement of the marginal verify-forward cost.
    private func timeForward(
        K: Int, L: Int, blocks: [Block],
        kHist: [MLXArray], vHist: [MLXArray], lmHead: QLinear, iters: Int
    ) -> Double {
        let mask = K > 1 ? causalMask(K, L) : nil
        func step() -> MLXArray {
            let x = (MLXRandom.normal([B, K, hidden]) * 0.1).asType(dtype)
            let h = stackForward(x, blocks: blocks, kHist: kHist, vHist: vHist, mask: mask, T: K)
            return qmm(h, lmHead)              // logits over all K positions
        }
        for _ in 0 ..< 8 { eval(step()) }      // warm
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< iters { eval(step()) }
        return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1e6  // us
    }

    /// Time the K-wide lm_head alone (to show its K-scaling contribution to tK).
    private func timeLMHead(K: Int, lmHead: QLinear, iters: Int) -> Double {
        func step() -> MLXArray {
            let x = (MLXRandom.normal([B, K, hidden]) * 0.1).asType(dtype)
            return qmm(x, lmHead)
        }
        for _ in 0 ..< 8 { eval(step()) }
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< iters { eval(step()) }
        return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1e6  // us
    }
}
