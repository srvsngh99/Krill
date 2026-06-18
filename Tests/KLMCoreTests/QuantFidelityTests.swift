import XCTest
import MLX
@testable import KLMCore

/// Quant-quality measurement: how faithfully does a quantized checkpoint
/// reproduce the FULL-PRECISION (bf16) output distribution? This is the honest,
/// runtime-consistent way to compare our mixed-nvfp4 (o_proj @ 8-bit) against a
/// naive uniform 4-bit checkpoint -- both loaded through the SAME Krill
/// runtime, so our per-module overrides are actually honored (mlx-lm would
/// silently evaluate the mixed checkpoint as uniform nvfp4 and misreport it).
///
/// Metric, per token position, of each quant's logits vs the bf16 reference:
///   - top-1 agreement: does argmax match bf16's argmax (greedy-decode fidelity)
///   - KL(softmax_bf16 || softmax_quant): distributional drift (lower = better)
/// Reported as the mean over a fixed token sequence. A better quant has HIGHER
/// top-1 agreement and LOWER KL to bf16.
///
/// Gated on three dirs (skip if unset):
///   KLM_QF_BF16 = full-precision reference checkpoint
///   KLM_QF_A    = quant A (e.g. naive uniform 4-bit)
///   KLM_QF_B    = quant B (e.g. our mixed nvfp4)
final class QuantFidelityTests: XCTestCase {

    /// Deterministic in-vocab token sequence (no tokenizer needed -- quant error
    /// propagation through the weights is what we measure).
    private func fixedTokens(_ n: Int, vocab: Int) -> [Int32] {
        var t = [Int32]()
        var x: UInt64 = 0x9E3779B97F4A7C15
        for _ in 0 ..< n {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17
            t.append(Int32(2 + Int(x % UInt64(vocab - 4))))
        }
        return t
    }

    private func logits(_ dir: String, tokens: MLXArray) throws -> MLXArray {
        let loaded = try loadModel(from: URL(fileURLWithPath: dir))
        let out = loaded.forward(tokens, nil)        // [1, L, V]
        eval(out)
        return out[0]                                // [L, V]
    }

    func testMixedNvfp4VsNaive4bitFidelity() throws {
        let env = ProcessInfo.processInfo.environment
        guard let bf16 = env["KLM_QF_BF16"], let a = env["KLM_QF_A"], let b = env["KLM_QF_B"] else {
            throw XCTSkip("Set KLM_QF_BF16, KLM_QF_A, KLM_QF_B (see docs)")
        }
        let labelA = env["KLM_QF_A_LABEL"] ?? "A"
        let labelB = env["KLM_QF_B_LABEL"] ?? "B"

        // Discover vocab from the reference, build the probe sequence. Prefer a
        // REAL tokenized paragraph passed via KLM_QF_TOKENS (comma-separated ids)
        // so absolute agreement is realistic; fall back to a deterministic
        // in-vocab sequence (a harsher probe) when not provided.
        let ref0 = try loadModel(from: URL(fileURLWithPath: bf16))
        let vocab = ref0.vocabSize
        let toks: [Int32]
        if let real = env["KLM_QF_TOKENS"] {
            toks = real.split(separator: ",").compactMap { Int32($0.trimmingCharacters(in: .whitespaces)) }
        } else {
            toks = fixedTokens(64, vocab: vocab)
        }
        let tokens = MLXArray(toks).reshaped([1, toks.count])

        let refLogits = ref0.forward(tokens, nil)[0]; eval(refLogits)
        let la = try logits(a, tokens: tokens)
        let lb = try logits(b, tokens: tokens)

        func compare(_ qIn: MLXArray, _ label: String) -> (top1: Double, kl: Double) {
            let L = refLogits.dim(0)
            // Compute the metric in float32: the quant paths run in fp16/bf16 and
            // the 1e-9 KL floor underflows in fp16, spuriously sending KL to inf.
            let ref = refLogits.asType(.float32)
            let q = qIn.asType(.float32)
            let pRef = softmax(ref, axis: -1)
            let logRef = MLX.log(pRef + 1e-9)
            let logQ = MLX.log(softmax(q, axis: -1) + 1e-9)
            // KL(ref || q) = sum_v pRef * (logRef - logQ), mean over positions.
            let kl = (pRef * (logRef - logQ)).sum(axis: -1).mean()
            eval(kl)
            let refArg = refLogits.argMax(axis: -1); let qArg = q.argMax(axis: -1)
            let agree = (refArg .== qArg).asType(.float32).sum(); eval(agree)
            return (Double(agree.item(Float.self)) / Double(L), Double(kl.item(Float.self)))
        }

        let (ta, ka) = compare(la, labelA)
        let (tb, kb) = compare(lb, labelB)
        print("=== QUANT FIDELITY vs bf16 (64 tokens, vocab \(vocab)) ===")
        print(String(format: "%@: top1=%.1f%%  KL=%.5f", labelA, ta * 100, ka))
        print(String(format: "%@: top1=%.1f%%  KL=%.5f", labelB, tb * 100, kb))
        print(String(format: "lower KL = closer to full precision; KL improvement B vs A: %.1f%%",
                     (ka - kb) / ka * 100))
        // Sanity: both quants should be reasonably faithful.
        XCTAssertGreaterThan(ta, 0.5)
        XCTAssertGreaterThan(tb, 0.5)
    }
}
