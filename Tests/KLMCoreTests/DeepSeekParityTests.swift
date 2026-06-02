import XCTest
import MLX
import KLMCache
@testable import KLMCore

/// Logit-parity check for the native DeepSeek-V2 / V2-Lite runtime against
/// mlx-lm. Gated on `KLM_DEEPSEEK_V2_PARITY_DIR`, a directory produced by
/// `tools/verify_deepseek_parity.py <dir> v2`. Validates MLA attention (low-rank
/// KV bottleneck, split rope/nope head dims), YaRN RoPE, the shared expert, the
/// `first_k_dense_replace` dense-layer prefix, the softmax/greedy router, and
/// the `gatherQuantizedMM` SwitchGLU on identical packed weights. Skipped when
/// unset.
///
/// DeepSeek-V3 is also covered, via `KLM_DEEPSEEK_V3_PARITY_DIR`
/// (`tools/verify_deepseek_parity.py <dir> v3`): the absorbed MLA layout
/// (`embed_q` / unembed_out per-head quantized linears + the latent KV cache,
/// exercised through the L > 1 prefill path), the `q_lora_rank` query
/// bottleneck, and the `noaux_tc` sigmoid group gate (`scoring_func=sigmoid`,
/// `n_group`/`topk_group`, `norm_topk_prob`). The 671B real V3 is RAM-blocked,
/// so the tiny synthetic V3 stands in for its numerics.
final class DeepSeekParityTests: XCTestCase {

    private struct Reference: Decodable {
        let tokens: [Int]
        let vocab_size: Int
        let last_token_logits: [Float]
        let argmax: Int
    }

    private func runParity(_ envVar: String) throws {
        guard let dirPath = ProcessInfo.processInfo.environment[envVar] else {
            throw XCTSkip("Set \(envVar) (see tools/verify_deepseek_parity.py)")
        }
        let dir = URL(fileURLWithPath: dirPath)
        let refData = try Data(contentsOf: dir.appendingPathComponent("reference_logits.json"))
        let ref = try JSONDecoder().decode(Reference.self, from: refData)

        let loaded = try loadModel(from: dir)
        XCTAssertEqual(loaded.vocabSize, ref.vocab_size)

        let tokens = MLXArray(ref.tokens.map { Int32($0) }).reshaped([1, ref.tokens.count])
        let logits = loaded.forward(tokens, nil)
        let last = logits[0, ref.tokens.count - 1, 0...]
        eval(last)
        let got = last.asArray(Float.self)
        XCTAssertEqual(got.count, ref.last_token_logits.count)

        var maxIdx = 0
        for i in 1 ..< got.count where got[i] > got[maxIdx] { maxIdx = i }
        XCTAssertEqual(maxIdx, ref.argmax,
            "[\(envVar)] native argmax \(maxIdx) != mlx-lm argmax \(ref.argmax)")

        var dot: Double = 0, na: Double = 0, nb: Double = 0, maxAbs: Double = 0
        for i in 0 ..< got.count {
            let a = Double(got[i]), b = Double(ref.last_token_logits[i])
            dot += a * b; na += a * a; nb += b * b
            maxAbs = max(maxAbs, abs(a - b))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        XCTAssertGreaterThan(cosine, 0.9999, "[\(envVar)] logits cosine \(cosine) too low")
        XCTAssertLessThan(maxAbs, 1e-2, "[\(envVar)] max abs logit diff \(maxAbs) too large")
    }

    func testNativeDeepSeekV2MatchesMLXLMLogits() throws {
        try runParity("KLM_DEEPSEEK_V2_PARITY_DIR")
    }

    func testNativeDeepSeekV3MatchesMLXLMLogits() throws {
        try runParity("KLM_DEEPSEEK_V3_PARITY_DIR")
    }

    /// The mlx-lm V3 parity reference is a single full-sequence (L > 1) forward,
    /// so it only exercises the absorbed-MLA *prefill* path. This pins the
    /// *decode* path (L == 1: query projected into the latent, attention over
    /// the cached `kv_latent`, `unembed_out` back to value space) by asserting
    /// that prefilling all-but-the-last token and then decoding the last token
    /// through the latent KV cache reproduces the last-position logits of a
    /// single full-sequence prefill. The two are the same identity, so the
    /// per-step decode must match the batched prefill.
    func testNativeDeepSeekV3DecodeMatchesPrefill() throws {
        guard let dirPath = ProcessInfo.processInfo.environment["KLM_DEEPSEEK_V3_PARITY_DIR"] else {
            throw XCTSkip("Set KLM_DEEPSEEK_V3_PARITY_DIR (see tools/verify_deepseek_parity.py)")
        }
        let dir = URL(fileURLWithPath: dirPath)
        let refData = try Data(contentsOf: dir.appendingPathComponent("reference_logits.json"))
        let ref = try JSONDecoder().decode(Reference.self, from: refData)
        let loaded = try loadModel(from: dir)

        let n = ref.tokens.count
        let all = MLXArray(ref.tokens.map { Int32($0) }).reshaped([1, n])

        // Full-sequence prefill: last-position logits via the L > 1 path.
        let full = loaded.forward(all, nil)
        let prefillLast = full[0, n - 1, 0...]
        eval(prefillLast)
        let want = prefillLast.asArray(Float.self)

        // Prefill the first n-1 tokens into a per-layer latent KV cache, then
        // decode the final token alone (the L == 1 absorbed path).
        let caches: [KVCache] = (0 ..< loaded.numLayers).map { _ in KVCache() }
        let prefix = MLXArray(ref.tokens[0 ..< (n - 1)].map { Int32($0) }).reshaped([1, n - 1])
        _ = loaded.forward(prefix, caches)
        let lastTok = MLXArray([Int32(ref.tokens[n - 1])]).reshaped([1, 1])
        let step = loaded.forward(lastTok, caches)
        let stepLast = step[0, 0, 0...]
        eval(stepLast)
        let got = stepLast.asArray(Float.self)

        XCTAssertEqual(got.count, want.count)
        var maxAbs: Double = 0
        for i in 0 ..< got.count { maxAbs = max(maxAbs, abs(Double(got[i]) - Double(want[i]))) }
        XCTAssertLessThan(maxAbs, 2e-3,
            "decode (L==1) logits diverge from prefill by \(maxAbs); absorbed-MLA "
            + "latent-cache path must match the L>1 path")
    }
}
