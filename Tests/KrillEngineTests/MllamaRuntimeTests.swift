import XCTest
import MLX
import KrillSampler
@testable import KrillCore
@testable import KrillEngine

/// End-to-end check of the native Llama-3.2-Vision decode driver
/// (`MllamaRuntime`): it must build the same cross-attention mask the model
/// forward expects and greedily sample the prefill's argmax as its first token.
/// Gated on `KRILL_MLLAMA_PARITY_DIR` (see `tools/verify_mllama_parity.py`); reuses
/// the multi-image fixture, whose recorded argmax is the oracle.
final class MllamaRuntimeTests: XCTestCase {

    private struct MultiRef: Decodable {
        let tokens: [Int]
        let image_token_id: Int
        let num_tiles: [Int]
        let argmax: Int
    }

    func testRuntimeGreedyFirstTokenMatchesMultiImagePrefill() throws {
        guard let dirPath = ProcessInfo.processInfo.environment["KRILL_MLLAMA_PARITY_DIR"] else {
            throw XCTSkip("Set KRILL_MLLAMA_PARITY_DIR (see tools/verify_mllama_parity.py)")
        }
        let dir = URL(fileURLWithPath: dirPath)
        let ref = try JSONDecoder().decode(
            MultiRef.self,
            from: try Data(contentsOf: dir.appendingPathComponent("reference_multiimage_logits.json")))

        let loaded = try loadModel(from: dir)
        guard let model = loaded.module as? Llama32VisionForCausalLM else {
            return XCTFail("expected Llama32VisionForCausalLM, got \(type(of: loaded.module))")
        }

        let inputs = try MLX.loadArrays(
            url: dir.appendingPathComponent("inputs/multiimage_inputs.safetensors"))
        let vision = MllamaProcessing.VisionInputs(
            pixelValues: inputs["pixel_values"]!,
            aspectRatioIds: inputs["aspect_ratio_ids"]!,
            aspectRatioMask: inputs["aspect_ratio_mask"]!,
            numTiles: ref.num_tiles)

        // One greedy token: the driver's prefill (cross-KV + cross mask built
        // from the prompt's <|image|> positions + last-token lm_head + sampler)
        // must reproduce the recorded multi-image argmax.
        let output = MllamaRuntime.generate(
            model: model,
            promptTokens: ref.tokens,
            vision: vision,
            maxTokens: 1,
            stopIds: [],
            params: .greedy)
        XCTAssertEqual(output.tokens.count, 1)
        XCTAssertEqual(output.tokens.first, ref.argmax,
            "runtime first token \(String(describing: output.tokens.first)) != prefill argmax \(ref.argmax)")
    }

    /// Task B (2026-09-30, docs/LOGPROBS_PLAN.md "Engine follow-ups"):
    /// `MllamaRuntime` never threaded `wantLogprobs`/`topLogprobs` at all - its
    /// `onToken` callback only ever carried a bare token id, so every
    /// Llama-3.2-Vision `TokenEvent.logprob` was `nil` regardless of the
    /// request. This is a real forward-pass check (against the tiny synthetic
    /// mllama checkpoint `tools/verify_mllama_parity.py` builds - not a full
    /// 11B checkpoint, but the actual Swift+MLX runtime code path, gated the
    /// same way as the test above) that `wantLogprobs: true` now populates
    /// `onToken`'s second argument, that greedy sampling still picks the raw
    /// top-1 alternate, and that the logprobs-OFF path samples the IDENTICAL
    /// token sequence as logprobs-ON (the reporting must never change what
    /// gets sampled).
    func testRuntimeLogprobsPopulatedAndOffPathTokensUnchanged() throws {
        guard let dirPath = ProcessInfo.processInfo.environment["KRILL_MLLAMA_PARITY_DIR"] else {
            throw XCTSkip("Set KRILL_MLLAMA_PARITY_DIR (see tools/verify_mllama_parity.py)")
        }
        let dir = URL(fileURLWithPath: dirPath)
        let ref = try JSONDecoder().decode(
            MultiRef.self,
            from: try Data(contentsOf: dir.appendingPathComponent("reference_multiimage_logits.json")))
        let loaded = try loadModel(from: dir)
        guard let model = loaded.module as? Llama32VisionForCausalLM else {
            return XCTFail("expected Llama32VisionForCausalLM, got \(type(of: loaded.module))")
        }
        let inputs = try MLX.loadArrays(
            url: dir.appendingPathComponent("inputs/multiimage_inputs.safetensors"))
        let vision = MllamaProcessing.VisionInputs(
            pixelValues: inputs["pixel_values"]!,
            aspectRatioIds: inputs["aspect_ratio_ids"]!,
            aspectRatioMask: inputs["aspect_ratio_mask"]!,
            numTiles: ref.num_tiles)

        let topN = 4
        var seenLogprobs: [TokenLogprobInfo?] = []
        let onOutput = MllamaRuntime.generate(
            model: model, promptTokens: ref.tokens, vision: vision,
            maxTokens: 5, stopIds: [], params: .greedy,
            wantLogprobs: true, topLogprobs: topN,
            onToken: { _, info in seenLogprobs.append(info) })

        XCTAssertEqual(seenLogprobs.count, onOutput.tokens.count)
        for (i, info) in seenLogprobs.enumerated() {
            guard let info else {
                XCTFail("token at index \(i) has no logprob info"); continue
            }
            XCTAssertEqual(info.topAlternates.count, topN)
            XCTAssertFalse(info.logprob.isNaN)
            XCTAssertLessThanOrEqual(info.logprob, 0.0001)
        }
        // Greedy prefill: the first yielded token's own logprob must be its
        // top-1 alternate (same consistency check as the qwen3_5/qwen2.5-vl
        // real-model logprobs tests).
        if let first = seenLogprobs.first ?? nil, let top0 = first.topAlternates.first {
            XCTAssertEqual(top0.tokenId, onOutput.tokens.first)
            XCTAssertEqual(top0.logprob, first.logprob, accuracy: 1e-5)
        }

        var offSeen: [Int] = []
        let offOutput = MllamaRuntime.generate(
            model: model, promptTokens: ref.tokens, vision: vision,
            maxTokens: 5, stopIds: [], params: .greedy,
            onToken: { tok, _ in offSeen.append(tok) })
        XCTAssertEqual(offOutput.tokens, onOutput.tokens,
            "requesting logprobs must never change which tokens are sampled")
        XCTAssertEqual(offSeen, onOutput.tokens)
    }
}
