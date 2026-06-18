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
}
