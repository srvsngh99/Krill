import XCTest
import MLX
@testable import KLMCore

/// Logit-parity check for the native Llama-3.2-Vision (mllama) runtime against
/// mlx-vlm. Gated on `KLM_MLLAMA_PARITY_DIR`, a directory produced by
/// `tools/verify_mllama_parity.py <dir>`. Validates the full mllama path on
/// identical weights + vision inputs: the tiled ViT vision tower (Conv2d patch
/// embed + class token + gated aspect-ratio / position embeddings + local
/// transformer + gated global transformer + intermediate-layer concatenation),
/// the multi-modal projector, and the Llama text decoder whose
/// `cross_attention_layers` attend to the projected vision features (gated
/// cross-attention with q/k RMSNorm). The fixture randomizes ALL parameters
/// (including the gates) so the gated cross-attention genuinely contributes.
///
/// The real Llama-3.2-11B-Vision is large; the tiny synthetic mllama stands in
/// for the numerics (both runtimes are MLX, so parity is ~bit-exact).
final class MllamaParityTests: XCTestCase {

    private struct Reference: Decodable {
        let tokens: [Int]
        let vocab_size: Int
        let last_token_logits: [Float]
        let argmax: Int
    }

    func testNativeMllamaMatchesMLXVLMLogits() throws {
        guard let dirPath = ProcessInfo.processInfo.environment["KLM_MLLAMA_PARITY_DIR"] else {
            throw XCTSkip("Set KLM_MLLAMA_PARITY_DIR (see tools/verify_mllama_parity.py)")
        }
        let dir = URL(fileURLWithPath: dirPath)
        let refData = try Data(contentsOf: dir.appendingPathComponent("reference_logits.json"))
        let ref = try JSONDecoder().decode(Reference.self, from: refData)

        let loaded = try loadModel(from: dir)
        XCTAssertEqual(loaded.vocabSize, ref.vocab_size)
        guard let model = loaded.module as? Llama32VisionForCausalLM else {
            return XCTFail("expected Llama32VisionForCausalLM, got \(type(of: loaded.module))")
        }

        let inputs = try MLX.loadArrays(
            url: dir.appendingPathComponent("inputs/vision_inputs.safetensors"))
        let pixelValues = inputs["pixel_values"]!
        let aspectRatioIds = inputs["aspect_ratio_ids"]!
        let aspectRatioMask = inputs["aspect_ratio_mask"]!
        let inputIds = MLXArray(ref.tokens.map { Int32($0) }).reshaped([1, ref.tokens.count])

        let logits = model(
            inputIds, pixelValues: pixelValues,
            aspectRatioIds: aspectRatioIds, aspectRatioMask: aspectRatioMask)
        let last = logits[0, ref.tokens.count - 1, 0...]
        eval(last)
        let got = last.asArray(Float.self)
        XCTAssertEqual(got.count, ref.last_token_logits.count)

        var maxIdx = 0
        for i in 1 ..< got.count where got[i] > got[maxIdx] { maxIdx = i }
        XCTAssertEqual(maxIdx, ref.argmax,
            "native argmax \(maxIdx) != mlx-vlm argmax \(ref.argmax)")

        var dot: Double = 0, na: Double = 0, nb: Double = 0, maxAbs: Double = 0
        for i in 0 ..< got.count {
            let a = Double(got[i]), b = Double(ref.last_token_logits[i])
            dot += a * b; na += a * a; nb += b * b
            maxAbs = max(maxAbs, abs(a - b))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        XCTAssertGreaterThan(cosine, 0.9999, "logits cosine \(cosine) too low")
        XCTAssertLessThan(maxAbs, 1e-2, "max abs logit diff \(maxAbs) too large")
    }
}
