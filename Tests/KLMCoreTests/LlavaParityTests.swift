import XCTest
import MLX
@testable import KLMCore

/// Logit-parity check for the native LLaVA-1.5 runtime against mlx-vlm.
/// Gated on `KLM_LLAVA_PARITY_DIR`, a directory produced by
/// `tools/verify_llava_parity.py <dir>`. Validates the full LLaVA path on
/// identical weights + pixel input: the CLIP ViT vision tower (Conv2d patch
/// embed + CLS + position embedding + pre/post LayerNorm + quick-gelu MLP
/// encoder), the `vision_feature_layer=-2` + drop-CLS selection, the
/// multi-modal projector (linear -> gelu -> linear), the image-token merge into
/// the text embeddings, and the Llama text backbone. Skipped when unset.
///
/// The real llava-1.5-7b is large; the tiny synthetic LLaVA stands in for the
/// numerics (both runtimes are MLX, so parity is ~bit-exact).
final class LlavaParityTests: XCTestCase {

    private struct Reference: Decodable {
        let tokens: [Int]
        let image_token: Int
        let vocab_size: Int
        let last_token_logits: [Float]
        let argmax: Int
    }

    func testNativeLlavaMatchesMLXVLMLogits() throws {
        guard let dirPath = ProcessInfo.processInfo.environment["KLM_LLAVA_PARITY_DIR"] else {
            throw XCTSkip("Set KLM_LLAVA_PARITY_DIR (see tools/verify_llava_parity.py)")
        }
        let dir = URL(fileURLWithPath: dirPath)
        let refData = try Data(contentsOf: dir.appendingPathComponent("reference_logits.json"))
        let ref = try JSONDecoder().decode(Reference.self, from: refData)

        let loaded = try loadModel(from: dir)
        XCTAssertEqual(loaded.vocabSize, ref.vocab_size)
        guard let model = loaded.module as? LlavaForCausalLM else {
            return XCTFail("expected LlavaForCausalLM, got \(type(of: loaded.module))")
        }

        let pixels = try MLX.loadArrays(
            url: dir.appendingPathComponent("pixel_values.safetensors"))["pixel_values"]!
        let inputIds = MLXArray(ref.tokens.map { Int32($0) }).reshaped([1, ref.tokens.count])

        let logits = model(inputIds, pixelValues: pixels)
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
