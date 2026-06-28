import XCTest
import MLX
import KrillCache
@testable import KrillCore

/// Text-only logit parity for the native Unlimited-OCR (DeepSeek-OCR) language
/// backbone vs the HF reference. Exercises the new non-MLA `DeepSeekStandardAttention`
/// path (use_mla:false: plain q/k/v/o_proj, NeoX RoPE) + the reused DeepSeek-MoE
/// stack, loaded through `loadUnlimitedOCRText` (nested language_config, vision
/// keys dropped).
///
/// Gated on two env vars (skipped when unset):
///   KRILL_UNLIMITED_OCR_DIR — the model snapshot dir (real bf16 weights)
///   KRILL_UNLIMITED_OCR_REF — the fixture from
///       `tools/unlimited_ocr_text_reference.py` (fp32 HF reference logits)
///
/// Krill runs the bf16 weights while the reference is fp32, so the gate is
/// argmax-equality + high cosine rather than bit-exactness. This is also what
/// confirms the two assumptions baked into the port: the RoPE convention
/// (traditional:false / NeoX) and full-causal masking (vs the config's
/// `sliding_window_size: 128`).
final class UnlimitedOCRTextParityTests: XCTestCase {

    private struct Reference: Decodable {
        let tokens: [Int]
        let vocab: Int
        let last_argmax: Int
        let last_logits: [Float]
    }

    func testUnlimitedOCRTextMatchesHFReference() throws {
        let env = ProcessInfo.processInfo.environment
        guard let dirPath = env["KRILL_UNLIMITED_OCR_DIR"],
              let refPath = env["KRILL_UNLIMITED_OCR_REF"] else {
            throw XCTSkip("Set KRILL_UNLIMITED_OCR_DIR (model snapshot) + "
                + "KRILL_UNLIMITED_OCR_REF (tools/unlimited_ocr_text_reference.py output)")
        }
        let ref = try JSONDecoder().decode(
            Reference.self, from: Data(contentsOf: URL(fileURLWithPath: refPath)))

        let loaded = try loadModel(from: URL(fileURLWithPath: dirPath))
        XCTAssertEqual(loaded.vocabSize, ref.vocab, "vocab size")

        let tokens = MLXArray(ref.tokens.map { Int32($0) }).reshaped([1, ref.tokens.count])
        let logits = loaded.forward(tokens, nil)
        let last = logits[0, ref.tokens.count - 1, 0...].asType(.float32)
        eval(last)
        let got = last.asArray(Float.self)
        let r = ref.last_logits
        XCTAssertEqual(got.count, r.count, "vocab length")

        var dot = 0.0, na = 0.0, nb = 0.0, maxAbs = 0.0
        var gi = 0, gMax = -Double.infinity
        for i in 0 ..< got.count {
            let x = Double(got[i]), y = Double(r[i])
            dot += x * y; na += x * x; nb += y * y
            maxAbs = max(maxAbs, abs(x - y))
            if x > gMax { gMax = x; gi = i }
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        print("[unlimited-ocr parity] argmax got=\(gi) ref=\(ref.last_argmax) "
            + "| cosine=\(cosine) | maxAbs=\(maxAbs)")

        XCTAssertEqual(gi, ref.last_argmax, "argmax mismatch (got \(gi), ref \(ref.last_argmax))")
        XCTAssertGreaterThan(cosine, 0.999, "logit cosine \(cosine) too low")
    }
}
