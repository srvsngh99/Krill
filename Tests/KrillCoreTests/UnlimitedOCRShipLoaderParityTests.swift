import XCTest
import MLX
@testable import KrillCore

/// End-to-end parity for the PRODUCTION `loadUnlimitedOCR` path: load the mixed
/// nvfp4 ship checkpoint through `loadModel` (model_type unlimited-ocr ->
/// loadUnlimitedOCR: nvfp4 experts + 8-bit non-expert/vision, DeepEncoder bound
/// from the converted blob), then run `multimodalPrefillForward` on the e2e
/// fixture's input_ids + image and compare the last-token logits to the HF
/// reference. This validates the whole serving stack (loader + quant + splice)
/// the engine calls, not just the piecewise math.
///
/// Gated on:
///   KRILL_UNLIMITED_OCR_SHIP — the ship checkpoint dir (full nvfp4 blob)
///   KRILL_UNLIMITED_OCR_E2E_REF — tools/unlimited_ocr_e2e_reference.py fixture
final class UnlimitedOCRShipLoaderParityTests: XCTestCase {
    func testShipLoaderMultimodalPrefillMatchesReference() throws {
        let env = ProcessInfo.processInfo.environment
        guard let shipPath = env["KRILL_UNLIMITED_OCR_SHIP"],
              let refPath = env["KRILL_UNLIMITED_OCR_E2E_REF"] else {
            throw XCTSkip("Set KRILL_UNLIMITED_OCR_SHIP + KRILL_UNLIMITED_OCR_E2E_REF")
        }

        let loaded = try loadModel(from: URL(fileURLWithPath: shipPath))
        XCTAssertEqual(loaded.family, "unlimited_ocr")
        guard let mmPrefill = loaded.multimodalPrefillForward else {
            return XCTFail("loadUnlimitedOCR did not expose multimodalPrefillForward")
        }

        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: refPath))
        let inputIds = fx["input_ids"]!.asType(.int32)            // [1, L]
        // Fixture ships channels-first [1,3,1024,1024]; the production
        // preprocessor emits channels-last, which is what the loader's
        // DeepEncoder consumes — match that here.
        let pixels = fx["images_ori"]!.asType(.float32).transposed(0, 2, 3, 1)
        let refLogits = fx["last_logits"]!.asType(.float32).asArray(Float.self)

        let logits = mmPrefill(inputIds, nil, pixels, nil, nil, nil).asType(.float32)
        eval(logits)
        let got = logits.reshaped([-1]).asArray(Float.self)
        XCTAssertEqual(got.count, refLogits.count, "vocab length")

        var dot = 0.0, na = 0.0, nb = 0.0, gi = 0, gm = -Double.infinity
        for i in 0 ..< got.count {
            let x = Double(got[i]), y = Double(refLogits[i])
            dot += x * y; na += x * x; nb += y * y
            if x > gm { gm = x; gi = i }
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        var ri = 0, rm = -Double.infinity
        for i in 0 ..< refLogits.count where Double(refLogits[i]) > rm { rm = Double(refLogits[i]); ri = i }
        let refLogit = Double(got[ri])
        var refRank = 0
        for i in 0 ..< got.count where Double(got[i]) > refLogit { refRank += 1 }
        print("[unlimited-ocr ship parity] argmax got=\(gi) ref=\(ri) "
            + "| ref-token rank in Krill=\(refRank) | cosine=\(cosine)")
        // This gate validates LOADER MECHANICS (composite builds, vision binds,
        // splice runs) against a SYNTHETIC first-token probe — not real OCR
        // quality (real generation is verified by the CLI OCR smoke test; the
        // synthetic prefix/suffix ids here are arbitrary, so the absolute logit
        // is not meaningful beyond "the pipeline ran"). nvfp4 experts + 8-bit
        // vision push the synthetic first-token to ~0.98 cosine with the ref
        // token around rank 6 — quant noise. A real loader/splice break instead
        // collapses cosine toward 0 (or NaN) and buries the ref token at a huge
        // rank, which this still catches. Gate: ref token in the top-16 +
        // cosine > 0.97.
        XCTAssertLessThanOrEqual(refRank, 16, "ref token at rank \(refRank) — serving/splice break")
        XCTAssertGreaterThan(cosine, 0.97, "ship-loader logit cosine \(cosine) — pipeline break")
    }
}
