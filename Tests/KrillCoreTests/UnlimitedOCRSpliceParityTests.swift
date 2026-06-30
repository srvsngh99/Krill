import XCTest
import MLX
@testable import KrillCore

/// G5 base-view multimodal token assembly parity: feed the (already-validated)
/// vision features + image_newline + view_seperator and check the [273,1280]
/// assembled sequence matches the HF reference exactly (it is pure
/// reshape/concat, so this should be bit-exact). Gated on
/// KRILL_UNLIMITED_OCR_SPLICE_REF (tools/unlimited_ocr_splice_reference.py).
final class UnlimitedOCRSpliceParityTests: XCTestCase {
    func testBaseViewAssemblyMatchesHFReference() throws {
        guard let refPath = ProcessInfo.processInfo.environment["KRILL_UNLIMITED_OCR_SPLICE_REF"] else {
            throw XCTSkip("Set KRILL_UNLIMITED_OCR_SPLICE_REF (tools/unlimited_ocr_splice_reference.py)")
        }
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: refPath))
        let features = fx["features"]!.asType(.float32)        // [1,256,1280]
        let newline = fx["image_newline"]!.asType(.float32)    // [1280]
        let sep = fx["view_seperator"]!.asType(.float32)       // [1280]
        let refAsm = fx["assembled"]!.asType(.float32)         // [273,1280]

        let got = assembleBaseViewTokens(features: features, imageNewline: newline, viewSeparator: sep)
        eval(got)
        XCTAssertEqual(got.shape, refAsm.shape, "assembled shape")

        let a = got.reshaped([-1]).asArray(Float.self)
        let b = refAsm.reshaped([-1]).asArray(Float.self)
        var maxAbs = 0.0
        for i in 0 ..< a.count { maxAbs = max(maxAbs, abs(Double(a[i]) - Double(b[i]))) }
        print("[unlimited-ocr SPLICE parity] maxAbs=\(maxAbs) shape=\(got.shape)")
        XCTAssertLessThan(maxAbs, 1e-5, "assembly mismatch (max-abs \(maxAbs))")
    }
}
