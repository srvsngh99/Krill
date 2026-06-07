import XCTest
@testable import KLMEngine

/// Gates the Gemma 4 media-placeholder prefix construction - in particular
/// the begin/end markers the encoder-free "unified" SKU REQUIRES around each
/// soft-token run, and the guarantee that the marker wrapping is scoped to
/// that family so the e2b/e4b bare-run behavior is unchanged.
///
/// The token-id round-trip of the marker strings (e.g. `<|image>` -> 255999)
/// needs the real Gemma tokenizer and is gated by
/// Gemma4UnifiedLiveDiag.testMediaMarkerTokenIds (env-gated). Here we gate the
/// STRUCTURE, which is what the serving path emits.
final class Gemma4UnifiedMarkersTests: XCTestCase {

    func testUnifiedImagePrefixIsWrappedWithBeginEndMarkers() {
        let prefix = InferenceEngine.mediaPlaceholderPrefix(
            imageTokenCount: 4, audioTokenCount: nil, wrapGemma4Markers: true)
        let expected = InferenceEngine.gemma4BeginImage
            + String(repeating: InferenceEngine.gemma4ImageSoftToken, count: 4)
            + InferenceEngine.gemma4EndImage
        XCTAssertEqual(prefix, expected)
        // Exactly one begin and one end marker, around exactly N soft tokens.
        XCTAssertTrue(prefix.hasPrefix("<|image>"))
        XCTAssertTrue(prefix.hasSuffix("<image|>"))
        XCTAssertEqual(
            prefix.components(separatedBy: "<|image|>").count - 1, 4,
            "must emit exactly imageTokenCount soft tokens")
    }

    func testUnifiedAudioPrefixIsWrappedWithBeginEndMarkers() {
        let prefix = InferenceEngine.mediaPlaceholderPrefix(
            imageTokenCount: nil, audioTokenCount: 3, wrapGemma4Markers: true)
        XCTAssertEqual(
            prefix,
            "<|audio>" + String(repeating: "<|audio|>", count: 3) + "<audio|>")
    }

    /// Critical e2b/e4b-safety invariant: with the markers OFF (the default for
    /// every non-unified family), the prefix is a BARE soft-token run with no
    /// begin/end markers, byte-identical to the pre-unified behavior.
    func testNonUnifiedPrefixHasNoMarkers() {
        let prefix = InferenceEngine.mediaPlaceholderPrefix(
            imageTokenCount: 5, audioTokenCount: 2, wrapGemma4Markers: false)
        XCTAssertEqual(
            prefix,
            String(repeating: "<|image|>", count: 5)
                + String(repeating: "<|audio|>", count: 2))
        XCTAssertFalse(prefix.contains("<|image>"), "no begin-image marker")
        XCTAssertFalse(prefix.contains("<image|>"), "no end-image marker")
        XCTAssertFalse(prefix.contains("<|audio>"), "no begin-audio marker")
        XCTAssertFalse(prefix.contains("<audio|>"), "no end-audio marker")
    }

    func testCombinedImageAndAudioOrdering() {
        let prefix = InferenceEngine.mediaPlaceholderPrefix(
            imageTokenCount: 1, audioTokenCount: 1, wrapGemma4Markers: true)
        // Image run precedes the audio run.
        XCTAssertEqual(prefix, "<|image><|image|><image|><|audio><|audio|><audio|>")
    }

    func testEmptyWhenNoMedia() {
        XCTAssertEqual(
            InferenceEngine.mediaPlaceholderPrefix(
                imageTokenCount: nil, audioTokenCount: nil, wrapGemma4Markers: true),
            "")
    }
}
