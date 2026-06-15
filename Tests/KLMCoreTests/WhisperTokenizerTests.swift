import XCTest
@testable import KLMCore

/// CI-runnable unit test for the GPT-2 byte-level detokenizer: no model weights
/// needed, just a tiny in-memory `vocab.json`. Pins the byte mapping (`Ġ` ->
/// space), multi-byte UTF-8 reassembly, the `<|endoftext|>` stop, and
/// special/timestamp-token suppression.
final class WhisperTokenizerTests: XCTestCase {

    /// Build a tokenizer from a temporary vocab file.
    private func makeTokenizer(_ vocab: [String: Int]) throws -> WhisperTokenizer {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("whisper-vocab-\(UUID().uuidString).json")
        try JSONEncoder().encode(vocab).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        return try WhisperTokenizer(vocabURL: url)
    }

    func testDecodesByteLevelTokens() throws {
        // Real whisper-small.en ids for "the quick brown fox".
        let tok = try makeTokenizer([
            "the": 1169, "Ġquick": 2068, "Ġbrown": 7586, "Ġfox": 21831,
        ])
        XCTAssertEqual(tok.decode([1169, 2068, 7586, 21831]), "the quick brown fox")
    }

    func testStopsAtEndOfTextAndDropsSpecials() throws {
        let tok = try makeTokenizer(["the": 1169, "Ġfox": 21831])
        // endOfText (50256) halts; anything after is ignored.
        XCTAssertEqual(tok.decode([1169, WhisperTokenizer.endOfText, 21831]), "the")
        // A special/timestamp id with no text mapping is skipped, not crashed.
        XCTAssertEqual(tok.decode([1169, WhisperTokenizer.transcribe, 21831]), "the fox")
    }

    func testReassemblesMultiByteUTF8() throws {
        // GPT-2 byte-level: "é" (U+00E9 -> UTF-8 0xC3 0xA9) encodes as the two
        // byte-unicode scalars "Ã©". The decoder must rebuild the raw bytes.
        let tok = try makeTokenizer(["\u{00C3}\u{00A9}": 5])
        XCTAssertEqual(tok.decode([5]), "\u{00E9}")
    }
}
