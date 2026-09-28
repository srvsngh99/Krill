import XCTest
@testable import KrillTokenizer

/// `KrillTokenizer.rawBytes(forPiece:isByteLevelBPE:)` backs the OpenAI
/// `logprobs.content[].bytes` field (docs/LOGPROBS_PLAN.md §4.2). It must
/// recover the exact bytes a raw vocabulary piece represents, independent of
/// the lossy `decode`/`decodeForOutput` path — tested here as a pure function
/// of the piece string (+ the tokenizer-wide byte-level-BPE flag), so no real
/// tokenizer/vocab needs to be on disk.
final class RawTokenBytesTests: XCTestCase {

    // MARK: - SentencePiece byte-fallback (`<0xHH>`) — style-independent

    func testByteFallbackASCII() {
        XCTAssertEqual(KrillTokenizer.rawBytes(forPiece: "<0x0A>", isByteLevelBPE: false), [0x0A]) // newline
        XCTAssertEqual(KrillTokenizer.rawBytes(forPiece: "<0x41>", isByteLevelBPE: false), [0x41]) // 'A'
    }

    func testByteFallbackNonASCIISingleByte() {
        // A single byte of a multi-byte UTF-8 sequence, e.g. the first byte of
        // "é" (0xC3 0xA9). decodeForOutput would drop this (see
        // recoverByteFallback's doc comment); rawBytes must not.
        XCTAssertEqual(KrillTokenizer.rawBytes(forPiece: "<0xC3>", isByteLevelBPE: false), [0xC3])
        XCTAssertEqual(KrillTokenizer.rawBytes(forPiece: "<0xA9>", isByteLevelBPE: false), [0xA9])
    }

    func testByteFallbackCaseInsensitiveHex() {
        XCTAssertEqual(KrillTokenizer.rawBytes(forPiece: "<0xff>", isByteLevelBPE: false), [0xFF])
        XCTAssertEqual(KrillTokenizer.rawBytes(forPiece: "<0xFF>", isByteLevelBPE: false), [0xFF])
    }

    func testByteFallbackTakesPriorityEvenWhenByteLevel() {
        // A byte-fallback literal cannot occur in a genuine byte-level-BPE
        // vocab, but the check must still take priority over the byte-level
        // branch if it somehow appears.
        XCTAssertEqual(KrillTokenizer.rawBytes(forPiece: "<0x0A>", isByteLevelBPE: true), [0x0A])
    }

    // MARK: - Byte-level BPE (GPT-2 Ġ/Ċ-style, Qwen/Llama-3-family)

    func testByteLevelBPESpaceAndWord() {
        // "Ġhello" is the byte-level-BPE encoding of " hello" (Ġ = U+0120 = byte 0x20).
        let bytes = KrillTokenizer.rawBytes(forPiece: "\u{0120}hello", isByteLevelBPE: true)
        XCTAssertEqual(String(decoding: bytes, as: UTF8.self), " hello")
        XCTAssertEqual(bytes, [0x20, 0x68, 0x65, 0x6C, 0x6C, 0x6F])
    }

    func testByteLevelBPENewline() {
        // "Ċ" (U+010A) is the byte-level-BPE encoding of byte 0x0A (newline).
        XCTAssertEqual(KrillTokenizer.rawBytes(forPiece: "\u{010A}", isByteLevelBPE: true), [0x0A])
    }

    func testByteLevelBPEMultibyteCharacter() {
        // "é" is UTF-8 bytes [0xC3, 0xA9]. Both bytes fall in the GPT-2 table's
        // "identity" range (printable Latin-1), so the byte-level-BPE piece is
        // literally "Ã©" (U+00C3, U+00A9) - each character maps back to its own
        // byte value. This is the "multibyte char" case via the byte-level path.
        let piece = "\u{00C3}\u{00A9}"
        let bytes = KrillTokenizer.rawBytes(forPiece: piece, isByteLevelBPE: true)
        XCTAssertEqual(bytes, [0xC3, 0xA9])
        XCTAssertEqual(String(decoding: bytes, as: UTF8.self), "\u{00E9}") // "é"
    }

    func testByteLevelBPEUnmappableCharacterFallsBackToUTF8() {
        // A character truly outside the 256-entry table (shouldn't happen for
        // a real byte-level-BPE piece) must not crash or drop data.
        let piece = "abc\u{1F419}" // trailing octopus emoji, not in the table
        let bytes = KrillTokenizer.rawBytes(forPiece: piece, isByteLevelBPE: true)
        XCTAssertEqual(bytes, Array(piece.utf8))
    }

    // MARK: - SentencePiece `▁`-prefixed piece

    func testSentencePieceLeadingSpace() {
        let bytes = KrillTokenizer.rawBytes(forPiece: "\u{2581}world", isByteLevelBPE: false)
        XCTAssertEqual(String(decoding: bytes, as: UTF8.self), " world")
        XCTAssertEqual(bytes.first, 0x20)
    }

    func testSentencePieceBareUnderscoreOnly() {
        XCTAssertEqual(KrillTokenizer.rawBytes(forPiece: "\u{2581}", isByteLevelBPE: false), [0x20])
    }

    func testSentencePieceMultibyteCharacterAfterMarker() {
        // A SentencePiece piece can carry a literal multi-byte UTF-8 character
        // directly (not byte-level-remapped) after the ▁ space marker. Note
        // this is the SAME two characters ("é") as the byte-level test above,
        // but with isByteLevelBPE: false it must decode to the literal
        // 2-byte UTF-8 form, not the byte-level-remapped single byte 0xE9.
        let piece = "\u{2581}\u{00E9}" // "▁é"
        let bytes = KrillTokenizer.rawBytes(forPiece: piece, isByteLevelBPE: false)
        XCTAssertEqual(String(decoding: bytes, as: UTF8.self), " \u{00E9}")
        XCTAssertEqual(bytes, [0x20] + Array("\u{00E9}".utf8))
    }

    // MARK: - Plain UTF-8 fallback (no markers at all)

    func testPlainASCIIPiece() {
        XCTAssertEqual(KrillTokenizer.rawBytes(forPiece: "hello", isByteLevelBPE: false), Array("hello".utf8))
    }

    func testPlainMultibyteCharacterPiece() {
        // A raw literal multi-byte character with none of the special markers
        // (e.g. a WordPiece/BERT-style vocab entry).
        let bytes = KrillTokenizer.rawBytes(forPiece: "\u{1F419}", isByteLevelBPE: false) // 🐙, 4-byte UTF-8
        XCTAssertEqual(bytes, Array("\u{1F419}".utf8))
        XCTAssertEqual(bytes.count, 4)
    }

    func testEmptyPieceReturnsEmptyBytes() {
        XCTAssertEqual(KrillTokenizer.rawBytes(forPiece: "", isByteLevelBPE: false), [])
        XCTAssertEqual(KrillTokenizer.rawBytes(forPiece: "", isByteLevelBPE: true), [])
    }

    // MARK: - lossyTokenString

    func testLossyTokenStringReplacesInvalidUTF8() {
        // A lone byte-fallback byte from a multi-byte sequence is not valid
        // UTF-8 on its own; the lossy string must use U+FFFD, not crash or
        // silently drop it (bytes stays exact regardless).
        let bytes = KrillTokenizer.rawBytes(forPiece: "<0xC3>", isByteLevelBPE: false)
        XCTAssertEqual(bytes, [0xC3])
        XCTAssertEqual(String(decoding: bytes, as: UTF8.self), "\u{FFFD}")
    }

    func testLossyTokenStringValidUTF8RoundTrips() {
        let bytes = KrillTokenizer.rawBytes(forPiece: "\u{0120}hi", isByteLevelBPE: true)
        XCTAssertEqual(String(decoding: bytes, as: UTF8.self), " hi")
    }
}
