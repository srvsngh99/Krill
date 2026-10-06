import XCTest
import Foundation
import KrillCore
import KrillTokenizer

/// `EG2SequenceBuilder`: placeholder positions and counts, text merging,
/// prefix rules. Uses a tiny synthetic Gemma-shaped tokenizer, so it needs no
/// weights. (The ids against sentence-transformers' own live in the parity test.)
final class EmbeddingGemma2SequenceTests: XCTestCase {
    private let T = EG2ModalityTokens.checkpointDefaults

    /// Vocab: ids 0-3 specials/space, letters, one merge ("\u{2581}a").
    private func makeTokenizer() throws -> CodePointBPETokenizer {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("eg2-tok-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        var vocab: [String: Int] = ["<pad>": 0, "<eos>": 1, "<bos>": 2, "\u{2581}": 3,
                                    "a": 4, "b": 5, "c": 6, ":": 7, "x": 8, "\u{2581}a": 9]
        let specials: [(String, Int)] = [
            ("<|image>", T.boi), ("<|image|>", T.image), ("<image|>", T.eoi),
            ("<|audio>", T.boa), ("<|audio|>", T.audio), ("<audio|>", T.eoa), ("<|video|>", T.video),
            ("<pad>", 0), ("<eos>", 1), ("<bos>", 2),
        ]
        for (s, id) in specials { vocab[s] = id }
        let doc: [String: Any] = [
            "normalizer": ["type": "Replace", "pattern": ["String": " "], "content": "\u{2581}"],
            "model": ["type": "BPE", "byte_fallback": true, "vocab": vocab,
                      "merges": [["\u{2581}", "a"]]],
            "added_tokens": specials.map { ["id": $0.1, "content": $0.0, "special": true] },
            "post_processor": ["type": "TemplateProcessing",
                               "single": [["SpecialToken": ["id": "<bos>"]], ["Sequence": ["id": "A"]],
                                          ["SpecialToken": ["id": "<eos>"]]]],
        ]
        try JSONSerialization.data(withJSONObject: doc).write(to: dir.appendingPathComponent("tokenizer.json"))
        return try CodePointBPETokenizer(directory: dir)
    }

    private func builder(max: Int = 8192) throws -> EG2SequenceBuilder {
        EG2SequenceBuilder(tokenizer: try makeTokenizer(), tokens: T, maxTokens: max)
    }

    private func img(_ n: Int) -> EG2Segment { .media(EG2MediaBlock(.image, softTokensPerBlock: n)) }

    func testImageOnlyLayout() throws {
        let s = try builder().build([img(5)])
        XCTAssertEqual(s.ids, [2, Int32(T.boi)] + [Int32](repeating: Int32(T.image), count: 5) + [Int32(T.eoi), 1])
        XCTAssertEqual(s.spans, [EG2SoftSpan(modality: .image, start: 2, count: 5)])
        XCTAssertEqual(s.softTokenCount(.image), 5)
        XCTAssertEqual(s.softTokenCount(.audio), 0)
    }

    func testTextImageTextAndSpanPositions() throws {
        let b = try builder()
        let s = try b.build([.text("a"), img(3), .text("b")])
        XCTAssertEqual(s.ids, [2, 4, Int32(T.boi), Int32(T.image), Int32(T.image), Int32(T.image),
                               Int32(T.eoi), 5, 1])
        XCTAssertEqual(s.spans, [EG2SoftSpan(modality: .image, start: 3, count: 3)])
        // every span slot holds the placeholder id, every other slot does not
        for (i, id) in s.ids.enumerated() {
            let inSpan = s.spans.contains { i >= $0.start && i < $0.start + $0.count }
            XCTAssertEqual(id == Int32(T.image), inSpan, "position \(i)")
        }
    }

    func testTwoImagesKeepOrderAndCounts() throws {
        let s = try builder().build([img(4), .text("c"), img(2)])
        XCTAssertEqual(s.spans.map { $0.count }, [4, 2])
        XCTAssertEqual(s.spans.map { $0.start }, [2, 2 + 4 + 1 + 1 + 1])  // bos boi [4] eoi c boi ...
        XCTAssertEqual(s.softTokenCount(.image), 6)
        XCTAssertEqual(s.count, 1 + (1 + 4 + 1) + 1 + (1 + 2 + 1) + 1)
    }

    func testTouchingTextIsTokenisedJointly() throws {
        let tok = try makeTokenizer()
        let b = EG2SequenceBuilder(tokenizer: tok, tokens: T)
        // "a" + " a" is ONE string "a a": a, then the merged "\u{2581}a" token.
        let s = try b.build([.text("a"), .text(" a")])
        XCTAssertEqual(s.ids.map { Int($0) }, tok.encode("a a"))
        XCTAssertEqual(s.ids, [2, 4, 9, 1])
        // "a " + "a" joins across the boundary ("a a" -> a, \u{2581}a) ...
        XCTAssertEqual(try b.build([.text("a "), .text("a")]).ids, [2, 4, 9, 1])
        // ... but media break the string, so no merge forms across it
        let t = try b.build([.text("a "), img(1), .text("a")])
        XCTAssertEqual(t.ids, [2, 4, 3, Int32(T.boi), Int32(T.image), Int32(T.eoi), 4, 1])
    }

    func testPrefixOnlyWhenThereIsText() throws {
        let b = try builder()
        let tok = try makeTokenizer()
        // image only: no prefix at all
        XCTAssertEqual(try b.build([img(2)], prefix: "a:").ids, try b.build([img(2)]).ids)
        // text + image: prefix glued in front of the FIRST text (jointly tokenised)
        let s = try b.build([.text("b"), img(2)], prefix: "a:")
        XCTAssertEqual(Array(s.ids.prefix(4)), [2] + tok.encode("a:b", addSpecialTokens: false).map { Int32($0) })
        // image first, then text: the prefix still comes first (before <boi>)
        let u = try b.build([img(2), .text("b")], prefix: "a:")
        XCTAssertEqual(Array(u.ids.prefix(3)), [2, 4, 7])
        XCTAssertEqual(u.ids[3], Int32(T.boi))
        XCTAssertEqual(u.spans[0].start, 4)
    }

    func testVideoAndAudioBlocksFollowTheReferenceLayout() throws {
        let b = try builder()
        // video: one <boi> <video>xN <eoi> block per frame
        let v = try b.build([.media(EG2MediaBlock(.video, softTokensPerBlock: 3, blocks: 2))])
        let block: [Int32] = [Int32(T.boi), Int32(T.video), Int32(T.video), Int32(T.video), Int32(T.eoi)]
        XCTAssertEqual(v.ids, [2] + block + block + [1])
        XCTAssertEqual(v.spans.map { $0.start }, [2, 7])
        XCTAssertEqual(v.softTokenCount(.video), 6)
        // audio: <boa> <audio>xN <eoa>
        let a = try b.build([.media(EG2MediaBlock(.audio, softTokensPerBlock: 2))])
        XCTAssertEqual(a.ids, [2, Int32(T.boa), Int32(T.audio), Int32(T.audio), Int32(T.eoa), 1])
    }

    func testReservedPlaceholdersInTextAreRejected() throws {
        let b = try builder()
        for lit in EG2ModalityTokens.reservedLiterals {
            XCTAssertThrowsError(try b.build([.text("a\(lit)b")]), lit) {
                guard case EG2SequenceError.reservedLiteral = $0 else { return XCTFail("\($0)") }
            }
        }
    }

    func testOverLongInputIsRejectedNotTruncated() throws {
        let b = try builder(max: 10)
        XCTAssertNoThrow(try b.build([img(6)]))  // 1 + 1 + 6 + 1 + 1 = 10
        XCTAssertThrowsError(try b.build([img(7)])) {
            XCTAssertEqual($0 as? EG2SequenceError, .tooLong(tokens: 11, limit: 10))
        }
    }

    func testEmptyInputIsRejected() throws {
        XCTAssertThrowsError(try builder().build([]))
    }
}
