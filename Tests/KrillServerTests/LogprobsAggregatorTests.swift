import XCTest
import KrillEngine
import KrillSampler
@testable import KrillServer

/// A fully synthetic `TokenLogprobResolver` for driving `LogprobsAggregator`
/// without a real loaded tokenizer/model (docs/LOGPROBS_PLAN.md finding #1).
/// Each synthetic token id maps to an explicit byte sequence, which can
/// differ from whatever text was fed to the filter for it - exactly the
/// byte-fallback / partial-UTF-8 case this fix targets.
private final class FakeResolver: TokenLogprobResolver {
    var bytesByToken: [Int: [UInt8]] = [:]
    var suppressedIds: Set<Int> = []

    func rawTokenBytes(for tokenId: Int) -> [UInt8]? { bytesByToken[tokenId] }
    func lossyTokenString(bytes: [UInt8]) -> String { String(decoding: bytes, as: UTF8.self) }
    func isOutputSuppressedToken(_ tokenId: Int) -> Bool { suppressedIds.contains(tokenId) }
}

/// `LogprobsAggregator` attributes a `logprobs.content[]` entry to every
/// generated token that reaches the visible answer, and none to reasoning-
/// block or suppressed tokens - even when the reasoning filter holds or
/// splits text across several `TokenEvent`s, and even when a token's own
/// `decodeForOutput` text is empty (finding #1). The core invariant these
/// tests check: concatenating every entry's `bytes` reproduces the visible
/// `content` string byte-for-byte, whenever `content` itself actually
/// contains what those tokens represent (see
/// `testEmptyDecodeTokenKeepsOwnEntryEvenThoughContentDropsIt` for the one
/// documented exception: a pre-existing gap in `decodeForOutput` that drops
/// a multi-byte SentencePiece byte-fallback character from `content`
/// entirely - not something this fix changes, but no longer also silently
/// dropped from `logprobs.content[]`).
final class LogprobsAggregatorTests: XCTestCase {
    private var nextId = 1000
    private func freshId() -> Int { nextId += 1; return nextId }

    private func info(_ lp: Float = -0.1) -> TokenLogprobInfo {
        TokenLogprobInfo(logprob: lp, topAlternates: [])
    }

    /// Feed `tokens` (id, text) through a fresh aggregator (registering each
    /// token's raw bytes as the UTF-8 bytes of its own text, unless
    /// overridden via `bytesOverride`), then `finish()`. Returns the visible
    /// content a real client would see, the concatenation of every entry's
    /// `bytes`, and the raw entries themselves.
    @discardableResult
    private func run(
        _ tokens: [(id: Int, text: String)],
        suppressed: Set<Int> = [],
        bytesOverride: [Int: [UInt8]] = [:]
    ) -> (content: String, bytesConcat: [UInt8], entries: [[String: Any]]) {
        let resolver = FakeResolver()
        resolver.suppressedIds = suppressed
        for (id, text) in tokens {
            resolver.bytesByToken[id] = bytesOverride[id] ?? Array(text.utf8)
        }
        let agg = LogprobsAggregator(engine: resolver, enabled: true)
        var content = ""
        for (id, text) in tokens {
            content += agg.consume(TokenEvent(tokenId: id, text: text, elapsed: 0, logprob: info()))
        }
        content += agg.finish()
        var bytesConcat: [UInt8] = []
        for entry in agg.entries {
            guard let bytes = entry["bytes"] as? [Int] else { continue }
            bytesConcat.append(contentsOf: bytes.map { UInt8($0) })
        }
        return (content, bytesConcat, agg.entries)
    }

    private func assertBytesReproduceContent(
        _ result: (content: String, bytesConcat: [UInt8], entries: [[String: Any]]),
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(
            result.bytesConcat, Array(result.content.utf8),
            "entries' bytes must concatenate to exactly the visible content",
            file: file, line: line)
    }

    // MARK: (i) Plain English

    func testPlainEnglish() {
        let a = freshId(), b = freshId(), c = freshId()
        let result = run([(a, "Hello"), (b, ", "), (c, "world!")])
        XCTAssertEqual(result.content, "Hello, world!")
        XCTAssertEqual(result.entries.count, 3)
        assertBytesReproduceContent(result)
    }

    // MARK: (ii) Code with `<`, `x < y`, and an HTML tag

    func testCodeWithAngleBrackets() {
        // `<` alone is a valid prefix of every recognized reasoning tag
        // (`<thinking>`, `<think>`, `<|channel>`, `<|think|>`), so the
        // filter HOLDS it until the next chunk disambiguates - this is
        // finding #1b's core scenario: neither token may be dropped.
        let t1 = freshId(), t2 = freshId(), t3 = freshId(), t4 = freshId()
        let result = run([
            (t1, "if x "), (t2, "<"), (t3, " y"), (t4, ":\n  return \"<div>ok</div>\"\n"),
        ])
        XCTAssertEqual(result.content, "if x < y:\n  return \"<div>ok</div>\"\n")
        XCTAssertEqual(result.entries.count, 4, "the held '<' token must not be dropped")
        assertBytesReproduceContent(result)
    }

    // MARK: (iii) Hindi / Devanagari (non-empty decode - a byte-level-BPE-
    // style tokenizer decodes real script text directly, no byte-fallback
    // assembly needed; the byte-fallback case is its own dedicated test
    // below, since it is a separate, pre-existing content-gap concern).

    func testDevanagariText() {
        let a = freshId(), b = freshId()
        // "नमस्ते" split across two tokens, each with real (non-empty) text.
        let result = run([(a, "नम"), (b, "स्ते")])
        XCTAssertEqual(result.content, "नमस्ते")
        XCTAssertEqual(result.entries.count, 2)
        assertBytesReproduceContent(result)
    }

    // MARK: (iv) Emoji

    func testEmoji() {
        let a = freshId(), b = freshId()
        let result = run([(a, "Great "), (b, "🎉")])
        XCTAssertEqual(result.content, "Great 🎉")
        XCTAssertEqual(result.entries.count, 2)
        assertBytesReproduceContent(result)
    }

    // MARK: (v) `<think>...</think>` block before the answer

    func testThinkBlockOnlyAnswerGetsEntries() {
        let think1 = freshId(), think2 = freshId(), close = freshId(), answer = freshId()
        let result = run([
            (think1, "<think>"),
            (think2, "Because 2+2=4, "),
            (close, "</think>"),
            (answer, "The answer is 4."),
        ])
        XCTAssertEqual(result.content, "The answer is 4.")
        XCTAssertEqual(result.entries.count, 1, "only the answer token gets an entry")
        assertBytesReproduceContent(result)
    }

    func testThinkBlockSplitAcrossManyTokens() {
        // The opening tag itself split across two tokens ("<thi" + "nk>"),
        // and reasoning content split across several more - every one of
        // them must still resolve to "no entry", and the answer must still
        // get its own entry once the block closes.
        let ids = (0 ..< 7).map { _ in freshId() }
        let result = run([
            (ids[0], "<thi"), (ids[1], "nk>"),
            (ids[2], "step one, "), (ids[3], "step two, "), (ids[4], "step three "),
            (ids[5], "</think>"),
            (ids[6], "Done."),
        ])
        XCTAssertEqual(result.content, "Done.")
        XCTAssertEqual(result.entries.count, 1)
        assertBytesReproduceContent(result)
    }

    // MARK: (vi) Held then flushed at end of stream

    func testHeldTextFlushedAtFinish() {
        // A stream that ends (max_tokens truncation) while the filter is
        // still holding a `<` prefix scan: `finish()` must flush it as
        // literal text (matching `StreamingReasoningFilter.finish()`) AND
        // attribute its entry - previously this was explicitly "never
        // attributed."
        let a = freshId(), b = freshId()
        let result = run([(a, "score "), (b, "<")])
        XCTAssertEqual(result.content, "score <")
        XCTAssertEqual(result.entries.count, 2, "the token held at EOS must still get an entry")
        assertBytesReproduceContent(result)
    }

    func testTruncatedInsideReasoningBlockGetsNoEntries() {
        // A stream truncated INSIDE an unterminated reasoning block: the
        // held reasoning content is discarded by `finish()`, not emitted -
        // it must get no entries either.
        let a = freshId(), b = freshId()
        let result = run([(a, "<think>"), (b, "unfinished thought")])
        XCTAssertEqual(result.content, "")
        XCTAssertEqual(result.entries.count, 0)
    }

    // MARK: Empty-decode tokens (finding #1a)

    func testEmptyDecodeTokenKeepsOwnEntryEvenThoughContentDropsIt() {
        // Simulates a SentencePiece byte-fallback multi-byte character: each
        // piece decodes to "" (the documented, pre-existing gap in
        // `decodeForOutput`/`recoverByteFallback` - NOT something this fix
        // changes), but each one must still get its OWN entry with its true
        // recovered bytes, so a client reconstructing from `bytes` gets the
        // correct text even though Krill's own `content` string does not.
        let hi = freshId(), b0 = freshId(), b1 = freshId(), b2 = freshId(), bang = freshId()
        // "न" (U+0928) = UTF-8 [0xE0, 0xA4, 0xA8].
        let result = run(
            [(hi, "Hi "), (b0, ""), (b1, ""), (b2, ""), (bang, "!")],
            bytesOverride: [b0: [0xE0], b1: [0xA4], b2: [0xA8]])
        // Pre-existing, documented gap: content is missing the character
        // entirely because every contributing token's OWN decode was "".
        XCTAssertEqual(result.content, "Hi !")
        // But every one of the empty-decode tokens still got its own entry -
        // this is the actual fix (they used to be silently dropped).
        XCTAssertEqual(result.entries.count, 5)
        // Decoding the FULL bytes-concat (not comparing to `content`, which
        // is known-deficient here) recovers the TRUE intended text.
        XCTAssertEqual(String(decoding: result.bytesConcat, as: UTF8.self), "Hi न!")
    }

    // MARK: Suppressed / special tokens

    func testSuppressedTokenNeverGetsEntry() {
        let visible1 = freshId(), marker = freshId(), visible2 = freshId()
        let result = run(
            [(visible1, "before "), (marker, ""), (visible2, "after")],
            suppressed: [marker])
        XCTAssertEqual(result.content, "before after")
        XCTAssertEqual(result.entries.count, 2)
        assertBytesReproduceContent(result)
    }

    // MARK: Disabled aggregator (wantLogprobs off) is a pure passthrough

    func testDisabledAggregatorAddsNoEntriesAndNoOverhead() {
        let resolver = FakeResolver()
        let agg = LogprobsAggregator(engine: resolver, enabled: false)
        let text = agg.consume(TokenEvent(tokenId: 1, text: "hello", elapsed: 0, logprob: info()))
        XCTAssertEqual(text, "hello")
        XCTAssertEqual(agg.entries.count, 0)
        XCTAssertEqual(agg.finish(), "")
    }
}
