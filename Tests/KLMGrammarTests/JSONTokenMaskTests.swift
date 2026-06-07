import XCTest
import MLX
@testable import KLMGrammar

/// Tests for the tokenizer-vocab logit mask. Uses a tiny synthetic
/// vocabulary so the mask's allow/forbid decisions are exactly predictable.
final class JSONTokenMaskTests: XCTestCase {

    // A small toy vocab. ids are the array indices.
    //  0:"{" 1:"}" 2:"[" 3:"]" 4:"\"" 5:"a" 6:":" 7:"," 8:"1"
    //  9:"true" 10:" " 11:"" (empty/special) 12:EOS-placeholder ""
    private let pieces = ["{", "}", "[", "]", "\"", "a", ":", ",", "1",
                          "true", " ", "", ""]
    private let eosId = 12

    private func makeMask() -> JSONTokenMask {
        JSONTokenMask(pieces: pieces, stopIds: [eosId])
    }

    /// Token ids whose mask bias == 0 (allowed) for a given state.
    private func allowed(_ mask: JSONTokenMask, _ state: JSONGrammar.State) -> Set<Int> {
        let arr = mask.mask(for: state)
        let host = arr.asArray(Float.self)
        var out = Set<Int>()
        for (i, v) in host.enumerated() where v == 0 { out.insert(i) }
        return out
    }

    func testInitialStateAllowsValueStarters() {
        let mask = makeMask()
        let a = allowed(mask, JSONGrammar.initialState)
        // Value starters: { [ " 1 true  (and leading whitespace " ").
        XCTAssertTrue(a.contains(0))   // {
        XCTAssertTrue(a.contains(2))   // [
        XCTAssertTrue(a.contains(4))   // "
        XCTAssertTrue(a.contains(8))   // 1
        XCTAssertTrue(a.contains(9))   // true
        XCTAssertTrue(a.contains(10))  // leading whitespace
        // Must forbid structural closers and a bare colon/comma at the start.
        XCTAssertFalse(a.contains(1))  // }
        XCTAssertFalse(a.contains(3))  // ]
        XCTAssertFalse(a.contains(6))  // :
        XCTAssertFalse(a.contains(7))  // ,
        // Empty pieces are always forbidden.
        XCTAssertFalse(a.contains(11))
    }

    func testEOSForbiddenUntilComplete() {
        let mask = makeMask()
        // At the very start nothing is emitted → not complete → EOS forbidden.
        XCTAssertFalse(allowed(mask, JSONGrammar.initialState).contains(eosId))

        // After a complete value, EOS is allowed.
        guard let complete = JSONGrammar.advance(JSONGrammar.initialState, piece: "true") else {
            return XCTFail("'true' should parse")
        }
        XCTAssertTrue(JSONGrammar.isComplete(complete))
        XCTAssertTrue(allowed(mask, complete).contains(eosId))
    }

    func testInsideObjectExpectsKeyOrClose() {
        let mask = makeMask()
        guard let afterBrace = JSONGrammar.advance(JSONGrammar.initialState, piece: "{") else {
            return XCTFail("'{' should parse")
        }
        let a = allowed(mask, afterBrace)
        XCTAssertTrue(a.contains(4))   // " (start a key)
        XCTAssertTrue(a.contains(1))   // } (empty object)
        XCTAssertTrue(a.contains(10))  // whitespace
        XCTAssertFalse(a.contains(8))  // 1 — a number is not a valid key
        XCTAssertFalse(a.contains(eosId))  // object not complete
    }

    func testAdvanceRejectsInvalidToken() {
        let mask = makeMask()
        // From the initial state, the "}" token (id 1) is not a valid value
        // start, so advance must return nil.
        XCTAssertNil(mask.advance(JSONGrammar.initialState, token: 1))
        // "{" (id 0) is valid.
        XCTAssertNotNil(mask.advance(JSONGrammar.initialState, token: 0))
    }

    // MARK: - Multi-character (BPE) pieces straddling structural boundaries

    /// A glued-token vocabulary: real BPE vocabs emit pieces that span
    /// structural boundaries (`{"`, `":"`, `","`, `"}`), not one char each.
    /// The whole point of a TOKEN-level mask (vs char-level) is to handle
    /// these, so build a complete object using ONLY straddling tokens. The
    /// value-side glue is `":"` (close key, colon, open value string) so the
    /// value is a proper JSON string, and `","` (close value, comma, open
    /// next key) chains object members.
    //  0:"{\"" 1:"\":\"" 2:"\",\"" 3:"\"}" 4:"k" 5:"v" 6:"}" 7:"" 8:EOS
    private let gluedPieces = ["{\"", "\":\"", "\",\"", "\"}", "k", "v",
                              "}", "", ""]
    private func gluedMask() -> JSONTokenMask {
        JSONTokenMask(pieces: gluedPieces, stopIds: [8])
    }

    /// Feed a token-id sequence through the mask's `advance`, asserting each
    /// step is accepted (non-nil). Returns the final grammar state.
    private func runTokens(_ mask: JSONTokenMask, _ ids: [Int],
                           file: StaticString = #filePath, line: UInt = #line) -> JSONGrammar.State? {
        var st = JSONGrammar.initialState
        for id in ids {
            guard let next = mask.advance(st, token: id) else {
                XCTFail("token id \(id) (piece \(gluedPieces[id])) rejected", file: file, line: line)
                return nil
            }
            st = next
        }
        return st
    }

    func testObjectBuiltFromStraddlingTokens() {
        let mask = gluedMask()
        // {"k":"v"}  via  {"  k  ":  v  "}
        guard let st = runTokens(mask, [0, 4, 1, 5, 3]) else { return }
        XCTAssertTrue(JSONGrammar.isComplete(st),
                      "object from straddling tokens should be complete")
    }

    func testMultiKeyObjectFromStraddlingTokens() {
        let mask = gluedMask()
        // {"k":"v","k":"v"}  via  {"  k  ":  v  ","  k  ":  v  "}
        guard let st = runTokens(mask, [0, 4, 1, 5, 2, 4, 1, 5, 3]) else { return }
        XCTAssertTrue(JSONGrammar.isComplete(st))
    }

    func testStraddlingTokenAllowedOnlyInRightState() {
        let mask = gluedMask()
        // A bare key character `k` (id 4) is not a valid value start, so it is
        // forbidden at the top level, but allowed once a key string is open.
        XCTAssertNil(mask.advance(JSONGrammar.initialState, token: 4))
        guard let afterOpen = mask.advance(JSONGrammar.initialState, token: 0) else {
            return XCTFail("'{\"' should open")
        }
        XCTAssertNotNil(mask.advance(afterOpen, token: 4))  // k inside key
        XCTAssertTrue(allowed(mask, afterOpen).contains(4))
        XCTAssertFalse(allowed(mask, JSONGrammar.initialState).contains(4))
    }

    func testStraddlingCloserForbiddenMidValue() {
        let mask = gluedMask()
        // After `{"` we are expecting a key string body; the `"}` closer
        // (id 3) is `"` (closes the key -> expecting `:`) then `}` — but `}`
        // where a `:` is required is invalid, so `{""}` must be rejected.
        guard let afterOpen = mask.advance(JSONGrammar.initialState, token: 0) else {
            return XCTFail("'{\"' should open")
        }
        XCTAssertNil(mask.advance(afterOpen, token: 3))
    }

    func testMaskBiasValues() {
        let mask = makeMask()
        let arr = mask.mask(for: JSONGrammar.initialState)
        let host = arr.asArray(Float.self)
        XCTAssertEqual(host.count, pieces.count)
        // Allowed → 0; forbidden → large negative.
        XCTAssertEqual(host[0], 0)          // "{"
        XCTAssertLessThan(host[1], -1e8)    // "}"
    }

    // MARK: - Padded logits width (Gemma 4: 262144 logits vs 261707 pieces)

    func testPaddedOutputWidthBlocksUnusedTail() {
        let pad = 5
        let width = pieces.count + pad
        let mask = JSONTokenMask(pieces: pieces, stopIds: [eosId], outputWidth: width)
        // The grammar still reasons over the real tokenizer pieces…
        XCTAssertEqual(mask.vocabSize, pieces.count)
        // …but emits a mask at the model's padded logits width.
        XCTAssertEqual(mask.maskWidth, width)

        let host = mask.mask(for: JSONGrammar.initialState).asArray(Float.self)
        XCTAssertEqual(host.count, width)
        // Real-token decisions are unchanged by padding.
        XCTAssertEqual(host[0], 0)          // "{" still allowed
        XCTAssertLessThan(host[1], -1e8)    // "}" still forbidden
        // Every padding slot is blocked so it can never be sampled.
        for id in pieces.count ..< width {
            XCTAssertLessThan(host[id], -1e8, "padding id \(id) must be blocked")
        }
        // advance() stays bounded by the real vocab.
        XCTAssertNil(mask.advance(JSONGrammar.initialState, token: pieces.count))
    }

    func testPaddedFailOpenStillBlocksTail() {
        // outputWidth below pieces.count is clamped up (mask must cover all
        // real tokens); equal width behaves like the default.
        let mask = JSONTokenMask(pieces: pieces, stopIds: [eosId], outputWidth: 2)
        XCTAssertEqual(mask.maskWidth, pieces.count)
    }
}
