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

    func testMaskBiasValues() {
        let mask = makeMask()
        let arr = mask.mask(for: JSONGrammar.initialState)
        let host = arr.asArray(Float.self)
        XCTAssertEqual(host.count, pieces.count)
        // Allowed → 0; forbidden → large negative.
        XCTAssertEqual(host[0], 0)          // "{"
        XCTAssertLessThan(host[1], -1e8)    // "}"
    }
}
