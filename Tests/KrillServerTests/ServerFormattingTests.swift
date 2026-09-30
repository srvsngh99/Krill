import Foundation
import XCTest
import KrillSampler
@testable import KrillServer

/// Minimal synthetic `TokenLogprobResolver` for `echoPromptLogprobEntry`
/// tests below - no real tokenizer needed, every alternate token id maps to
/// its own fixed byte sequence.
private final class FakeEchoResolver: TokenLogprobResolver {
    var bytesByToken: [Int: [UInt8]] = [:]
    func rawTokenBytes(for tokenId: Int) -> [UInt8]? { bytesByToken[tokenId] }
    func lossyTokenString(bytes: [UInt8]) -> String { String(decoding: bytes, as: UTF8.self) }
    func isOutputSuppressedToken(_ tokenId: Int) -> Bool { false }
}

final class ServerFormattingTests: XCTestCase {
    func testSSEContentChunkShape() throws {
        let payload = try parseSSE(sseChunk(
            id: "chatcmpl-test", content: "hello", finishReason: nil))

        XCTAssertEqual(payload["id"] as? String, "chatcmpl-test")
        XCTAssertEqual(payload["object"] as? String, "chat.completion.chunk")
        XCTAssertNotNil(payload["created"] as? Int)
        let choices = try XCTUnwrap(payload["choices"] as? [[String: Any]])
        let choice = try XCTUnwrap(choices.first)
        XCTAssertEqual(choice["index"] as? Int, 0)
        XCTAssertNil(choice["finish_reason"])
        let delta = try XCTUnwrap(choice["delta"] as? [String: Any])
        XCTAssertEqual(delta["role"] as? String, "assistant")
        XCTAssertEqual(delta["content"] as? String, "hello")
    }

    func testSSEFinishChunkHasEmptyDelta() throws {
        let payload = try parseSSE(sseChunk(
            id: "chatcmpl-test", content: nil, finishReason: "stop"))
        let choices = try XCTUnwrap(payload["choices"] as? [[String: Any]])
        let choice = try XCTUnwrap(choices.first)
        XCTAssertEqual(choice["finish_reason"] as? String, "stop")
        XCTAssertEqual((choice["delta"] as? [String: Any])?.count, 0)
    }

    func testSSEUsageChunkShape() throws {
        let payload = try parseSSE(sseUsageChunk(
            id: "chatcmpl-test", promptTokens: 7, completionTokens: 5))
        XCTAssertEqual((payload["choices"] as? [Any])?.count, 0)
        let usage = try XCTUnwrap(payload["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 7)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 5)
        XCTAssertEqual(usage["total_tokens"] as? Int, 12)
    }

    // MARK: - logprobs (docs/LOGPROBS_PLAN.md, Phase 1)

    func testSSEChunkWithoutLogprobsIsByteIdenticalToBeforeTheField() throws {
        // The `logprobs` parameter defaults to nil and must be omittable
        // without changing the emitted chunk's *shape* at all - the
        // "zero cost / zero risk when off" requirement for a hot-path
        // formatter (docs/LOGPROBS_PLAN.md §5.2, §3.3). Compared structurally
        // (not as raw strings): `JSONSerialization` does not guarantee key
        // order between calls, so two independently-built dictionaries with
        // identical content can legitimately serialize to different byte
        // strings - that is a `JSONSerialization` property, not a regression.
        let withDefault = try parseSSE(sseChunk(id: "chatcmpl-x", content: "hi", finishReason: nil))
        let explicitNil: Any? = nil
        let withExplicitNil = try parseSSE(
            sseChunk(id: "chatcmpl-x", content: "hi", finishReason: nil, logprobs: explicitNil))
        let choiceDefault = try XCTUnwrap((withDefault["choices"] as? [[String: Any]])?.first)
        let choiceExplicitNil = try XCTUnwrap((withExplicitNil["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(Set(choiceDefault.keys), Set(choiceExplicitNil.keys))
        XCTAssertFalse(choiceDefault.keys.contains("logprobs"), "omitting logprobs must not add the key")
        XCTAssertEqual(NSDictionary(dictionary: choiceDefault), NSDictionary(dictionary: choiceExplicitNil))

        let finishDefault = try parseSSE(sseChunk(id: "chatcmpl-x", content: nil, finishReason: "stop"))
        let finishExplicitNil = try parseSSE(
            sseChunk(id: "chatcmpl-x", content: nil, finishReason: "stop", logprobs: explicitNil))
        let finishChoiceDefault = try XCTUnwrap((finishDefault["choices"] as? [[String: Any]])?.first)
        let finishChoiceExplicitNil = try XCTUnwrap((finishExplicitNil["choices"] as? [[String: Any]])?.first)
        XCTAssertFalse(finishChoiceDefault.keys.contains("logprobs"))
        XCTAssertEqual(NSDictionary(dictionary: finishChoiceDefault), NSDictionary(dictionary: finishChoiceExplicitNil))
    }

    func testSSEContentChunkWithLogprobsShape() throws {
        let entry = logprobEntryJSON(token: "Hello", logprob: -0.25, bytes: [72, 101], topLogprobs: [])
        let payload = try parseSSE(sseChunk(
            id: "chatcmpl-test", content: "Hello", finishReason: nil,
            logprobs: logprobsChoiceJSON(content: [entry])))
        let choices = try XCTUnwrap(payload["choices"] as? [[String: Any]])
        let choice = try XCTUnwrap(choices.first)
        let logprobs = try XCTUnwrap(choice["logprobs"] as? [String: Any])
        let content = try XCTUnwrap(logprobs["content"] as? [[String: Any]])
        XCTAssertEqual(content.count, 1)
        XCTAssertEqual(content[0]["token"] as? String, "Hello")
        XCTAssertEqual(content[0]["logprob"] as? Float, -0.25)
        XCTAssertEqual(content[0]["bytes"] as? [Int], [72, 101])
        XCTAssertEqual((content[0]["top_logprobs"] as? [Any])?.count, 0)
    }

    func testSSEFinishChunkWithLogprobsIsExplicitNull() throws {
        // A chunk carrying no token (the finish chunk) gets `logprobs: null`,
        // not an omitted key, whenever the REQUEST asked for logprobs.
        let raw = sseChunk(id: "chatcmpl-test", content: nil, finishReason: "stop", logprobs: NSNull())
        XCTAssertTrue(raw.contains("\"logprobs\":null"))
        let payload = try parseSSE(raw)
        let choices = try XCTUnwrap(payload["choices"] as? [[String: Any]])
        let choice = try XCTUnwrap(choices.first)
        XCTAssertTrue(choice["logprobs"] is NSNull)
    }

    func testLogprobEntryJSONGoldenShape() {
        let alt = logprobEntryJSON(token: "Hi", logprob: -1.8, bytes: [72, 105])
        let entry = logprobEntryJSON(
            token: "Hello", logprob: -0.31, bytes: [72, 101, 108, 108, 111], topLogprobs: [alt])
        XCTAssertEqual(entry["token"] as? String, "Hello")
        XCTAssertEqual(entry["logprob"] as? Float, -0.31)
        XCTAssertEqual(entry["bytes"] as? [Int], [72, 101, 108, 108, 111])
        let top = entry["top_logprobs"] as? [[String: Any]]
        XCTAssertEqual(top?.count, 1)
        XCTAssertEqual(top?.first?["token"] as? String, "Hi")
        XCTAssertEqual(top?.first?["bytes"] as? [Int], [72, 105])
    }

    func testLogprobEntryJSONOmitsTopLogprobsKeyWhenNil() {
        // Only the SAMPLED-token entry always carries `top_logprobs` (added
        // explicitly by `logprobsContentEntry`); the bare builder leaves it
        // out when the caller passes nil, e.g. when building an alternate's
        // own entry (alternates never carry a nested top_logprobs list).
        let entry = logprobEntryJSON(token: "x", logprob: -1.0, bytes: [120])
        XCTAssertNil(entry["top_logprobs"])
    }

    func testLogprobsChoiceJSONAlwaysEmitsContentArray() {
        // Decision: `top_logprobs`/`content` are never omitted - the OpenAI
        // SDK types treat these as required lists, so an empty request still
        // gets `"content": []`, never a missing key.
        let empty = logprobsChoiceJSON(content: [])
        let content = empty["content"] as? [[String: Any]]
        XCTAssertEqual(content?.count, 0)
    }

    // MARK: - Tool-call logprobs (2026-09-30)

    func testToolChatLogprobsJSONIsNullWhenNotRequested() {
        let result = toolChatLogprobsJSON(wantLogprobs: false, hasToolCalls: true, content: [])
        XCTAssertTrue(result is NSNull)
    }

    func testToolChatLogprobsJSONForPureToolCallTurnIsObjectWithNullContent() {
        // The real OpenAI shape for a tool_calls turn: a present `logprobs`
        // object with `content`/`refusal` null - NOT a bare `logprobs: null`
        // - per `openai/types/chat/chat_completion.py`'s `ChoiceLogprobs`
        // and a live forum report of `ChoiceLogprobs(content=None)` on an
        // actual function-call response. See docs/LOGPROBS_PLAN.md.
        let result = toolChatLogprobsJSON(wantLogprobs: true, hasToolCalls: true, content: [])
        let obj = result as? [String: Any]
        XCTAssertNotNil(obj, "must be a real object, not a bare null")
        XCTAssertTrue(obj?["content"] is NSNull)
        XCTAssertTrue(obj?["refusal"] is NSNull)
    }

    func testToolChatLogprobsJSONForPlainReplyStillGetsRealEntries() throws {
        let entry = logprobEntryJSON(token: "x", logprob: -0.1, bytes: [120])
        let result = toolChatLogprobsJSON(wantLogprobs: true, hasToolCalls: false, content: [entry])
        let obj = try XCTUnwrap(result as? [String: Any])
        let content = obj["content"] as? [[String: Any]]
        XCTAssertEqual(content?.count, 1)
    }

    func testEscapeJSONRoundTripsQuotesSlashesAndControls() throws {
        let original = "quote=\" slash=\\ newline=\n return=\r tab=\t control=\u{0001} unicode=🐙"
        let document = "{\"value\":\"\(escapeJSON(original))\"}"
        let parsed = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(document.utf8)) as? [String: String])
        XCTAssertEqual(parsed["value"], original)
    }

    // MARK: - Ollama + legacy completions logprobs (2026-09-30 follow-up)

    func testOllamaLogprobEntryJSONDropsEmptyTopLogprobsKey() {
        // Go's `TopLogprobs []TokenLogprob json:"top_logprobs,omitempty"` -
        // an EMPTY alternates list is omitted entirely, unlike OpenAI chat's
        // always-present `[]` (a required list per that SDK's own model).
        let entry = logprobEntryJSON(token: "x", logprob: -1.0, bytes: [120], topLogprobs: [])
        let converted = ollamaLogprobEntryJSON(entry)
        XCTAssertNil(converted["top_logprobs"])
        XCTAssertEqual(converted["token"] as? String, "x")
        XCTAssertEqual(converted["logprob"] as? Float, -1.0)
        XCTAssertEqual(converted["bytes"] as? [Int], [120])
    }

    func testOllamaLogprobEntryJSONKeepsNonEmptyTopLogprobs() {
        let alt = logprobEntryJSON(token: "Hi", logprob: -1.8, bytes: [72, 105])
        let entry = logprobEntryJSON(token: "Hello", logprob: -0.31, bytes: [72, 101], topLogprobs: [alt])
        let converted = ollamaLogprobEntryJSON(entry)
        let top = converted["top_logprobs"] as? [[String: Any]]
        XCTAssertEqual(top?.count, 1)
    }

    func testOllamaLogprobsArrayJSONShape() {
        let e1 = logprobEntryJSON(token: "a", logprob: -0.1, bytes: [97])
        let e2 = logprobEntryJSON(token: "b", logprob: -0.2, bytes: [98])
        let arr = ollamaLogprobsArrayJSON(entries: [e1, e2])
        XCTAssertEqual(arr.count, 2)
        XCTAssertEqual(arr[0]["token"] as? String, "a")
        XCTAssertEqual(arr[1]["token"] as? String, "b")
        XCTAssertNil(arr[0]["top_logprobs"])
    }

    func testLegacyCompletionLogprobsJSONGoldenShapeWithAlternates() throws {
        // logprobs: 2 requested - each position's dict gets the top-2
        // alternates, plus the sampled token folded in when it is not
        // already one of them ("up to logprobs+1 elements").
        let alt1 = logprobEntryJSON(token: " world", logprob: -0.05, bytes: Array(" world".utf8))
        let alt2 = logprobEntryJSON(token: " there", logprob: -3.2, bytes: Array(" there".utf8))
        let entry = logprobEntryJSON(
            token: " earth", logprob: -4.1, bytes: Array(" earth".utf8), topLogprobs: [alt1, alt2])
        let json = legacyCompletionLogprobsJSON(entries: [entry])

        let tokens = try XCTUnwrap(json["tokens"] as? [String])
        XCTAssertEqual(tokens, [" earth"])
        let tokenLogprobs = try XCTUnwrap(json["token_logprobs"] as? [Float])
        XCTAssertEqual(tokenLogprobs, [-4.1])
        let textOffset = try XCTUnwrap(json["text_offset"] as? [Int])
        XCTAssertEqual(textOffset, [0])
        let topLogprobs = try XCTUnwrap(json["top_logprobs"] as? [[String: Any]])
        XCTAssertEqual(topLogprobs.count, 1)
        let dict = topLogprobs[0]
        XCTAssertEqual(dict.count, 3, "2 alternates + the sampled token, not already among them")
        XCTAssertEqual(dict[" world"] as? Float, -0.05)
        XCTAssertEqual(dict[" there"] as? Float, -3.2)
        XCTAssertEqual(dict[" earth"] as? Float, -4.1)
    }

    func testLegacyCompletionLogprobsJSONSampledTokenNotDuplicatedWhenAlreadyTopAlternate() {
        // When the sampled token IS already one of the reported alternates,
        // the dict must not gain a second (redundant) entry under the same
        // key - "up to logprobs+1", not always +1.
        let sampledAsAlt = logprobEntryJSON(token: " world", logprob: -0.05, bytes: Array(" world".utf8))
        let entry = logprobEntryJSON(
            token: " world", logprob: -0.05, bytes: Array(" world".utf8), topLogprobs: [sampledAsAlt])
        let json = legacyCompletionLogprobsJSON(entries: [entry])
        let topLogprobs = json["top_logprobs"] as? [[String: Any]]
        XCTAssertEqual(topLogprobs?.first?.count, 1)
    }

    func testLegacyCompletionLogprobsJSONZeroRequestedGivesEmptyDicts() {
        // logprobs: 0 - sampled-token logprob only via `token_logprobs`; the
        // per-position dict stays `{}`, the sampled token is NOT duplicated
        // into it (docs/LOGPROBS_PLAN.md §3.2, the 2026-09-30 addendum).
        let entry = logprobEntryJSON(token: "OK", logprob: -0.02, bytes: Array("OK".utf8), topLogprobs: [])
        let json = legacyCompletionLogprobsJSON(entries: [entry])
        let tokenLogprobs = json["token_logprobs"] as? [Float]
        XCTAssertEqual(tokenLogprobs, [-0.02])
        let topLogprobs = json["top_logprobs"] as? [[String: Any]]
        XCTAssertEqual(topLogprobs?.first?.count, 0)
    }

    func testLegacyCompletionLogprobsJSONTextOffsetsAccumulate() throws {
        let e1 = logprobEntryJSON(token: "ab", logprob: -0.1, bytes: Array("ab".utf8))
        let e2 = logprobEntryJSON(token: "cde", logprob: -0.2, bytes: Array("cde".utf8))
        let e3 = logprobEntryJSON(token: "f", logprob: -0.3, bytes: Array("f".utf8))
        let json = legacyCompletionLogprobsJSON(entries: [e1, e2, e3])
        let textOffset = try XCTUnwrap(json["text_offset"] as? [Int])
        XCTAssertEqual(textOffset, [0, 2, 5])
    }

    func testLegacyCompletionLogprobsJSONEmptyEntriesGivesEmptyArrays() throws {
        let json = legacyCompletionLogprobsJSON(entries: [])
        XCTAssertEqual((json["tokens"] as? [String])?.count, 0)
        XCTAssertEqual((json["token_logprobs"] as? [Any])?.count, 0)
        XCTAssertEqual((json["top_logprobs"] as? [Any])?.count, 0)
        XCTAssertEqual((json["text_offset"] as? [Int])?.count, 0)
    }

    // MARK: - `echo` prompt logprobs (Phase 3, docs/LOGPROBS_PLAN.md §3.2/§5.4)

    func testEchoPromptLogprobEntryFirstTokenIsNull() {
        let resolver = FakeEchoResolver()
        let entry = echoPromptLogprobEntry(tokenString: "Hello", info: nil, eng: resolver)
        XCTAssertEqual(entry["token"] as? String, "Hello")
        XCTAssertTrue(entry["logprob"] is NSNull)
        XCTAssertTrue(entry["top_logprobs"] is NSNull)
    }

    func testEchoPromptLogprobEntryNonFirstTokenHasRealValues() {
        let resolver = FakeEchoResolver()
        resolver.bytesByToken[7] = Array(" world".utf8)
        let info = TokenLogprobInfo(
            logprob: -0.4, topAlternates: [TokenAltLogprob(tokenId: 7, logprob: -0.4)])
        let entry = echoPromptLogprobEntry(tokenString: " world", info: info, eng: resolver)
        XCTAssertEqual(entry["token"] as? String, " world")
        XCTAssertEqual(entry["logprob"] as? Float, -0.4)
        let alternates = entry["top_logprobs"] as? [[String: Any]]
        XCTAssertEqual(alternates?.count, 1)
        XCTAssertEqual(alternates?.first?["token"] as? String, " world")
    }

    func testLegacyCompletionLogprobsJSONHandlesNullFirstPromptEntry() throws {
        // The whole point of the null sentinel: prepending an echo prompt's
        // first-token entry must NOT crash or coerce to `{}`/`0` - it stays
        // `null` in both `token_logprobs` and `top_logprobs`, and does not
        // disturb the running `text_offset` for the entries after it.
        let resolver = FakeEchoResolver()
        let firstPromptEntry = echoPromptLogprobEntry(tokenString: "Hel", info: nil, eng: resolver)
        let completionEntry = logprobEntryJSON(token: "lo", logprob: -0.02, bytes: Array("lo".utf8))
        let json = legacyCompletionLogprobsJSON(entries: [firstPromptEntry, completionEntry])

        let tokens = try XCTUnwrap(json["tokens"] as? [String])
        XCTAssertEqual(tokens, ["Hel", "lo"])
        let tokenLogprobs = try XCTUnwrap(json["token_logprobs"] as? [Any])
        XCTAssertTrue(tokenLogprobs[0] is NSNull)
        XCTAssertEqual(tokenLogprobs[1] as? Float, -0.02)
        let topLogprobs = try XCTUnwrap(json["top_logprobs"] as? [Any])
        XCTAssertTrue(topLogprobs[0] is NSNull)
        XCTAssertEqual((topLogprobs[1] as? [String: Any])?.count, 0)
        let textOffset = try XCTUnwrap(json["text_offset"] as? [Int])
        XCTAssertEqual(textOffset, [0, 3], "offset accumulates across prompt+completion uniformly")
    }

    private func parseSSE(_ event: String) throws -> [String: Any] {
        XCTAssertTrue(event.hasPrefix("data: "))
        XCTAssertTrue(event.hasSuffix("\n\n"))
        let json = String(event.dropFirst("data: ".count).dropLast(2))
        return try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }
}
