import Foundation
import XCTest
@testable import KrillServer

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

    func testEscapeJSONRoundTripsQuotesSlashesAndControls() throws {
        let original = "quote=\" slash=\\ newline=\n return=\r tab=\t control=\u{0001} unicode=🐙"
        let document = "{\"value\":\"\(escapeJSON(original))\"}"
        let parsed = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(document.utf8)) as? [String: String])
        XCTAssertEqual(parsed["value"], original)
    }

    private func parseSSE(_ event: String) throws -> [String: Any] {
        XCTAssertTrue(event.hasPrefix("data: "))
        XCTAssertTrue(event.hasSuffix("\n\n"))
        let json = String(event.dropFirst("data: ".count).dropLast(2))
        return try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }
}
