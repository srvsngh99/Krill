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
