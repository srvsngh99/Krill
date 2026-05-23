import XCTest
@testable import KLMServer

/// Verifies the parsing contract DaemonClient holds against the
/// server's own `/v1/status` and `/v1/chat/completions` SSE shapes.
/// Network-level probing (timeout, connection refused, mid-stream
/// kill) is covered by manual end-to-end recipes in the PR; this
/// test guards only the pure-parser surface so a future server
/// response-shape change breaks loudly here.
final class DaemonClientTests: XCTestCase {

    // MARK: - parseStatus

    func testParseStatus_modelLoadedReturnsModelName() throws {
        // Shape from Server.swift handleStatus when engine.isLoaded is true.
        let json = """
        {
          "status": "ready",
          "model_loaded": true,
          "uptime_seconds": 42,
          "memory_mb": 1024,
          "installed_models": ["qwen2.5-3b"],
          "version": "0.3.1",
          "model": "qwen2.5-3b",
          "family": "qwen",
          "model_loaded_at": "2026-05-24T00:00:00Z",
          "model_uptime_seconds": 12
        }
        """.data(using: .utf8)!

        let status = try XCTUnwrap(DaemonClient.parseStatus(data: json))
        XCTAssertTrue(status.modelLoaded)
        XCTAssertEqual(status.model, "qwen2.5-3b")
    }

    func testParseStatus_idleDaemonReturnsModelLoadedFalse() throws {
        // Shape from Server.swift handleStatus when engine.isLoaded is false:
        // the "model" field is intentionally absent.
        let json = """
        {
          "status": "idle",
          "model_loaded": false,
          "uptime_seconds": 7,
          "memory_mb": 512,
          "installed_models": [],
          "version": "0.3.1"
        }
        """.data(using: .utf8)!

        let status = try XCTUnwrap(DaemonClient.parseStatus(data: json))
        XCTAssertFalse(status.modelLoaded)
        XCTAssertNil(status.model)
    }

    func testParseStatus_malformedJSONReturnsNil() {
        let bogus = "not even close to json".data(using: .utf8)!
        XCTAssertNil(DaemonClient.parseStatus(data: bogus))
    }

    func testParseStatus_missingModelLoadedFieldReturnsNil() {
        // A well-formed JSON object that does not carry the required
        // `model_loaded` key (could happen if the daemon's contract
        // drifts). Parser must not silently assume false.
        let json = """
        { "status": "weird", "version": "0.3.1" }
        """.data(using: .utf8)!
        XCTAssertNil(DaemonClient.parseStatus(data: json))
    }

    // MARK: - parseChunkContent

    func testParseChunkContent_extractsDeltaContent() {
        // Shape from sseChunk in Server.swift line 2639 with content set.
        let payload = """
        {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1,"choices":[{"index":0,"delta":{"role":"assistant","content":"hello"}}]}
        """
        XCTAssertEqual(DaemonClient.parseChunkContent(payload), "hello")
    }

    func testParseChunkContent_finishChunkHasNoContent() {
        // sseChunk emits an empty delta dict when finishReason is set;
        // parser must return nil (no token to forward) without crashing.
        let payload = """
        {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
        """
        XCTAssertNil(DaemonClient.parseChunkContent(payload))
    }

    func testParseChunkContent_malformedPayloadReturnsNil() {
        XCTAssertNil(DaemonClient.parseChunkContent("data: not json"))
    }

    // MARK: - parseChunkError

    func testParseChunkError_busyFrameSurfacesMessage() {
        // Exact shape Server.swift line 1010 writes on queue overflow.
        let payload = """
        {"error":"server busy: max queue exceeded"}
        """
        XCTAssertEqual(
            DaemonClient.parseChunkError(payload),
            "server busy: max queue exceeded"
        )
    }

    func testParseChunkError_contentChunkReturnsNil() {
        // A normal content chunk must not be misclassified as an error.
        let payload = """
        {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"choices":[{"index":0,"delta":{"role":"assistant","content":"hello"}}]}
        """
        XCTAssertNil(DaemonClient.parseChunkError(payload))
    }

    func testParseChunkError_malformedPayloadReturnsNil() {
        XCTAssertNil(DaemonClient.parseChunkError("oops"))
    }
}
