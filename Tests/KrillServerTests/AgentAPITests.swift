import XCTest
import NIOCore
import NIOEmbedded
import NIOHTTP1
import KrillEngine
import KrillRegistry
@testable import KrillServer

/// EmbeddedChannel coverage for the `/v1/agent/*` HTTP surface (AgentAPI.swift)
/// and the `/ui` shell (WebUI.swift), mirroring the pattern established in
/// `ServerTests.swift`: `KrillServer._makeHTTPHandlerForTesting` wired into an
/// `EmbeddedChannel`, writing `HTTPServerRequestPart`s in and reading response
/// parts back out. The fallback engine used here is always unloaded, so
/// `activeRef.current` is nil — any session that actually tries to generate
/// fails with "No model available"; these tests only exercise request
/// validation and the session/event bookkeeping, not generation itself.
final class AgentAPITests: XCTestCase {

    // MARK: - Helpers (mirrors ServerTests.swift's private helpers)

    private func makeTempWorkspace() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-agent-api-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func makeChannel(apiKey: String? = nil) throws -> EmbeddedChannel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-agent-api-tests-\(UUID().uuidString)")
        let engine = InferenceEngine(modelDirectory: root.appendingPathComponent("model"))
        let registry = Registry(baseDir: root.appendingPathComponent("registry"))
        let channel = EmbeddedChannel()
        try channel.pipeline.addHandler(
            KrillServer._makeHTTPHandlerForTesting(engine: engine, registry: registry, apiKey: apiKey)
        ).wait()
        return channel
    }

    private func writeRequest(
        to channel: EmbeddedChannel, method: HTTPMethod, uri: String,
        headers: [(String, String)] = []
    ) throws {
        var head = HTTPRequestHead(version: .http1_1, method: method, uri: uri)
        for (k, v) in headers { head.headers.add(name: k, value: v) }
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.head(head)))
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.end(nil)))
    }

    private func writeJSONRequest(
        to channel: EmbeddedChannel, method: HTTPMethod, uri: String, body: [String: Any]
    ) throws {
        let data = try JSONSerialization.data(withJSONObject: body)
        var head = HTTPRequestHead(version: .http1_1, method: method, uri: uri)
        head.headers.add(name: "Content-Type", value: "application/json")
        head.headers.add(name: "Content-Length", value: "\(data.count)")
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.head(head)))
        var buffer = channel.allocator.buffer(capacity: data.count)
        buffer.writeBytes(data)
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.body(buffer)))
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.end(nil)))
    }

    private func readResponseHead(from channel: EmbeddedChannel) throws -> HTTPResponseHead {
        let part = try XCTUnwrap(channel.readOutbound(as: HTTPServerResponsePart.self))
        guard case .head(let head) = part else {
            XCTFail("Expected response head, got \(part)")
            throw TestError.unexpectedResponsePart
        }
        return head
    }

    private func readResponseBody(from channel: EmbeddedChannel) throws -> String {
        let part = try XCTUnwrap(channel.readOutbound(as: HTTPServerResponsePart.self))
        guard case .body(.byteBuffer(var buffer)) = part,
              let body = buffer.readString(length: buffer.readableBytes) else {
            XCTFail("Expected response body, got \(part)")
            throw TestError.unexpectedResponsePart
        }
        return body
    }

    private func readJSONResponseBody(from channel: EmbeddedChannel) throws -> [String: Any] {
        let body = try readResponseBody(from: channel)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any])
    }

    private func readResponseEnd(from channel: EmbeddedChannel) throws {
        let part = try XCTUnwrap(channel.readOutbound(as: HTTPServerResponsePart.self))
        guard case .end = part else {
            XCTFail("Expected response end, got \(part)")
            throw TestError.unexpectedResponsePart
        }
    }

    /// Create a session on `channel` (via the real HTTP route) and return its id.
    @discardableResult
    private func createSession(on channel: EmbeddedChannel, workspace: URL? = nil) throws -> String {
        let dir = try workspace ?? makeTempWorkspace()
        try writeJSONRequest(
            to: channel, method: .POST, uri: "/v1/agent/sessions", body: ["workspace": dir.path])
        XCTAssertEqual(try readResponseHead(from: channel).status, .ok)
        let json = try readJSONResponseBody(from: channel)
        try readResponseEnd(from: channel)
        return try XCTUnwrap(json["id"] as? String)
    }

    private enum TestError: Error {
        case unexpectedResponsePart
    }

    // MARK: - 1. Create session

    func testCreateSessionWithValidWorkspaceReturns200() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        let tempDir = try makeTempWorkspace()

        try writeJSONRequest(
            to: channel, method: .POST, uri: "/v1/agent/sessions", body: ["workspace": tempDir.path])
        XCTAssertEqual(try readResponseHead(from: channel).status, .ok)
        let json = try readJSONResponseBody(from: channel)
        XCTAssertNotNil(json["id"] as? String)
        XCTAssertEqual(json["status"] as? String, "idle")
        XCTAssertEqual(json["workspace"] as? String, tempDir.standardizedFileURL.path)
        try readResponseEnd(from: channel)
    }

    func testCreateSessionWithBogusWorkspaceReturns400() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(
            to: channel, method: .POST, uri: "/v1/agent/sessions",
            body: ["workspace": "/no/such/directory-\(UUID().uuidString)"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .badRequest)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testCreateSessionWithUninstalledModelReturns400() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        let tempDir = try makeTempWorkspace()

        try writeJSONRequest(
            to: channel, method: .POST, uri: "/v1/agent/sessions",
            body: ["workspace": tempDir.path, "model": "not-an-installed-model"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .badRequest)
        let json = try readJSONResponseBody(from: channel)
        XCTAssertTrue((json["error"] as? String)?.contains("not installed") ?? false, "got: \(json)")
        try readResponseEnd(from: channel)
    }

    // MARK: - 2. List sessions

    func testListSessionsIncludesTheCreatedSession() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        let id = try createSession(on: channel)

        try writeRequest(to: channel, method: .GET, uri: "/v1/agent/sessions")
        XCTAssertEqual(try readResponseHead(from: channel).status, .ok)
        let body = try readJSONResponseBody(from: channel)
        let sessions = try XCTUnwrap(body["sessions"] as? [[String: Any]])
        XCTAssertTrue(sessions.contains { ($0["id"] as? String) == id })
        try readResponseEnd(from: channel)
    }

    // MARK: - 3. Get unknown session

    func testGetUnknownSessionReturns404() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeRequest(to: channel, method: .GET, uri: "/v1/agent/sessions/does-not-exist")
        XCTAssertEqual(try readResponseHead(from: channel).status, .notFound)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    // MARK: - 4. Post message with empty text

    func testPostMessageWithEmptyTextReturns400() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        let id = try createSession(on: channel)

        try writeJSONRequest(
            to: channel, method: .POST, uri: "/v1/agent/sessions/\(id)/messages", body: ["text": "   "])
        XCTAssertEqual(try readResponseHead(from: channel).status, .badRequest)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    // MARK: - 5. Post approval with nothing pending

    func testPostApprovalWithNothingPendingReturns409() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        let id = try createSession(on: channel)

        try writeJSONRequest(
            to: channel, method: .POST, uri: "/v1/agent/sessions/\(id)/approvals", body: ["allow": true])
        XCTAssertEqual(try readResponseHead(from: channel).status, .conflict)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    // MARK: - 6. GET events: SSE head

    func testEventsEndpointReturnsSSEHead() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        let id = try createSession(on: channel)

        try writeRequest(to: channel, method: .GET, uri: "/v1/agent/sessions/\(id)/events")
        let head = try readResponseHead(from: channel)
        XCTAssertEqual(head.status, .ok)
        XCTAssertEqual(head.headers.first(name: "Content-Type"), "text/event-stream")
        // No prior events were emitted (nothing but `create` happened), so
        // replay produces no body frames; event-log replay semantics
        // (`id: <seq>` lines) are covered directly in AgentSessionTests.
        //
        // Note: this route parks a 25s heartbeat `Task` on the connection
        // (`agentEventsSub`) that `finish()`'s `channelInactive` cancels.
        // Because that cancellation is observed off-thread from the
        // EmbeddedChannel's loop (a real Task vs. NIO's single-threaded
        // embedded loop), NIO may log a benign
        // "EmbeddedEventLoop is not thread-safe" diagnostic during teardown.
        // It has not been observed to fail a run; see the handoff notes for
        // this file if it ever needs a firmer fix upstream (not something to
        // paper over from the test side, since it's a real production race
        // in the SSE heartbeat's interaction with EmbeddedChannel, not a
        // defect in this test).
    }

    // MARK: - 7. Auth

    func testUIShellIsExemptFromBearerAuth() throws {
        let channel = try makeChannel(apiKey: "sekrit")
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeRequest(to: channel, method: .GET, uri: "/ui")
        XCTAssertEqual(try readResponseHead(from: channel).status, .ok)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testAgentSessionsRequiresBearerAuthWhenAPIKeySet() throws {
        let channel = try makeChannel(apiKey: "sekrit")
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeRequest(to: channel, method: .GET, uri: "/v1/agent/sessions")
        XCTAssertEqual(try readResponseHead(from: channel).status, .unauthorized)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testAgentSessionsSucceedsWithCorrectBearerToken() throws {
        let channel = try makeChannel(apiKey: "sekrit")
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeRequest(
            to: channel, method: .GET, uri: "/v1/agent/sessions",
            headers: [("Authorization", "Bearer sekrit")])
        XCTAssertEqual(try readResponseHead(from: channel).status, .ok)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    // MARK: - 8. /ui static routes

    func testUIRouteServesHTMLShell() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeRequest(to: channel, method: .GET, uri: "/ui")
        let head = try readResponseHead(from: channel)
        XCTAssertEqual(head.headers.first(name: "Content-Type"), "text/html; charset=utf-8")
        let body = try readResponseBody(from: channel)
        XCTAssertTrue(body.lowercased().contains("<!doctype html>"), "got: \(body)")
        try readResponseEnd(from: channel)
    }

    func testUIManifestRouteServesWebManifest() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeRequest(to: channel, method: .GET, uri: "/ui/manifest.webmanifest")
        let head = try readResponseHead(from: channel)
        XCTAssertEqual(head.headers.first(name: "Content-Type"), "application/manifest+json")
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testUnknownUIRouteReturns404() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeRequest(to: channel, method: .GET, uri: "/ui/x")
        XCTAssertEqual(try readResponseHead(from: channel).status, .notFound)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    // MARK: - 9. queryParams

    func testQueryParamsParsesSinceAndPath() {
        let params = HTTPHandler.queryParams("/v1/agent/sessions/x/events?since=42&path=%7E/code")
        XCTAssertEqual(params["since"], "42")
        XCTAssertEqual(params["path"], "~/code")
    }

    func testQueryParamsDecodesPlusAsSpace() {
        let params = HTTPHandler.queryParams("/v1/agent/workspaces?path=my+folder+name")
        XCTAssertEqual(params["path"], "my folder name")
    }
}
