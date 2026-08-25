import XCTest
@testable import KrillHarness
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

private actor StubHarnessHTTPTransport: HarnessHTTPTransport {
    let body: Data
    let status: Int
    private var capturedRequest: URLRequest?

    init(body: String, status: Int = 200) {
        self.body = Data(body.utf8)
        self.status = status
    }

    func send(_ request: URLRequest) async throws -> (Data, URLResponse) {
        capturedRequest = request
        let response = HTTPURLResponse(
            url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"])!
        return (body, response)
    }

    func request() -> URLRequest? { capturedRequest }
}

private struct FailingHarnessHTTPTransport: HarnessHTTPTransport {
    struct Failure: LocalizedError { var errorDescription: String? { "offline" } }
    func send(_ request: URLRequest) async throws -> (Data, URLResponse) { throw Failure() }
}

/// Replays a fixed script of statuses, one per attempt, and counts calls so a
/// test can assert on how many attempts the retry policy actually made.
private actor ScriptedHarnessHTTPTransport: HarnessHTTPTransport {
    private let statuses: [Int]
    private let body: String
    private var calls = 0

    init(statuses: [Int], body: String = #"{"choices":[{"finish_reason":"stop","message":{"content":"ok"}}]}"#) {
        self.statuses = statuses
        self.body = body
    }

    func send(_ request: URLRequest) async throws -> (Data, URLResponse) {
        let status = statuses[min(calls, statuses.count - 1)]
        calls += 1
        let response = HTTPURLResponse(
            url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"])!
        return (Data(body.utf8), response)
    }

    func callCount() -> Int { calls }
}

/// Throws for the first `failures` attempts, then succeeds.
private actor FlakyHarnessHTTPTransport: HarnessHTTPTransport {
    struct Failure: LocalizedError { var errorDescription: String? { "connection reset" } }
    private let failures: Int
    private var calls = 0

    init(failures: Int) { self.failures = failures }

    func send(_ request: URLRequest) async throws -> (Data, URLResponse) {
        calls += 1
        if calls <= failures { throw Failure() }
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"])!
        return (Data(#"{"choices":[{"finish_reason":"stop","message":{"content":"recovered"}}]}"#.utf8), response)
    }

    func callCount() -> Int { calls }
}

private actor StubCodexRunner: CodexProcessRunning {
    let result: CodexProcessResult
    private var capturedExecutable: String?
    private var capturedArguments: [String] = []
    private var capturedInput = ""

    init(result: CodexProcessResult) { self.result = result }

    func run(executable: String, arguments: [String], standardInput: String) async throws -> CodexProcessResult {
        capturedExecutable = executable
        capturedArguments = arguments
        capturedInput = standardInput
        return result
    }

    func executable() -> String? { capturedExecutable }
    func arguments() -> [String] { capturedArguments }
    func input() -> String { capturedInput }
}

private struct ThrowingCodexRunner: CodexProcessRunning {
    struct Failure: LocalizedError { var errorDescription: String? { "codex not installed" } }
    func run(executable: String, arguments: [String], standardInput: String) async throws -> CodexProcessResult {
        throw Failure()
    }
}

final class RemoteGeneratorsTests: XCTestCase {
    func testOpenCodeDiscoversOnlyFreeModelsAndPrefersOxAlpha() async throws {
        let transport = StubHarnessHTTPTransport(body: """
        {"data":[
          {"id":"paid-model"},
          {"id":"nemotron-3-ultra-free"},
          {"id":"x-preview-f-free"},
          {"id":"another-free"}
        ]}
        """)

        let models = try await OpenCodeZen.freeModels(transport: transport)
        XCTAssertEqual(models.map(\.id), ["another-free", "nemotron-3-ultra-free", "x-preview-f-free"])
        XCTAssertEqual(OpenCodeZen.defaultFreeModel(from: models)?.id, "x-preview-f-free")
        let request = await transport.request()
        XCTAssertEqual(request?.url?.absoluteString, "https://opencode.ai/zen/v1/models")
        XCTAssertNil(request?.value(forHTTPHeaderField: "Authorization"),
                     "the free OpenCode route must not send a made-up key")
    }

    func testOpenCodeDefaultFallsBackToFirstFreeModel() {
        let models = [OpenCodeZen.Model(id: "a-free"), OpenCodeZen.Model(id: "b-free")]
        XCTAssertEqual(OpenCodeZen.defaultFreeModel(from: models)?.id, "a-free")
        XCTAssertNil(OpenCodeZen.defaultFreeModel(from: []))
    }

    func testOpenAICompatibleGeneratorBuildsKeylessChatRequestAndParsesContent() async throws {
        let transport = StubHarnessHTTPTransport(
            body: #"{"choices":[{"message":{"content":"<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"README.md\"}}</tool_call>"}}]}"#)
        let generator = OpenAICompatibleHarnessGenerator(
            model: "x-preview-f-free", maxTokens: 777, transport: transport)

        let completion = await generator.complete(messages: [
            ["role": "system", "content": "You are Krill."],
            ["role": "user", "content": "Read the README"],
        ])
        XCTAssertTrue(completion.contains("read_file"))
        let captured = await transport.request()
        let request = try XCTUnwrap(captured)
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.url?.absoluteString, "https://opencode.ai/zen/v1/chat/completions")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "application/json")
        XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"))
        let body = try XCTUnwrap(request.httpBody)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        XCTAssertEqual(object["model"] as? String, "x-preview-f-free")
        XCTAssertEqual(object["max_tokens"] as? Int, 777)
        XCTAssertEqual((object["messages"] as? [[String: String]])?.last?["content"], "Read the README")
    }

    func testOpenAICompatibleGeneratorUsesHostedDefaultAndReportsLengthTruncation() throws {
        let generator = OpenAICompatibleHarnessGenerator(model: "x-preview-f-free")
        XCTAssertEqual(generator.maxTokens, OpenAICompatibleHarnessGenerator.defaultMaxTokens)

        let response = Data(#"{"choices":[{"finish_reason":"length","message":{"content":"partial"}}]}"#.utf8)
        XCTAssertThrowsError(try generator.parseCompletion(response)) { error in
            XCTAssertEqual(error as? HostedProviderError, .responseTruncated)
            XCTAssertTrue(error.localizedDescription.contains("--max-tokens"))
        }
    }

    func testRetriesTransient503ThenSucceeds() async {
        let transport = ScriptedHarnessHTTPTransport(statuses: [503, 200])
        let generator = OpenAICompatibleHarnessGenerator(
            model: "x-preview-f-free", transport: transport)
        let reply = await generator.complete(messages: [])
        XCTAssertEqual(reply, "ok")
        let calls = await transport.callCount()
        XCTAssertEqual(calls, 2, "a transient 503 must be retried, not surfaced")
    }

    func testRetriesTransportErrorThenSucceeds() async {
        let transport = FlakyHarnessHTTPTransport(failures: 1)
        let generator = OpenAICompatibleHarnessGenerator(
            model: "x-preview-f-free", transport: transport)
        let reply = await generator.complete(messages: [])
        XCTAssertEqual(reply, "recovered")
        let calls = await transport.callCount()
        XCTAssertEqual(calls, 2)
    }

    func testGivesUpAfterMaxAttemptsAndSurfacesTheStatus() async {
        let transport = ScriptedHarnessHTTPTransport(statuses: [503])
        let generator = OpenAICompatibleHarnessGenerator(
            model: "x-preview-f-free", transport: transport)
        let reply = await generator.complete(messages: [])
        XCTAssertTrue(reply.contains("503"), reply)
        let calls = await transport.callCount()
        XCTAssertEqual(calls, HostedProviderDefaults.maxAttempts)
    }

    func testDoesNotRetryDeterministicClientErrors() async {
        let transport = ScriptedHarnessHTTPTransport(statuses: [400])
        let generator = OpenAICompatibleHarnessGenerator(
            model: "x-preview-f-free", transport: transport)
        let reply = await generator.complete(messages: [])
        XCTAssertTrue(reply.contains("400"), reply)
        let calls = await transport.callCount()
        XCTAssertEqual(calls, 1, "a 4xx is deterministic; retrying only wastes time")
    }

    func testURLSessionTransportAppliesAGenerousTimeout() {
        XCTAssertEqual(HostedProviderDefaults.requestTimeout, 300)
        XCTAssertGreaterThan(
            HostedProviderDefaults.requestTimeout, 60,
            "URLSession's 60s default is shorter than a measured hosted reasoning turn")
    }

    func testOpenAICompatibleGeneratorSurfacesNetworkAndHTTPFailures() async {
        let offline = OpenAICompatibleHarnessGenerator(model: "x-preview-f-free", transport: FailingHarnessHTTPTransport())
        let offlineReply = await offline.complete(messages: [])
        XCTAssertTrue(offlineReply.contains("offline"), offlineReply)

        let rejected = OpenAICompatibleHarnessGenerator(
            model: "x-preview-f-free",
            transport: StubHarnessHTTPTransport(body: "no access", status: 401))
        let rejectedReply = await rejected.complete(messages: [])
        XCTAssertTrue(rejectedReply.contains("HTTP 401: no access"))
    }

    func testCodexUsesSubscriptionSafeArgumentsAndPassesTranscriptOnStdin() async {
        let runner = StubCodexRunner(result: .init(stdout: "next Krill message\n", stderr: "", exitStatus: 0))
        let generator = CodexCLIHarnessGenerator(model: nil, runner: runner)

        let answer = await generator.complete(messages: [["role": "user", "content": "List files"]])
        XCTAssertEqual(answer, "next Krill message")
        let executable = await runner.executable()
        XCTAssertEqual(executable, "codex")
        let arguments = await runner.arguments()
        XCTAssertEqual(arguments, [
            "exec", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check", "--ignore-rules", "--color", "never", "-",
        ])
        XCTAssertFalse(arguments.contains("--model"), "no explicit model preserves the user's Codex default")
        let prompt = await runner.input()
        XCTAssertTrue(prompt.contains("List files"))
        XCTAssertTrue(prompt.contains("Do not execute shell commands"))
        XCTAssertTrue(prompt.contains("Hermes tool-call syntax"))
    }

    func testCodexAddsRequestedModelAndReportsProcessFailures() async {
        let failedRunner = StubCodexRunner(result: .init(stdout: "", stderr: "not logged in", exitStatus: 1))
        let generator = CodexCLIHarnessGenerator(model: "gpt-5.1-codex", runner: failedRunner)
        let failure = await generator.complete(messages: [])
        XCTAssertTrue(failure.contains("not logged in"))
        XCTAssertTrue(failure.contains("codex login"))
        let arguments = await failedRunner.arguments()
        XCTAssertEqual(arguments.suffix(3), ["--model", "gpt-5.1-codex", "-"])

        let unavailable = CodexCLIHarnessGenerator(runner: ThrowingCodexRunner())
        let unavailableReply = await unavailable.complete(messages: [])
        XCTAssertTrue(unavailableReply.contains("codex not installed"))
        XCTAssertTrue(unavailableReply.contains("codex login"))
    }
}
