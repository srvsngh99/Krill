import XCTest
import NIOCore
import NIOHTTP1
import NIOPosix
import KrillEngine
import KrillRegistry
@testable import KrillServer

/// `POST /v1/decide` over a real in-process server. Most tests register the tiny fixture bundle
/// under a model name; `testMatchesRealBundleGolden` needs `KRILL_DECISION_MODEL_PATH` and skips without it.
final class DecideEndpointTests: XCTestCase {
    private static let engine = DecisionEngine()
    private var group: MultiThreadedEventLoopGroup!
    private var channel: Channel!
    private var base: URL!
    private var root: URL!
    private var registry: Registry!

    override func setUp() async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("krill-decide-\(UUID().uuidString)")
        registry = Registry(baseDir: root)
        try await startServer(apiKey: nil)
    }

    override func tearDown() async throws {
        try? await channel?.close()
        try? await group?.shutdownGracefully()
        try? FileManager.default.removeItem(at: root)
    }

    private func startServer(apiKey: String?) async throws {
        group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let registry = self.registry!
        var lastError: Error?
        for port in 9300 ... 9399 {
            let bootstrap = ServerBootstrap(group: group).childChannelInitializer { ch in
                ch.pipeline.configureHTTPServerPipeline().flatMap {
                    ch.pipeline.addHandler(KrillServer._makeHTTPHandlerForTesting(
                        engine: InferenceEngine(modelDirectory: registry.baseDir.appendingPathComponent("none")),
                        registry: registry, decisionEngine: Self.engine, apiKey: apiKey))
                }
            }
            do {
                channel = try await bootstrap.bind(host: "127.0.0.1", port: port).get()
                base = URL(string: "http://127.0.0.1:\(port)")!
                return
            } catch let error as IOError where error.errnoCode == EADDRINUSE {
                lastError = error
            }
        }
        throw NSError(domain: "DecideEndpointTests", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "no free scratch port in 9300-9399: \(String(describing: lastError))"])
    }

    private static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("KrillCoreTests/Fixtures/decision")

    private func installTiny(named name: String = "decider") throws {
        try installDecisionModel(named: name, linkingTo: Self.fixtures.appendingPathComponent("tiny").path)
    }

    private func post(_ path: String, _ body: [String: Any], bearer: String? = nil) async throws
        -> (status: Int, json: [String: Any])
    {
        try await post(path, raw: JSONSerialization.data(withJSONObject: body), bearer: bearer)
    }

    private func post(_ path: String, raw: Data, bearer: String? = nil) async throws
        -> (status: Int, json: [String: Any])
    {
        var request = URLRequest(url: base.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let bearer { request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization") }
        request.httpBody = raw
        let (data, response) = try await URLSession.shared.data(for: request)
        let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        return ((response as! HTTPURLResponse).statusCode, json)
    }

    private func installDecisionModel(named name: String, linkingTo bundle: String?) throws {
        try registry.saveManifest(ModelManifest(
            name: name, family: .decision, params: "150M", quant: "fp16", source: "local", context: 2048,
            files: [], draftPair: nil, chatTemplate: "none", sizeBytes: 1, pulledAt: Date()))
        if let bundle {
            let fm = FileManager.default
            let blob = registry.modelPath(name)
            try fm.createDirectory(at: blob, withIntermediateDirectories: true)
            for file in try fm.contentsOfDirectory(atPath: bundle) {
                try fm.createSymbolicLink(
                    at: blob.appendingPathComponent(file),
                    withDestinationURL: URL(fileURLWithPath: bundle).appendingPathComponent(file))
            }
        }
    }

    private let options = ["billing question", "password reset", "shipping delay"]

    // MARK: - Results

    func testChoiceByRegisteredName() async throws {
        try installTiny()
        let (status, json) = try await post("/v1/decide", [
            "model": "decider", "text": "I forgot my password and cannot sign in.", "options": options])
        XCTAssertEqual(status, 200, "\(json)")
        XCTAssertEqual(json["model"] as? String, "decider")
        let result = try XCTUnwrap((json["results"] as? [[String: Any]])?.first)
        XCTAssertTrue(options.contains(try XCTUnwrap(result["label"] as? String)))
        XCTAssertEqual((result["scores"] as? [Double])?.count, 3)
        XCTAssertNotNil(result["p_none"] as? Double)
        let (_, models) = try await get("/v1/models")
        XCTAssertTrue((models["data"] as? [[String: Any]] ?? []).contains { $0["id"] as? String == "decider" })
    }

    func testBatch() async throws {
        try installTiny()
        let (status, json) = try await post("/v1/decide", [
            "model": "decider",
            "texts": ["My card was charged twice.", "Where is my parcel? It is a week late."],
            "options": options])
        XCTAssertEqual(status, 200)
        XCTAssertEqual((json["results"] as? [[String: Any]])?.count, 2)
    }

    func testYesNo() async throws {
        try installTiny()
        let (status, json) = try await post("/v1/decide", [
            "model": "decider", "text": "The invoice was paid in full on Monday.",
            "yes_no": "Is the invoice fully paid?"])
        XCTAssertEqual(status, 200)
        let result = try XCTUnwrap((json["results"] as? [[String: Any]])?.first)
        XCTAssertTrue(["yes", "no"].contains(result["answer"] as? String ?? ""))
        XCTAssertNotNil(result["p_yes"] as? Double)
    }

    func testNoneResultAndAllowNoneFalse() async throws {
        try installTiny()
        let c = try XCTUnwrap(Self.golden("golden_tiny.json").first { $0["id"] as? String == "c13" })
        XCTAssertTrue(c["index"] is NSNull)
        var body: [String: Any] = ["model": "decider", "text": c["text"]!, "options": c["options"]!]
        let none = try await post("/v1/decide", body)
        let r = try XCTUnwrap((none.json["results"] as? [[String: Any]])?.first)
        XCTAssertTrue(r["label"] is NSNull)
        XCTAssertTrue(r["index"] is NSNull)

        body["allow_none"] = false
        let forced = try await post("/v1/decide", body)
        XCTAssertNotNil(((forced.json["results"] as? [[String: Any]])?.first)?["label"] as? String)
    }

    func testMatchesTinyGolden() async throws {
        try installTiny()
        let checked = try await checkGolden("golden_tiny.json", model: "decider")
        XCTAssertEqual(checked, 35)
    }

    func testMatchesRealBundleGolden() async throws {
        guard let path = ProcessInfo.processInfo.environment["KRILL_DECISION_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KRILL_DECISION_MODEL_PATH not set; skipping real-bundle endpoint test")
        }
        try installDecisionModel(named: "real", linkingTo: path)
        let checked = try await checkGolden("golden.json", model: "real")
        XCTAssertEqual(checked, 35)
    }

    private func checkGolden(_ file: String, model: String) async throws -> Int {
        var checked = 0
        for c in try Self.golden(file) {
            var body: [String: Any] = ["model": model, "text": c["text"] as! String]
            if let q = c["question"] as? String, c["kind"] as? String == "yesno" { body["yes_no"] = q }
            else {
                body["options"] = c["options"]
                body["allow_none"] = c["allow_none"]
                if let q = c["question"] as? String { body["question"] = q }
            }
            let (status, json) = try await post("/v1/decide", body)
            XCTAssertEqual(status, 200, c["id"] as! String)
            let r = try XCTUnwrap((json["results"] as? [[String: Any]])?.first)
            if c["kind"] as? String == "yesno" {
                XCTAssertEqual(r["answer"] as? String, c["answer"] as? String, c["id"] as! String)
                XCTAssertEqual(r["p_yes"] as! Double, c["p_yes"] as! Double, accuracy: 1e-3)
            } else {
                XCTAssertEqual(r["index"] as? Int, c["index"] as? Int, c["id"] as! String)
                XCTAssertEqual(r["p_none"] as! Double, c["p_none"] as! Double, accuracy: 1e-3)
                for (a, b) in zip(r["scores"] as! [Double], c["scores"] as! [Double]) {
                    XCTAssertEqual(a, b, accuracy: 1e-3, c["id"] as! String)
                }
            }
            checked += 1
        }
        return checked
    }

    // MARK: - Errors

    func testOverLimitInputIsBadRequest() async throws {
        try installTiny()
        let longText = String(repeating: "word ", count: 2_200)
        let text = try await post("/v1/decide", ["model": "decider", "text": longText, "options": options])
        XCTAssertEqual(text.status, 400)
        XCTAssertTrue((text.json["error"] as? String)?.contains("limit is 2048") ?? false)

        let longOption = String(repeating: "refund request ", count: 40)
        let option = try await post("/v1/decide", ["model": "decider", "text": "hi", "options": [longOption, "other"]])
        XCTAssertEqual(option.status, 400)
        XCTAssertTrue((option.json["error"] as? String)?.contains("limit is 32") ?? false)
    }

    func testMalformedRequestsAreBadRequest() async throws {
        let bad = try await post("/v1/decide", raw: Data("{nope".utf8))
        XCTAssertEqual(bad.status, 400)
        let cases: [[String: Any]] = [
            ["text": "hi", "options": ["a"]],
            ["model": "m", "options": ["a"]],
            ["model": "m", "text": "hi"],
            ["model": "m", "text": "hi", "options": []],
            ["model": "m", "text": "hi", "options": ["a"], "yes_no": "Is it?"],
            ["model": "m", "text": "hi", "texts": ["hi"], "options": ["a"]],
            ["model": "m", "texts": [], "options": ["a"]],
            ["model": "m", "text": "hi", "options": ["a"], "allow_none": "yes"],
            ["model": "m", "text": "hi", "yes_no": ""],
            ["model": "m", "text": "hi", "options": ["a", 2]],
        ]
        for body in cases {
            let r = try await post("/v1/decide", body)
            XCTAssertEqual(r.status, 400, "\(body)")
            XCTAssertNotNil(r.json["error"] as? String, "\(body)")
        }
    }

    func testRequestSizeCapsAndEmptyInputsAreBadRequest() async throws {
        try installTiny()
        let tooMany = try await post("/v1/decide", [
            "model": "decider", "texts": Array(repeating: "hi", count: 257), "options": options])
        XCTAssertEqual(tooMany.status, 400)
        let tooManyOptions = try await post("/v1/decide", [
            "model": "decider", "text": "hi", "options": (0 ..< 129).map { "option \($0)" }])
        XCTAssertEqual(tooManyOptions.status, 400)
        let emptyText = try await post("/v1/decide", ["model": "decider", "text": "", "options": options])
        XCTAssertEqual(emptyText.status, 400)
        let emptyOption = try await post("/v1/decide", ["model": "decider", "text": "hi", "options": ["a", ""]])
        XCTAssertEqual(emptyOption.status, 400)
        let wide = try await post("/v1/decide", [
            "model": "decider", "texts": (0 ..< 256).map { "ticket \($0)" }, "options": (0 ..< 77).map { "option \($0)" }])
        XCTAssertEqual(wide.status, 200)
    }

    func testPathsAreNotAcceptedAsModel() async throws {
        let body: [String: Any] = ["text": "hi", "options": ["a", "b"]]
        let fixture = Self.fixtures.appendingPathComponent("tiny").path
        for name in ["nope", "/no/such/bundle", fixture, root.path] {
            let r = try await post("/v1/decide", body.merging(["model": name]) { $1 })
            XCTAssertEqual(r.status, 404, name)
            XCTAssertEqual(r.json["error"] as? String, "decision model not found")
        }
    }

    func testChatModelIsNotADecisionModel() async throws {
        try registry.saveManifest(ServerTests.fixtureManifest(name: "qwen-chat"))
        let r = try await post("/v1/decide", ["model": "qwen-chat", "text": "hi", "options": ["a", "b"]])
        XCTAssertEqual(r.status, 400)
        XCTAssertTrue((r.json["error"] as? String)?.contains("not a decision model") ?? false)
    }

    func testDecisionModelSentToOtherRoutesIsRefused() async throws {
        try installDecisionModel(named: "decider", linkingTo: nil)
        let rerank = try await post("/v1/rerank", ["model": "decider", "query": "q", "documents": ["d"]])
        XCTAssertEqual(rerank.status, 400)
        XCTAssertTrue((rerank.json["error"] as? String)?.contains("not a reranker") ?? false)
        let embed = try await post("/v1/embeddings", ["model": "decider", "input": "hi"])
        XCTAssertEqual(embed.status, 400)
        XCTAssertTrue((embed.json["error"] as? String)?.contains("not a sentence-embedding") ?? false)
    }

    func testDecisionBundleSentToChatIsRefused() async throws {
        try installTiny()
        let chat = try await post("/v1/chat/completions", [
            "model": "decider", "messages": [["role": "user", "content": "hi"]]])
        XCTAssertGreaterThanOrEqual(chat.status, 400)
        XCTAssertTrue("\(chat.json)".contains("/v1/decide"))
    }

    func testRequiresBearerKeyWhenConfigured() async throws {
        try? await channel.close()
        try? await group.shutdownGracefully()
        try await startServer(apiKey: "secret")
        let body: [String: Any] = ["model": "m", "text": "hi", "options": ["a"]]
        let denied = try await post("/v1/decide", body)
        XCTAssertEqual(denied.status, 401)
        let allowed = try await post("/v1/decide", body, bearer: "secret")
        XCTAssertEqual(allowed.status, 404)
    }

    // MARK: - Helpers

    private func get(_ path: String) async throws -> (status: Int, json: [String: Any]) {
        let (data, response) = try await URLSession.shared.data(from: base.appendingPathComponent(path))
        let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        return ((response as! HTTPURLResponse).statusCode, json)
    }

    private static func golden(_ name: String) throws -> [[String: Any]] {
        let file = fixtures.appendingPathComponent(name)
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]
        return try XCTUnwrap(json?["cases"] as? [[String: Any]])
    }
}
