import XCTest
import NIOCore
import NIOEmbedded
import NIOHTTP1
import KrillEngine
import KrillRegistry
import KrillRuntime
@testable import KrillServer

/// Decision requests through the in-process HTTP handler with registered fixture bundles.
final class DecideEndpointTests: XCTestCase {
    private static let engine = DecisionEngine()
    private var channel: NIOAsyncTestingChannel!
    private var root: URL!
    private var registry: Registry!

    override func setUp() async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("krill-decide-\(UUID().uuidString)")
        registry = Registry(baseDir: root)
        channel = await makeChannel()
    }

    override func tearDown() async throws {
        _ = try? await channel?.finish(acceptAlreadyClosed: true)
        try? FileManager.default.removeItem(at: root)
    }

    private func makeChannel(apiKey: String? = nil, compat: CompatMode = .both) async -> NIOAsyncTestingChannel {
        await NIOAsyncTestingChannel(handler: KrillServer._makeHTTPHandlerForTesting(
            engine: InferenceEngine(modelDirectory: root.appendingPathComponent("none")),
            registry: registry, compat: compat, decisionEngine: Self.engine, apiKey: apiKey))
    }

    private static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("KrillCoreTests/Fixtures/decision")

    private func installTiny(named name: String = "decider") throws {
        try requireMetal()
        try installDecisionModel(named: name, linkingTo: Self.fixtures.appendingPathComponent("tiny").path)
    }

    private func requireMetal() throws {
        guard MLXMetalRuntime.canInitializeMLXForTests else {
            throw XCTSkip("MLX Metal runtime is not available in this test process")
        }
    }

    private func post(_ path: String, _ body: [String: Any], bearer: String? = nil) async throws
        -> (status: Int, json: [String: Any])
    {
        try await post(path, raw: JSONSerialization.data(withJSONObject: body), bearer: bearer)
    }

    private func post(_ path: String, raw: Data, bearer: String? = nil) async throws
        -> (status: Int, json: [String: Any])
    {
        var head = HTTPRequestHead(version: .http1_1, method: .POST, uri: path)
        head.headers.add(name: "Content-Type", value: "application/json")
        if let bearer { head.headers.add(name: "Authorization", value: "Bearer \(bearer)") }
        try await channel.writeInbound(HTTPServerRequestPart.head(head))
        try await channel.writeInbound(HTTPServerRequestPart.body(ByteBuffer(bytes: raw)))
        try await channel.writeInbound(HTTPServerRequestPart.end(nil))
        return try await readResponse()
    }

    private func readResponse() async throws -> (status: Int, json: [String: Any]) {
        guard case .head(let response) = try await channel.waitForOutboundWrite(as: HTTPServerResponsePart.self),
              case .body(.byteBuffer(let bytes)) = try await channel.waitForOutboundWrite(as: HTTPServerResponsePart.self),
              case .end = try await channel.waitForOutboundWrite(as: HTTPServerResponsePart.self) else {
            XCTFail("Expected one JSON response")
            return (-1, [:])
        }
        let json = try JSONSerialization.jsonObject(with: Data(bytes.readableBytesView)) as? [String: Any]
        return (Int(response.status.code), try XCTUnwrap(json))
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
        XCTAssertEqual(checked, 37)
    }

    func testMatchesRealBundleGolden() async throws {
        guard let path = ProcessInfo.processInfo.environment["KRILL_DECISION_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KRILL_DECISION_MODEL_PATH not set; skipping real-bundle endpoint test")
        }
        try requireMetal()
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

    func testLongYesNoQuestionHasItsOwnError() async throws {
        try installTiny()
        let r = try await post("/v1/decide", [
            "model": "decider", "text": "hi", "yes_no": String(repeating: "is it late ", count: 40)])
        XCTAssertEqual(r.status, 400)
        XCTAssertTrue((r.json["error"] as? String)?.contains("the yes_no question") ?? false, "\(r.json)")
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
            ["model": "m", "text": "hi", "yes_no": "Is it?", "question": "Which?"],
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
        let tooManyYesNo = try await post("/v1/decide", [
            "model": "decider", "texts": Array(repeating: "hi", count: 257), "yes_no": "Is it?"])
        XCTAssertEqual(tooManyYesNo.status, 400)
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
        _ = try await channel.finish()
        channel = await makeChannel(apiKey: "secret")
        let body: [String: Any] = ["model": "m", "text": "hi", "options": ["a"]]
        let denied = try await post("/v1/decide", body)
        XCTAssertEqual(denied.status, 401)
        let allowed = try await post("/v1/decide", body, bearer: "secret")
        XCTAssertEqual(allowed.status, 404)
    }

    // MARK: - Helpers

    private func get(_ path: String) async throws -> (status: Int, json: [String: Any]) {
        try await channel.writeInbound(HTTPServerRequestPart.head(HTTPRequestHead(version: .http1_1, method: .GET, uri: path)))
        try await channel.writeInbound(HTTPServerRequestPart.end(nil))
        return try await readResponse()
    }

    private static func golden(_ name: String) throws -> [[String: Any]] {
        let file = fixtures.appendingPathComponent(name)
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]
        return try XCTUnwrap(json?["cases"] as? [[String: Any]])
    }
}

extension DecideEndpointTests {
    private func systemOneAnswers(_ response: (status: Int, json: [String: Any])) throws -> [String: [String: Any]] {
        XCTAssertEqual(response.status, 200, "\(response.json)")
        XCTAssertEqual(response.json["model"] as? String, "decider")
        return try XCTUnwrap(response.json["answers"] as? [String: [String: Any]])
    }

    private func checkProbabilities(_ answer: [String: Any], keys: [String]) throws -> [Double] {
        let probabilities = try XCTUnwrap(answer["probabilities"] as? [String: Double])
        XCTAssertEqual(Set(probabilities.keys), Set(keys))
        let p = try keys.map { try XCTUnwrap(probabilities[$0]) }
        XCTAssertTrue(p.allSatisfy { (0 ... 1).contains($0) })
        XCTAssertEqual(p.reduce(0, +), 1, accuracy: 1e-12)
        let entropy = -p.filter { $0 > 0 }.reduce(0) { $0 + $1 * log($1) }
        XCTAssertEqual(try XCTUnwrap(answer["confidence"] as? Double), 1 - entropy / log(Double(p.count)),
                       accuracy: 1e-12)
        return p
    }

    func testSystemOneRendersAmberStructuredStateInWireOrder() throws {
        let questions = #""questions":{"q":{"type":"noul","instructions":"Is it urgent?"}}"#
        let state = #"{"z":"line\nnext","a":{"y":[1,true,null],"b":"\u00e9"}}"#
        let body = #"{"model":"amber","state":\#(state),\#(questions)}"#
        let request = try SystemOneRequest(ByteBuffer(string: body))
        XCTAssertEqual(request.amberState, "z: line\nnext\na: {\"y\": [1, true, null], \"b\": \"é\"}")

        for empty in ["{}", "[]"] {
            let body = #"{"model":"amber","state":\#(empty),\#(questions)}"#
            XCTAssertEqual(try SystemOneRequest(ByteBuffer(string: body)).amberState, "")
        }
    }

    func testSystemOneRendersDMJEPAStructuredStateInWireOrder() throws {
        let questions = #""questions":{"q":{"type":"noul","instructions":"Is it urgent?"}}"#
        let state = #"{"z":{"emoji":"😀","empty":{}},"a":[1,true,null,[],1.20E-6]}"#
        let body = #"{"model":"dm-jepa","state":\#(state),\#(questions)}"#
        let request = try SystemOneRequest(ByteBuffer(string: body))
        XCTAssertEqual(request.dmjepaState, #"""
        {
          "z": {
            "emoji": "\ud83d\ude00",
            "empty": {}
          },
          "a": [
            1,
            true,
            null,
            [],
            1.2e-06
          ]
        }
        """#)
        for empty in ["{}", "[]"] {
            let body = #"{"model":"dm-jepa","state":\#(empty),\#(questions)}"#
            XCTAssertEqual(try SystemOneRequest(ByteBuffer(string: body)).dmjepaState, "")
        }
    }

    func testSystemOneChoiceKeepsWireOrderOnTie() async throws {
        try installTiny()
        let raw = #"""
        {"state":[{"criteria":{"same":0,"z-option":1},"text":"[\"}]"}],
         "questions":{"best":{"criteria":{"z\u002doption":"same","same":null},
                              "instructions":"Which fits?","type":"choice"}},
         "model":"decider"}
        """#
        let response = try await post("/v1/systemone", raw: Data(raw.utf8))
        let answer = try XCTUnwrap(systemOneAnswers(response)["best"])
        XCTAssertEqual(Set(answer.keys), ["type", "choice", "probabilities", "confidence"])
        XCTAssertEqual(answer["type"] as? String, "choice")
        XCTAssertEqual(answer["choice"] as? String, "z-option")
        XCTAssertEqual(try checkProbabilities(answer, keys: ["z-option", "same"]), [0.5, 0.5])
    }

    func testSystemOneChoiceAccepts255Criteria() async throws {
        try installTiny()
        let keys = (0 ..< 255).reversed().map { "option-\($0)" }
        let criteria = keys.map { "\"\($0)\":\"same\"" }.joined(separator: ",")
        let raw = """
        {"model":"decider","state":"hi","questions":{"best":{
          "type":"choice","instructions":"Which fits?","criteria":{\(criteria)}}}}
        """
        let response = try await post("/v1/systemone", raw: Data(raw.utf8))
        let answer = try XCTUnwrap(systemOneAnswers(response)["best"])
        XCTAssertEqual(answer["choice"] as? String, keys.first)
        XCTAssertEqual(try checkProbabilities(answer, keys: keys).count, 255)
    }

    func testSystemOneNoulUsesYesNoAndSupportsCriteria() async throws {
        try installTiny()
        let text = "The invoice was paid in full on Monday."
        let question = "Is the invoice fully paid?"
        let response = try await post("/v1/systemone", [
            "model": "decider", "state": text, "questions": [
                "paid": ["type": "noul", "instructions": question],
                "custom": ["type": "noul", "instructions": question, "criteria": ["true": "paid"]],
            ],
        ])
        let answers = try systemOneAnswers(response)
        let paid = try XCTUnwrap(answers["paid"])
        let custom = try XCTUnwrap(answers["custom"])
        XCTAssertEqual(Set(paid.keys), ["type", "noul"])
        XCTAssertEqual(paid["type"] as? String, "noul")
        XCTAssertEqual(Set(custom.keys), ["type", "noul"])
        XCTAssertEqual(custom["type"] as? String, "noul")

        let yesNo = try await post("/v1/decide", ["model": "decider", "text": text, "yes_no": question])
        let native = try XCTUnwrap((yesNo.json["results"] as? [[String: Any]])?.first?["p_yes"] as? Double)
        XCTAssertEqual(try XCTUnwrap(paid["noul"] as? Double), native, accuracy: 1e-6)
        let choice = try await post("/v1/decide", [
            "model": "decider", "text": text, "question": question, "options": ["paid", "false"], "allow_none": false,
        ])
        let scores = try XCTUnwrap((choice.json["results"] as? [[String: Any]])?.first?["scores"] as? [Double])
        XCTAssertEqual(try XCTUnwrap(custom["noul"] as? Double), scores[0] / scores.reduce(0, +), accuracy: 1e-6)
    }

    func testSystemOneScoreReturnsExpectationAndLegend() async throws {
        try installTiny()
        let criteria = ["low", "medium", "high"]
        let body: [String: Any] = ["model": "decider", "state": "The checkout is failing.", "questions": [
            "severity": ["type": "score", "instructions": "How severe is it?", "criteria": criteria],
        ]]
        let response = try await post("/v1/systemone", body)
        let answer = try XCTUnwrap(systemOneAnswers(response)["severity"])
        XCTAssertEqual(Set(answer.keys), ["type", "score", "legend", "probabilities", "confidence"])
        XCTAssertEqual(answer["type"] as? String, "score")
        XCTAssertEqual(answer["legend"] as? [String: String], ["0": "low", "1": "medium", "2": "high"])
        let p = try checkProbabilities(answer, keys: ["0", "1", "2"])
        XCTAssertEqual(try XCTUnwrap(answer["score"] as? Double), p[1] + 2 * p[2], accuracy: 1e-12)
    }

    func testSystemOneMixedQuestionsAndUsage() async throws {
        try installTiny()
        let questions: [String: [String: Any]] = [
            "category": ["type": "choice", "instructions": "Which category?", "criteria": ["a": "low", "b": "high"]],
            "urgent": ["type": "noul", "instructions": "Is it urgent?"],
            "severity.1": ["type": "score", "instructions": "How severe?", "criteria": ["low", "high"]],
        ]
        let body: [String: Any] = ["model": "decider", "state": ["message": "Checkout is down"]]
        let response = try await post("/v1/systemone", body.merging([
            "questions": questions, "images": [], "keep_alive": "5m",
        ]) { $1 })
        let answers = try systemOneAnswers(response)
        XCTAssertEqual(Set(answers.keys), Set(questions.keys))
        let usage = try XCTUnwrap(response.json["usage"] as? [String: Int])
        XCTAssertEqual(usage["output_tokens"], 3)
        var expectedTokens = 0
        for (name, question) in questions {
            let single = try await post("/v1/systemone", body.merging(["questions": [name: question]]) { $1 })
            let answer = try XCTUnwrap(systemOneAnswers(single)[name])
            XCTAssertEqual(answers[name]?["type"] as? String, answer["type"] as? String)
            let singleUsage = try XCTUnwrap(single.json["usage"] as? [String: Int])
            let tokens = try XCTUnwrap(singleUsage["input_tokens"])
            XCTAssertGreaterThan(tokens, 0)
            XCTAssertEqual(singleUsage["output_tokens"], 1)
            expectedTokens += tokens
        }
        XCTAssertEqual(usage["input_tokens"], expectedTokens)
    }

    func testSystemOneRejectsInvalidRequests() async throws {
        let choice: [String: Any] = ["type": "choice", "instructions": "Which?", "criteria": ["a": "A", "b": "B"]]
        let valid: [String: Any] = ["model": "m", "state": "hi", "questions": ["q": choice]]
        let bad: [[String: Any]] = [
            ["model": " \n"], ["model": 1], ["state": " \n"], ["state": true],
            ["images": ["aGVsbG8="]], ["images": "text"], ["keep_alive": true],
            ["keep_alive": 1e30], ["keep_alive": "\(Int.max)d"], ["keep_alive": "5min"],
            ["questions": [:]], ["questions": [choice]], ["questions": ["q\n": choice]],
            ["questions": [String(repeating: "a", count: 101): choice]],
            ["questions": Dictionary(uniqueKeysWithValues: (0 ..< 65).map { ("q\($0)", choice) })],
        ]
        let badQuestions: [[String: Any]] = [
            ["type": "other", "instructions": "Which?"],
            ["type": "choice", "instructions": " \n", "criteria": ["a": "A", "b": "B"]],
            ["type": "choice", "instructions": "Which?", "criteria": [" ": "A", "b": "B"]],
            ["type": "choice", "instructions": "Which?", "criteria": ["a": "A", "b": 1]],
            ["type": "choice", "instructions": "Which?", "criteria": ["a": "A"]],
            ["type": "choice", "instructions": "Which?",
             "criteria": Dictionary(uniqueKeysWithValues: (0 ..< 256).map { ("k\($0)", "A") })],
            ["type": "noul", "instructions": "Is it?", "criteria": ["yes": "A"]],
            ["type": "noul", "instructions": "Is it?", "criteria": ["true": NSNull()]],
            ["type": "score", "instructions": "How?", "criteria": ["low", " "]],
            ["type": "score", "instructions": "How?", "criteria": Array(repeating: "low", count: 11)],
        ]
        for fields in bad + badQuestions.map({ ["questions": ["q": $0]] }) {
            let response = try await post("/v1/systemone", valid.merging(fields) { $1 })
            XCTAssertEqual(response.status, 400, "\(fields)")
            XCTAssertNotNil(response.json["error"] as? String)
        }
        let images = try await post("/v1/systemone", valid.merging(["images": ["aGVsbG8="]]) { $1 })
        XCTAssertTrue((images.json["error"] as? String)?.contains("text-only") ?? false)
        let malformed = try await post("/v1/systemone", raw: Data("{nope".utf8))
        XCTAssertEqual(malformed.status, 400)
        for key in ["model", "state", "questions"] {
            let response = try await post("/v1/systemone", valid.filter { $0.key != key })
            XCTAssertEqual(response.status, 400, "Missing \(key)")
        }
    }

    func testSystemOneModelLookup() async throws {
        try registry.saveManifest(ServerTests.fixtureManifest(name: "chat"))
        let questions = ["q": ["type": "noul", "instructions": "Is it?"]]
        for (name, expected) in [("unknown", 404), ("chat", 400)] {
            let response = try await post("/v1/systemone", ["model": name, "state": "hi", "questions": questions])
            XCTAssertEqual(response.status, expected, "\(response.json)")
            XCTAssertNotNil(response.json["error"] as? String)
        }
        let validChoice = #"""
        {"model":"unknown","state":{"criteria":{"other":0},"text":"[\"}]"},
         "questions":{"best":{"criteria":{"z\u002doption":"same","same":null},
                              "instructions":"Which fits?","type":"choice"}}}
        """#
        let response = try await post("/v1/systemone", raw: Data(validChoice.utf8))
        XCTAssertEqual(response.status, 404, "\(response.json)")
    }

    func testSystemOneLoadFailureAndReaderLimit() async throws {
        try installTiny()
        try installDecisionModel(named: "broken", linkingTo: nil)
        let questions: [String: [String: Any]] = [
            "q": ["type": "score", "instructions": "How severe?", "criteria": ["low", "high"]],
        ]
        let broken = try await post("/v1/systemone", ["model": "broken", "state": "hi", "questions": questions])
        XCTAssertEqual(broken.status, 500)
        XCTAssertEqual(broken.json["error"] as? String, "systemone failed")
        let response = try await post("/v1/systemone", [
            "model": "decider", "state": String(repeating: "word ", count: 2_200), "questions": questions,
        ])
        XCTAssertEqual(response.status, 400)
        XCTAssertTrue((response.json["error"] as? String)?.contains("limit is 2048") ?? false)
    }

    func testSystemOneTextLimitUsesUTF8AndIncludesQuestions() async throws {
        let questions = ["q": ["type": "noul", "instructions": "Is it?"]]
        let schemaSize = try JSONSerialization.data(withJSONObject: questions).count
        let state = String(repeating: "é", count: (64 * 1024 - schemaSize) / 2)
        let valid: [String: Any] = ["model": "unknown", "state": state, "questions": questions]
        let atLimit = try await post("/v1/systemone", valid)
        XCTAssertEqual(atLimit.status, 404)
        let overLimit = try await post("/v1/systemone", valid.merging(["state": state + "é"]) { $1 })
        XCTAssertEqual(overLimit.status, 413)
        XCTAssertEqual(overLimit.json["error"] as? String, "text and schema must not exceed 64 KiB")
    }

    func testSystemOneBodyLimit() async throws {
        let head = HTTPRequestHead(version: .http1_1, method: .POST, uri: "/v1/systemone")
        try await channel.writeInbound(HTTPServerRequestPart.head(head))
        var bytes = channel.allocator.buffer(capacity: 32 * 1024 * 1024 + 1)
        bytes.writeRepeatingByte(0x20, count: 32 * 1024 * 1024)
        try await channel.writeInbound(HTTPServerRequestPart.body(bytes))
        let pending = try await channel.readOutbound(as: HTTPServerResponsePart.self)
        XCTAssertNil(pending)
        try await channel.writeInbound(HTTPServerRequestPart.body(ByteBuffer(string: " ")))
        let response = try await readResponse()
        XCTAssertEqual(response.status, 413)
        XCTAssertEqual(response.json["error"] as? String, "request body must not exceed 32 MiB")
    }

    func testSystemOneWorksInOllamaModeWithBearerKey() async throws {
        _ = try await channel.finish()
        channel = await makeChannel(apiKey: "secret", compat: .ollama)
        let body: [String: Any] = [
            "model": "unknown", "state": "hi", "questions": ["q": ["type": "noul", "instructions": "Is it?"]],
        ]
        let denied = try await post("/v1/systemone", body)
        XCTAssertEqual(denied.status, 401)
        let allowed = try await post("/v1/systemone", body, bearer: "secret")
        XCTAssertEqual(allowed.status, 404)
        XCTAssertEqual(allowed.json["error"] as? String, "decision model not found")
    }
}
