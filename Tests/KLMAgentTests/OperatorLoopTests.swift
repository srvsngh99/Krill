import XCTest
@testable import KLMAgent

final class OperatorLoopTests: XCTestCase {

    // MARK: - Fixtures

    /// Scripts a fixed sequence of assistant turns. Each call to
    /// `generate` returns the next turn in `replies`, then loops on
    /// the final one (so a stuck-detection test can keep firing the
    /// same call).
    final class ScriptedGenerator: OperatorGenerator, @unchecked Sendable {
        private var index = 0
        private let replies: [String]

        init(_ replies: [String]) {
            self.replies = replies
        }

        func generate(messages: [[String: String]]) async throws -> OperatorTurn {
            let i = min(index, replies.count - 1)
            index += 1
            return OperatorTurn(
                text: replies[i],
                tokenCount: max(1, replies[i].count / 4))
        }
    }

    /// Wraps a closure as a tool. Used to script `pull_model` /
    /// `recommend_model` style behavior in the loop tests.
    struct EchoTool: OperatorTool {
        let name: String
        let description: String
        let parametersJSON: String
        let body: @Sendable ([String: Any]) async throws -> String

        init(
            name: String, description: String = "",
            parametersJSON: String = "{}",
            body: @escaping @Sendable ([String: Any]) async throws -> String
        ) {
            self.name = name
            self.description = description
            self.parametersJSON = parametersJSON
            self.body = body
        }

        func execute(arguments: [String: Any]) async throws -> String {
            try await body(arguments)
        }
    }

    private func collect(_ stream: AsyncStream<OperatorEvent>) async -> [OperatorEvent] {
        var out: [OperatorEvent] = []
        for await event in stream { out.append(event) }
        return out
    }

    // MARK: - Termination conditions

    func testFinishOnNoToolCallEmitsGoalCompleted() async {
        let gen = ScriptedGenerator(["The answer is 42."])
        let loop = OperatorLoop(
            generator: gen,
            tools: OperatorToolRegistry([]),
            systemPrompt: "system")
        let events = await collect(loop.run(goal: "what's the answer?"))
        XCTAssertEqual(events.first, .goalStarted(goal: "what's the answer?"))
        XCTAssertTrue(events.contains { e in
            if case .assistantMessage(let s) = e { return s.contains("42") }
            return false
        })
        XCTAssertTrue(events.last.map {
            if case .goalCompleted = $0 { return true } else { return false }
        } ?? false)
    }

    func testToolCallDispatchAndResultThreadedBack() async {
        // Step 1: model calls echo(arg=hello)
        // Step 2: model replies with the result
        let gen = ScriptedGenerator([
            "<tool_call>{\"name\": \"echo\", \"arguments\": {\"arg\": \"hello\"}}</tool_call>",
            "Tool said: hello.",
        ])
        let echo = EchoTool(name: "echo") { args in
            return (args["arg"] as? String) ?? "(none)"
        }
        let loop = OperatorLoop(
            generator: gen,
            tools: OperatorToolRegistry([echo]),
            systemPrompt: "system")
        let events = await collect(loop.run(goal: "echo hello"))

        let toolStarts = events.compactMap {
            if case .toolCallStarted(let name, _) = $0 { return name }
            return nil
        }
        XCTAssertEqual(toolStarts, ["echo"])

        let toolResults = events.compactMap { evt -> String? in
            if case .toolCallResult(_, let c, _) = evt { return c }
            return nil
        }
        XCTAssertEqual(toolResults, ["hello"])

        XCTAssertTrue(events.contains { evt in
            if case .assistantMessage(let s) = evt {
                return s.contains("hello")
            }
            return false
        })
    }

    func testUnknownToolReturnsErrorResultButLoopContinues() async {
        let gen = ScriptedGenerator([
            "<tool_call>{\"name\": \"nope\", \"arguments\": {}}</tool_call>",
            "I'll stop now.",
        ])
        let loop = OperatorLoop(
            generator: gen,
            tools: OperatorToolRegistry([]),
            systemPrompt: "system")
        let events = await collect(loop.run(goal: "do nothing"))

        XCTAssertTrue(events.contains { evt in
            if case .toolCallResult(_, let content, let isError) = evt {
                return isError && content.contains("unknown tool")
            }
            return false
        })
        XCTAssertTrue(events.last.map {
            if case .goalCompleted = $0 { return true } else { return false }
        } ?? false)
    }

    func testStuckDetectionFiresOnThreeIdenticalToolCalls() async {
        let repeated = "<tool_call>{\"name\": \"echo\", \"arguments\": {\"x\": 1}}</tool_call>"
        let gen = ScriptedGenerator([repeated, repeated, repeated, repeated])
        let echo = EchoTool(name: "echo") { _ in "ok" }
        let loop = OperatorLoop(
            generator: gen,
            tools: OperatorToolRegistry([echo]),
            systemPrompt: "system")
        let events = await collect(loop.run(goal: "loop forever"))
        XCTAssertTrue(events.last.map {
            if case .stuck = $0 { return true } else { return false }
        } ?? false, "expected `.stuck` terminal event, got \(events.last as Any)")
    }

    func testBudgetExhaustedOnMaxSteps() async {
        // Endless tool-calling generator. Vary args each turn so stuck
        // detection does not trip first.
        final class CountingGenerator: OperatorGenerator, @unchecked Sendable {
            var n = 0
            func generate(messages: [[String: String]]) async throws -> OperatorTurn {
                n += 1
                let text =
                    "<tool_call>{\"name\": \"echo\", \"arguments\": {\"n\": \(n)}}</tool_call>"
                return OperatorTurn(text: text, tokenCount: 5)
            }
        }
        let gen = CountingGenerator()
        let echo = EchoTool(name: "echo") { _ in "ok" }
        let loop = OperatorLoop(
            generator: gen,
            tools: OperatorToolRegistry([echo]),
            systemPrompt: "system",
            budget: OperatorBudget(maxSteps: 3))
        let events = await collect(loop.run(goal: "spin"))
        XCTAssertTrue(events.last.map {
            if case .budgetExhausted = $0 { return true } else { return false }
        } ?? false)
    }

    func testBudgetExhaustedOnMaxOutputTokens() async {
        // Each turn claims 1000 tokens; budget is 500 so the first
        // turn already trips it.
        final class FatGenerator: OperatorGenerator, @unchecked Sendable {
            func generate(messages: [[String: String]]) async throws -> OperatorTurn {
                OperatorTurn(text: "x", tokenCount: 1000)
            }
        }
        let loop = OperatorLoop(
            generator: FatGenerator(),
            tools: OperatorToolRegistry([]),
            systemPrompt: "system",
            budget: OperatorBudget(maxOutputTokens: 500))
        let events = await collect(loop.run(goal: "big"))
        XCTAssertTrue(events.last.map {
            if case .budgetExhausted = $0 { return true } else { return false }
        } ?? false)
    }

    // MARK: - JSON event serialization

    func testJSONEncodingOfToolCallStartedPreservesParsedArguments() {
        let evt: OperatorEvent = .toolCallStarted(
            name: "echo", argumentsJSON: "{\"x\": 1}")
        let line = evt.encodeJSON()
        XCTAssertTrue(line.contains("\"type\":\"tool_call_started\""))
        XCTAssertTrue(line.contains("\"name\":\"echo\""))
        XCTAssertTrue(line.contains("\"arguments\""))
    }

    func testJSONEncodingOfWarningEmitsSeverityString() {
        let evt: OperatorEvent = .warning(
            severity: .risky, message: "too big")
        let line = evt.encodeJSON()
        XCTAssertTrue(line.contains("\"severity\":\"risky\""))
        XCTAssertTrue(line.contains("\"message\":\"too big\""))
    }
}

// MARK: - Hermes extractor unit tests

final class HermesExtractorTests: XCTestCase {

    func testExtractsSingleToolCall() {
        let text = "<tool_call>{\"name\": \"hardware_info\", \"arguments\": {}}</tool_call>"
        let (calls, cleaned) = HermesToolCallExtractor.extract(from: text)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "hardware_info")
        XCTAssertEqual(cleaned, "")
    }

    func testExtractsMultipleToolCallsInSourceOrder() {
        let text = """
        I'll check first.
        <tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>
        And then:
        <tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>
        """
        let (calls, cleaned) = HermesToolCallExtractor.extract(from: text)
        XCTAssertEqual(calls.map(\.name), ["a", "b"])
        XCTAssertTrue(cleaned.contains("I'll check first"))
        XCTAssertFalse(cleaned.contains("<tool_call>"))
    }

    func testStrayMarkerWithoutJSONIsDropped() {
        let text = "Some text <tool_call> with no JSON. Final answer: 5."
        let (calls, cleaned) = HermesToolCallExtractor.extract(from: text)
        XCTAssertTrue(calls.isEmpty)
        XCTAssertFalse(cleaned.contains("<tool_call>"))
        XCTAssertTrue(cleaned.contains("Final answer"))
    }

    func testBareJSONObjectWithNameAndArgumentsParsed() {
        let text = "{\"name\": \"x\", \"arguments\": {\"k\": 1}}"
        let (calls, cleaned) = HermesToolCallExtractor.extract(from: text)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "x")
        XCTAssertEqual(cleaned, "")
    }
}
