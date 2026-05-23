import XCTest
import KLMRegistry
@testable import KLMAgent

/// Pin the loop's `.familyAware` extractor against the family-tagged
/// wire formats `KLMServer.AgentToolBridge` knows about.
///
/// We don't try to re-test every parser branch (those are covered by
/// the server's own suite); we just confirm the bridge wiring is
/// faithful for each format the operator agent will see in practice.
final class FamilyAwareExtractorTests: XCTestCase {

    func testHermesFormatForQwenFamily() {
        let text =
            "<tool_call>{\"name\": \"server_status\", \"arguments\": {}}</tool_call>"
        let (calls, _) = FamilyAwareToolCallExtractor.extract(
            from: text, family: ModelFamily.qwen.rawValue)
        XCTAssertEqual(calls.map(\.name), ["server_status"])
    }

    func testLlamaNativeFormatBareJSON() {
        // Llama 3 emits a bare {"name":..., "parameters": ...} object.
        let text =
            "{\"name\": \"hardware_info\", \"parameters\": {}}"
        let (calls, _) = FamilyAwareToolCallExtractor.extract(
            from: text, family: ModelFamily.llama.rawValue)
        XCTAssertEqual(calls.map(\.name), ["hardware_info"])
    }

    func testGemma4NativeFormatCallSentinels() {
        // Gemma 4 native: <|tool_call>call:NAME{...args...}<tool_call|>
        let text = "<|tool_call>call:disk_usage{}<tool_call|>"
        let (calls, _) = FamilyAwareToolCallExtractor.extract(
            from: text, family: ModelFamily.gemma4.rawValue)
        XCTAssertEqual(calls.map(\.name), ["disk_usage"])
    }

    func testUnknownFamilyFallsThroughToHermes() {
        let text =
            "<tool_call>{\"name\": \"hardware_info\", \"arguments\": {}}</tool_call>"
        let (calls, _) = FamilyAwareToolCallExtractor.extract(
            from: text, family: "totally-fake-family")
        XCTAssertEqual(calls.map(\.name), ["hardware_info"])
    }

    func testNilFamilyFallsThroughToHermes() {
        let text =
            "<tool_call>{\"name\": \"hardware_info\", \"arguments\": {}}</tool_call>"
        let (calls, _) = FamilyAwareToolCallExtractor.extract(
            from: text, family: nil)
        XCTAssertEqual(calls.map(\.name), ["hardware_info"])
    }

    // MARK: - End-to-end: loop + default toolset + scripted generator

    func testLoopRunsScriptedGoalAgainstDefaultToolset() async throws {
        // The model: pretends to be the router; on turn 1 calls
        // hardware_info, on turn 2 writes the final answer.
        final class Scripted: OperatorGenerator, @unchecked Sendable {
            var index = 0
            let replies = [
                "<tool_call>{\"name\": \"hardware_info\", \"arguments\": {}}</tool_call>",
                "Your machine has 32 GB RAM.",
            ]
            func generate(messages: [[String: String]]) async throws -> OperatorTurn {
                let r = replies[min(index, replies.count - 1)]
                index += 1
                return OperatorTurn(text: r, tokenCount: 10)
            }
        }

        // Wire a minimal real toolset. Hardware is the static fixture
        // from ToolWrapperTests so the assistant prose can ground out.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("klmagent-e2e-\(UUID().uuidString)")
        let reg = Registry(baseDir: dir)
        try reg.ensureDirectories()
        let hw: @Sendable () -> HardwareInfo = {
            ToolWrapperTests.staticFixtureHardware
        }
        struct OfflineOps: AgentDaemonOps {
            func status() async -> AgentDaemonStatus? { nil }
            func loadModel(_ name: String) async throws -> String { "n/a" }
            func unloadModel() async throws -> String { "n/a" }
        }
        struct NoopPuller: AgentPuller {
            func pull(alias: String) async throws -> String { "n/a" }
        }
        let toolset = DefaultOperatorToolset.make(
            registry: reg,
            daemonOps: OfflineOps(),
            puller: NoopPuller(),
            hardware: hw)
        let loop = OperatorLoop(
            generator: Scripted(),
            tools: toolset,
            systemPrompt: "test",
            toolFormat: .familyAware(family: "qwen"))
        var events: [OperatorEvent] = []
        for await ev in loop.run(goal: "what is my ram?") {
            events.append(ev)
        }
        // Must have run hardware_info exactly once and finished.
        let toolStarts = events.compactMap { evt -> String? in
            if case .toolCallStarted(let n, _) = evt { return n }
            return nil
        }
        XCTAssertEqual(toolStarts, ["hardware_info"])
        let toolResults = events.compactMap { evt -> String? in
            if case .toolCallResult(_, let c, _) = evt { return c }
            return nil
        }
        XCTAssertEqual(toolResults.count, 1)
        XCTAssertTrue(toolResults.first?.contains("\"total_ram_gb\":32") ?? false,
                      "expected hardware_info JSON in tool result")
        XCTAssertTrue(events.last.map {
            if case .goalCompleted = $0 { return true } else { return false }
        } ?? false)
    }
}
