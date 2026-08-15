import XCTest
@testable import KrillTokenizer

/// Gates for the `reasoning_effort` control Qwen3.8 adds to the `qwen3_5` chat
/// template.
///
/// The qwen3_5 prompt is hand-rolled as ChatML rather than rendered through
/// Jinja, so the template's `reasoning_instructions` block — a system-message
/// injection that fires only when thinking is on — has to be mirrored in Swift.
/// That mirroring is invisible from the model's output: asked to quote its own
/// system message, the model answered "NONE" and then quoted the injected
/// sentence back as if it came from the user turn. These assert the token
/// stream instead of trusting the model's self-report.
final class Qwen38ReasoningEffortTests: XCTestCase {

    private func requireTokenizer() async throws -> KrillTokenizer {
        let env = ProcessInfo.processInfo.environment
        guard let dir = [env["KRILL_QWEN35_MODEL_PATH"], env["KRILL_ORNITH_MODEL_PATH"]]
            .compactMap({ $0 }).first(where: { !$0.isEmpty }) else {
            throw XCTSkip("KRILL_QWEN35_MODEL_PATH not set")
        }
        return try await KrillTokenizer(from: URL(fileURLWithPath: dir))
    }

    /// Render and decode back, so the assertion is about the actual prompt text
    /// the decoder sees.
    private func rendered(
        _ tok: KrillTokenizer, messages: [[String: String]],
        thinking: Bool, effort: String?
    ) -> String {
        let ids = tok.formatQwen35VLTokenIds(
            messages: messages, imagePadCount: 0, imageTokenId: 0,
            visionStartTokenId: 0, visionEndTokenId: 0,
            enableThinking: thinking, reasoningEffort: effort)
        return tok.decode(ids)
    }

    private let user = [["role": "user", "content": "hi"]]

    func testLowEffortInjectsItsInstructionAsASystemTurn() async throws {
        let tok = try await requireTokenizer()
        try XCTSkipUnless(tok.supportsReasoningEffort, "template has no reasoning_effort branch")
        let text = rendered(tok, messages: user, thinking: true, effort: "low")
        XCTAssertTrue(text.contains("Reasoning effort is set to low"), text)
        XCTAssertTrue(
            text.contains("<|im_start|>system\nReasoning effort is set to low"),
            "must be a SYSTEM turn, not folded into the user turn:\n\(text)")
    }

    /// The template defaults to `xhigh` when thinking is on and no level is
    /// given — passing nil must not mean "no instruction".
    func testDefaultsToXhighWhenThinking() async throws {
        let tok = try await requireTokenizer()
        try XCTSkipUnless(tok.supportsReasoningEffort, "template has no reasoning_effort branch")
        let text = rendered(tok, messages: user, thinking: true, effort: nil)
        XCTAssertTrue(text.contains("Reasoning effort is set to xhigh"), text)
    }

    /// `medium` is the template's silent level: its `elif` chain leaves
    /// `reasoning_instructions` empty rather than writing a third string.
    func testMediumInjectsNothing() async throws {
        let tok = try await requireTokenizer()
        try XCTSkipUnless(tok.supportsReasoningEffort, "template has no reasoning_effort branch")
        let text = rendered(tok, messages: user, thinking: true, effort: "medium")
        XCTAssertFalse(text.contains("Reasoning effort is set to"), text)
    }

    /// Thinking off means no instruction at any level — the template guards the
    /// whole block behind `if enable_thinking`.
    func testThinkingOffSuppressesTheInstruction() async throws {
        let tok = try await requireTokenizer()
        let text = rendered(tok, messages: user, thinking: false, effort: "low")
        XCTAssertFalse(text.contains("Reasoning effort"), text)
        XCTAssertTrue(text.contains("<think>\n\n</think>"), "closed scaffold when thinking is off")
    }

    /// An existing system message must be preserved, with the instruction
    /// prepended — matching the template's `reasoning_instructions + '\n\n' + content`.
    func testExistingSystemMessageIsPreservedNotReplaced() async throws {
        let tok = try await requireTokenizer()
        try XCTSkipUnless(tok.supportsReasoningEffort, "template has no reasoning_effort branch")
        let messages = [
            ["role": "system", "content": "You are a pirate."],
            ["role": "user", "content": "hi"],
        ]
        let text = rendered(tok, messages: messages, thinking: true, effort: "low")
        XCTAssertTrue(text.contains("You are a pirate."), "original system prompt must survive")
        XCTAssertTrue(text.contains("Reasoning effort is set to low"), text)
        guard let effortAt = text.range(of: "Reasoning effort is set to low"),
              let pirateAt = text.range(of: "You are a pirate.") else {
            return XCTFail("both strings must be present")
        }
        XCTAssertLessThan(effortAt.lowerBound, pirateAt.lowerBound, "instruction goes FIRST")
        // One system turn, not two.
        XCTAssertEqual(text.components(separatedBy: "<|im_start|>system").count - 1, 1)
    }

    /// An unrecognized level must not fabricate an instruction — the template
    /// would `raise_exception`, and the engine resolves junk to nil.
    func testUnknownEffortInjectsNothing() async throws {
        let tok = try await requireTokenizer()
        let text = rendered(tok, messages: user, thinking: true, effort: "turbo")
        XCTAssertFalse(text.contains("Reasoning effort is set to"), text)
    }
}
