import Foundation

/// Request promotion from read-only planning to a guarded execution posture.
/// Adaptive runs self-promote; ordinary plan runs ask through the shared question
/// gate. The model can never use this tool to reach `acceptAll`.
public struct RequestExecuteTool: Tool {
    public let name = "request_execute"
    public let isReadOnly = true
    public let description =
        "When planning is complete, request permission to start implementing it. "
        + "Pass a one-sentence summary of the intended work."
    public let parametersJSON = """
    {"type":"object","properties":{\
    "summary":{"type":"string","description":"One sentence naming what you will do if approved."}},\
    "required":["summary"]}
    """

    private let box: PermissionBox
    private let gate: any UserQuestionGate
    private let timeout: TimeInterval

    public init(
        permissionBox: PermissionBox,
        gate: any UserQuestionGate,
        timeout: TimeInterval = 300
    ) {
        self.box = permissionBox
        self.gate = gate
        self.timeout = timeout
    }

    /// Label-compatible convenience for callers that naturally spell the shared
    /// state as `box`.
    public init(box: PermissionBox, gate: any UserQuestionGate, timeout: TimeInterval = 300) {
        self.init(permissionBox: box, gate: gate, timeout: timeout)
    }

    public func run(argumentsJSON: String) async -> ToolResult {
        let current = box.effective
        guard current == .plan || current == .adaptive else {
            return ToolResult(
                content: "You are already implementing with \(current.label) permission. Just do the work.",
                isError: false)
        }

        if box.origin == .adaptive {
            return promote(to: .executePosture, adaptive: true)
        }

        let object = jsonObject(argumentsJSON) ?? [:]
        let rawSummary = object["summary"] as? String ?? ""
        let summary = rawSummary.trimmingCharacters(in: .whitespacesAndNewlines)
        let body = summary.isEmpty ? "The agent says its plan is ready." : summary
        let question = UserQuestion(
            header: "Approve plan",
            question: "Start implementing this plan?",
            body: body,
            options: [
                "Yes — apply edits automatically, ask before commands",
                "Yes — ask me before every edit and command",
                "No — keep planning",
            ])
        let answer = await askQuestion(question, gate: gate, timeout: timeout)
        guard !answer.declined, let index = answer.optionIndex else {
            return declined(question: question, answer: answer)
        }
        switch index {
        case 0:
            return promote(to: .acceptEdits, question: question, answer: answer)
        case 1:
            return promote(to: .ask, question: question, answer: answer)
        default:
            return declined(question: question, answer: answer)
        }
    }

    private func promote(
        to mode: PermissionMode,
        adaptive: Bool = false,
        question: UserQuestion? = nil,
        answer: UserAnswer? = nil
    ) -> ToolResult {
        guard box.promote(to: mode) else {
            return ToolResult(
                content: "You are already implementing. Continue with the work.",
                isError: false)
        }
        let prefix = adaptive ? "Adaptive planning is complete. " : "The user approved execution. "
        let display: ToolDisplay? = question.flatMap { q in
            answer.map { ToolDisplay.question(q, answer: $0) }
        }
        // The permission notice alone is not enough: without an explicit
        // instruction to proceed, models treat "get permission" as the task and
        // end the turn having changed nothing.
        return ToolResult(
            content: prefix + "Permission is now \(mode.label). "
                + "Ignore the earlier PLAN MODE instruction in your system prompt — it no longer applies. "
                + "Now execute the plan you just presented, step by step, starting with the first item. "
                + "Do not re-present the plan and do not call this tool again.",
            isError: false,
            display: display,
            effect: .permissionMode(mode))
    }

    private func declined(question: UserQuestion, answer: UserAnswer) -> ToolResult {
        let typed = answer.optionIndex == nil
            ? answer.text.trimmingCharacters(in: .whitespacesAndNewlines) : ""
        let feedback = typed.isEmpty ? "" : " The user said: \(typed)"
        return ToolResult(
            content: "Execution was not approved. Remain in read-only planning and respect the user's decision."
                + feedback,
            isError: false,
            display: .question(question, answer: answer))
    }
}
