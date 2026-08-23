import Foundation

/// Ask one clarifying question and return the answer to the model so the same
/// agent turn can continue.
public struct AskUserTool: Tool {
    public let name = "ask_user"
    public let isReadOnly = true
    public let description =
        "Ask the user one focused clarifying question when their answer materially affects the work. "
        + "Provide up to six concise options when useful; free text is also allowed."
    public let parametersJSON = """
    {"type":"object","properties":{\
    "header":{"type":"string","description":"Short heading for the question."},\
    "question":{"type":"string","description":"The focused question to ask."},\
    "body":{"type":"string","description":"Optional context or tradeoffs."},\
    "options":{"type":"array","description":"Up to six concise choices.","items":{"type":"string"}}},\
    "required":["question"]}
    """

    private let gate: any UserQuestionGate
    private let timeout: TimeInterval

    public init(gate: any UserQuestionGate, timeout: TimeInterval = 300) {
        self.gate = gate
        self.timeout = timeout
    }

    public func run(argumentsJSON: String) async -> ToolResult {
        guard let object = jsonObject(argumentsJSON) else {
            return ToolResult(content: "Error: ask_user requires a JSON object.", isError: true)
        }
        guard let text = firstString(in: object, keys: ["question", "prompt", "text"]) else {
            return ToolResult(content: "Error: ask_user requires a non-empty 'question'.", isError: true)
        }

        let question = UserQuestion(
            header: firstString(in: object, keys: ["header", "title"]) ?? "Question",
            question: text,
            body: firstString(in: object, keys: ["body", "description", "context"]) ?? "",
            options: parseOptions(object["options"]))
        let answer = await askQuestion(question, gate: gate, timeout: timeout)

        if answer.declined {
            return ToolResult(
                content: "The user declined to answer. Continue by making the safest reasonable assumption, "
                    + "state the assumption you made, and do not ask again.",
                isError: false,
                display: .question(question, answer: answer))
        }

        let selectedText: String
        if !answer.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            selectedText = answer.text
        } else if let i = answer.optionIndex, question.options.indices.contains(i) {
            selectedText = question.options[i]
        } else {
            selectedText = "(no text provided)"
        }
        let options = question.options.enumerated()
            .map { "\($0.offset + 1). \($0.element)" }
            .joined(separator: "\n")
        let optionBlock = options.isEmpty ? "" : "\nOptions presented:\n\(options)"
        return ToolResult(
            content: "User answered: \(selectedText)\(optionBlock)",
            isError: false,
            display: .question(question, answer: answer))
    }

    private func firstString(in object: [String: Any], keys: [String]) -> String? {
        for key in keys {
            guard let value = object[key] as? String else { continue }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        return nil
    }

    private func parseOptions(_ raw: Any?) -> [String] {
        var candidates: [String] = []
        if let strings = raw as? [String] {
            candidates = strings
        } else if let objects = raw as? [[String: Any]] {
            let keys = ["label", "text", "title", "value"]
            candidates = objects.compactMap { firstString(in: $0, keys: keys) }
        } else if let string = raw as? String {
            candidates = string.components(separatedBy: CharacterSet(charactersIn: ",;|\n"))
        }

        var seen = Set<String>()
        return candidates.compactMap { candidate in
            let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            let key = trimmed.lowercased()
            guard seen.insert(key).inserted else { return nil }
            return trimmed
        }.prefix(6).map { $0 }
    }
}
