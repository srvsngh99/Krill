import Foundation

/// Model-agnostic tool/function calling (WS-D D1).
///
/// KrillLM does not depend on any single model's native tool-call template.
/// Instead it injects a compact instruction + the tool JSON-schemas as a
/// system turn and asks the model to emit a sentinel-wrapped call:
///
///     <tool_call>{"name": "...", "arguments": { ... }}</tool_call>
///
/// This convention (Hermes/Qwen-style) works acceptably across the Llama /
/// Qwen / Mistral / Gemma / Phi families KrillLM serves. Parsing is tolerant
/// of three shapes: `<tool_call>…</tool_call>`, the legacy Gemma
/// `<|tool_call|>…<tool_call|>`, and a bare leading JSON object that has
/// both `name` and `arguments`.
///
/// Pure and Sendable so it is unit-testable without a model or a channel.
internal enum ToolCalling {

    struct ParsedToolCall: Equatable, Sendable {
        let name: String
        /// Arguments as a JSON *string* (OpenAI wants a string; Ollama wants
        /// the decoded object — callers convert as needed).
        let argumentsJSON: String
    }

    // MARK: - Prompt injection

    static func toolSystemPrompt(_ tools: [ServerToolSpec]) -> String {
        var lines = [
            "You can call tools. The available tools are listed as JSON schemas:",
            "",
        ]
        for t in tools {
            lines.append(
                "{\"name\": \"\(t.name)\", \"description\": \"\(escapeForPrompt(t.description))\", \"parameters\": \(t.parametersJSON)}")
        }
        lines.append("")
        lines.append("To call a tool, output ONLY this exact line and nothing else — no explanation, no code fences, do not repeat the schema:")
        lines.append("<tool_call>{\"name\": \"<tool-name>\", \"arguments\": {<the actual argument values>}}</tool_call>")
        lines.append("`arguments` must be the concrete values for this request, not the schema.")
        lines.append("Use multiple <tool_call> lines to call multiple tools.")
        lines.append("If no tool is needed, just answer the user normally with no <tool_call>.")
        return lines.joined(separator: "\n")
    }

    /// Prepend the tool instruction. If a leading system message exists, the
    /// tool block is appended to it so a single system turn is preserved.
    static func injectToolSystem(
        into messages: [[String: String]],
        tools: [ServerToolSpec]
    ) -> [[String: String]] {
        guard !tools.isEmpty else { return messages }
        let block = toolSystemPrompt(tools)
        var out = messages
        if let first = out.first, first["role"] == "system" {
            out[0]["content"] = (first["content"] ?? "") + "\n\n" + block
        } else {
            out.insert(["role": "system", "content": block], at: 0)
        }
        return out
    }

    // MARK: - Extraction

    private static let pairs: [(open: String, close: String)] = [
        ("<tool_call>", "</tool_call>"),
        ("<|tool_call|>", "<tool_call|>"),
    ]

    static func extractToolCalls(from text: String)
        -> (calls: [ParsedToolCall], cleanedText: String)
    {
        var calls: [ParsedToolCall] = []
        var cleaned = text

        // For each opening sentinel, extract the first balanced JSON object
        // after it — tolerating a missing close tag, surrounding backticks,
        // and trailing punctuation (small models routinely do all three).
        for (open, close) in pairs {
            while let s = cleaned.range(of: open) {
                let after = cleaned[s.upperBound...]
                guard let (json, jsonEnd) = Self.firstJSONObject(in: after) else {
                    // No JSON after the marker — drop the bare marker so it
                    // doesn't leak into user-visible content.
                    cleaned.removeSubrange(s.lowerBound ..< s.upperBound)
                    continue
                }
                if let c = parseCallJSON(json) { calls.append(c) }
                // Remove from the open marker through the JSON end, plus an
                // optional matching close tag / trailing junk on that span.
                var removeEnd = jsonEnd
                let tail = cleaned[jsonEnd...]
                if let cr = tail.range(of: close),
                   cr.lowerBound == tail.startIndex
                       || tail[tail.startIndex ..< cr.lowerBound]
                           .allSatisfy({ " `;\n\t".contains($0) })
                {
                    removeEnd = cr.upperBound
                }
                cleaned.removeSubrange(s.lowerBound ..< removeEnd)
            }
        }

        if calls.isEmpty {
            // Fenced ```json block whose object has BOTH `name` and
            // `arguments`. Requiring `arguments` avoids mistaking an echoed
            // tool *schema* (which has `parameters`/`properties`, not
            // `arguments`) for an actual call.
            if let f = cleaned.range(of: "```"),
               let e = cleaned.range(of: "```", range: f.upperBound ..< cleaned.endIndex)
            {
                var body = String(cleaned[f.upperBound ..< e.lowerBound])
                if body.hasPrefix("json") { body.removeFirst(4) }
                body = body.trimmingCharacters(in: .whitespacesAndNewlines)
                if body.contains("\"arguments\""), let c = parseCallJSON(body) {
                    calls.append(c)
                    cleaned.removeSubrange(f.lowerBound ..< e.upperBound)
                }
            }
        }

        if calls.isEmpty {
            // Bare JSON object with name + arguments (no sentinel).
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("{"), trimmed.contains("\"arguments\""),
               let c = parseCallJSON(trimmed), !c.name.isEmpty {
                calls.append(c)
                cleaned = ""
            }
        }

        return (calls, cleaned.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Find the first balanced `{...}` JSON object in `s` (string-literal
    /// aware so braces inside quotes don't miscount). Returns the object
    /// substring and the index just past its closing brace.
    private static func firstJSONObject(in s: Substring)
        -> (json: String, end: Substring.Index)?
    {
        guard let start = s.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inStr = false
        var esc = false
        var i = start
        while i < s.endIndex {
            let ch = s[i]
            if inStr {
                if esc { esc = false }
                else if ch == "\\" { esc = true }
                else if ch == "\"" { inStr = false }
            } else {
                if ch == "\"" { inStr = true }
                else if ch == "{" { depth += 1 }
                else if ch == "}" {
                    depth -= 1
                    if depth == 0 {
                        let end = s.index(after: i)
                        return (String(s[start ..< end]), end)
                    }
                }
            }
            i = s.index(after: i)
        }
        return nil
    }

    private static func parseCallJSON(_ json: String) -> ParsedToolCall? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String, !name.isEmpty
        else { return nil }

        let argsString: String
        if let a = obj["arguments"] {
            if let s = a as? String {
                argsString = s
            } else if let d = try? JSONSerialization.data(withJSONObject: a) {
                argsString = String(data: d, encoding: .utf8) ?? "{}"
            } else {
                argsString = "{}"
            }
        } else {
            argsString = "{}"
        }
        return ParsedToolCall(name: name, argumentsJSON: argsString)
    }

    // MARK: - Response shaping

    /// OpenAI `message.tool_calls` array (arguments as a JSON string).
    static func openAIToolCalls(_ calls: [ParsedToolCall]) -> [[String: Any]] {
        calls.enumerated().map { i, c in
            [
                "id": "call_\(randomId())\(i)",
                "type": "function",
                "function": ["name": c.name, "arguments": c.argumentsJSON],
            ]
        }
    }

    /// Ollama `message.tool_calls` array (arguments as a decoded object).
    static func ollamaToolCalls(_ calls: [ParsedToolCall]) -> [[String: Any]] {
        calls.map { c in
            let argsObj = (try? JSONSerialization.jsonObject(
                with: Data(c.argumentsJSON.utf8))) ?? [String: Any]()
            return ["function": ["name": c.name, "arguments": argsObj]]
        }
    }

    private static func randomId() -> String {
        String(UUID().uuidString.prefix(8)).lowercased()
    }

    private static func escapeForPrompt(_ s: String) -> String {
        s.replacingOccurrences(of: "\"", with: "'")
            .replacingOccurrences(of: "\n", with: " ")
    }
}
