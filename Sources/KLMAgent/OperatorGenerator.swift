import Foundation

/// One model turn from the loop's perspective: the full assistant
/// text (already accumulated by the generator), plus how many tokens
/// the generator counted for that turn.
public struct OperatorTurn: Equatable, Sendable {
    public let text: String
    public let tokenCount: Int

    public init(text: String, tokenCount: Int) {
        self.text = text
        self.tokenCount = tokenCount
    }
}

/// Pluggable "give me one assistant turn for these messages" surface.
///
/// The loop is engine-agnostic so it can be unit-tested with a fixture
/// generator that scripts a fixed sequence of replies. Sub-PR B wires
/// a `KLMEngine.InferenceEngine`-backed generator on top of this.
public protocol OperatorGenerator: Sendable {
    /// Run one full turn against `messages` and return the assistant
    /// text. The implementation owns its own EOS / max-tokens cutoff;
    /// the loop only enforces the cross-turn budget.
    func generate(messages: [[String: String]]) async throws -> OperatorTurn
}

/// Tool-call extraction strategy. The operator loop defaults to
/// `.hermes`, which is the wire format the pinned router model
/// (`qwen2.5-1.5b`) emits natively. Sub-PR B will switch to the
/// family-aware `KLMServer.ToolCalling.extractToolCalls` once the
/// loop is wired to a real `InferenceEngine`.
public enum OperatorToolFormat: Equatable, Sendable {
    case hermes
}

/// Minimal Hermes-style `<tool_call>{...}</tool_call>` extractor.
///
/// Returns the parsed calls in source order plus the assistant text
/// with all tool-call sentinels stripped (so the user-visible reply
/// never includes the raw JSON envelope).
///
/// Sub-PR A's KLMAgent module is self-contained - we keep a tight
/// re-implementation here so the loop can be unit-tested without
/// pulling in the `KLMServer` target. Sub-PR B replaces this with
/// the server's family-aware extractor.
internal enum HermesToolCallExtractor {
    static func extract(from text: String)
        -> (calls: [OperatorToolCall], cleanedText: String)
    {
        var calls: [OperatorToolCall] = []
        var cleaned = text

        let pairs: [(open: String, close: String)] = [
            ("<tool_call>", "</tool_call>"),
            ("<|tool_call|>", "<tool_call|>"),
        ]

        for (open, close) in pairs {
            while let s = cleaned.range(of: open) {
                let after = cleaned[s.upperBound...]
                guard let (json, jsonEnd) = firstJSONObject(in: after) else {
                    cleaned.removeSubrange(s.lowerBound ..< s.upperBound)
                    continue
                }
                if let call = parseCallJSON(json) { calls.append(call) }
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
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("{"), trimmed.contains("\"arguments\""),
               let call = parseCallJSON(trimmed), !call.name.isEmpty
            {
                calls.append(call)
                cleaned = ""
            }
        }

        return (calls, cleaned.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// First balanced `{...}` JSON object in `s`, string-literal aware.
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

    private static func parseCallJSON(_ json: String) -> OperatorToolCall? {
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
        return OperatorToolCall(name: name, argumentsJSON: argsString)
    }
}
