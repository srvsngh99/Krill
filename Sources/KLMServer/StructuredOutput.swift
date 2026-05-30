import Foundation
import KLMEngine

/// Structured output (WS-D D2 / T1-1): `format:"json"` / JSON-schema and
/// the OpenAI `response_format` equivalent.
///
/// KrillLM uses guided prompting + a tolerant post-extraction pass rather
/// than true grammar-constrained decoding (the latter is the highest-
/// uncertainty item in the parity plan §8 and is tracked as a follow-up
/// because it must not regress the prefix/int8-KV decode path). In
/// practice instruction-tuned models reliably emit valid JSON when the
/// system turn demands it; the extractor then strips any prose/fences so
/// the client receives parseable JSON.
internal enum StructuredOutput {

    /// Map a server-internal `ResponseFormat` to the engine's
    /// grammar-decoding `OutputFormat`. `.json` requests the any-JSON-value
    /// mask (Stage A); `.schema` requests the schema-constrained mask (Stage
    /// B) so the decoder is held to the schema's structure, not just JSON
    /// well-formedness. The injected system prompt + post-extraction `coerce`
    /// remain as a backstop (and cover schema features the compiler relaxes).
    static func engineFormat(for format: ResponseFormat?) -> OutputFormat? {
        guard let format else { return nil }
        switch format {
        case .json: return .json
        case .schema(let schema): return .jsonSchema(schema)
        }
    }

    static func systemPrompt(for format: ResponseFormat) -> String {
        switch format {
        case .json:
            return "You must respond with a single valid JSON value and nothing else - no prose, no markdown, no code fences."
        case .schema(let schema):
            return """
            You must respond with a single valid JSON value and nothing else \
            - no prose, no markdown, no code fences. The JSON must conform to \
            this JSON Schema:
            \(schema)
            """
        }
    }

    static func injectFormatSystem(
        into messages: [[String: String]],
        format: ResponseFormat?
    ) -> [[String: String]] {
        guard let format else { return messages }
        let block = systemPrompt(for: format)
        var out = messages
        if let first = out.first, first["role"] == "system" {
            out[0]["content"] = (first["content"] ?? "") + "\n\n" + block
        } else {
            out.insert(["role": "system", "content": block], at: 0)
        }
        return out
    }

    /// Best-effort: return the first balanced JSON value (object or array)
    /// in `text`, re-serialized compactly. `nil` if no valid JSON is found.
    static func extractJSON(from text: String) -> String? {
        guard let (raw, _) = firstJSONValue(in: Substring(text)) else { return nil }
        guard let data = raw.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(
                  with: data, options: [.fragmentsAllowed]),
              let out = try? JSONSerialization.data(
                  withJSONObject: obj, options: [.sortedKeys]),
              let s = String(data: out, encoding: .utf8)
        else { return nil }
        return s
    }

    /// Coerce model output for a structured request: extracted JSON if any,
    /// otherwise the original text (so a refusal/error is still visible).
    static func coerce(_ text: String, format: ResponseFormat?) -> String {
        guard format != nil else { return text }
        return extractJSON(from: text) ?? text
    }

    /// First balanced `{...}` or `[...]` substring, string-literal aware.
    private static func firstJSONValue(in s: Substring)
        -> (json: String, end: Substring.Index)?
    {
        let opens: Set<Character> = ["{", "["]
        guard let start = s.firstIndex(where: { opens.contains($0) }) else { return nil }
        let open = s[start]
        let close: Character = open == "{" ? "}" : "]"
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
            } else if ch == "\"" {
                inStr = true
            } else if ch == open {
                depth += 1
            } else if ch == close {
                depth -= 1
                if depth == 0 {
                    let end = s.index(after: i)
                    return (String(s[start ..< end]), end)
                }
            }
            i = s.index(after: i)
        }
        return nil
    }
}
