import Foundation
import KrillTooling

/// Anthropic Messages API compatibility (WS-F / T2-9) so Claude Code and
/// the Anthropic SDK work when pointed at Krill via `ANTHROPIC_BASE_URL`.
///
/// Pure request→internal and internal→Anthropic mapping (Sendable,
/// unit-testable). Transport/streaming lives in ``Server``. Tool calling
/// reuses the model-agnostic `ToolCalling` sentinel path: Anthropic
/// `tool_use`/`tool_result` blocks are flattened into the same
/// `<tool_call>`/`<tool_response>` text convention the chat path uses, so
/// one extraction implementation serves every surface.
internal enum AnthropicCompat {

    struct Parsed: Sendable {
        var messages: [[String: String]]
        var tools: [ServerToolSpec]
        var maxTokens: Int
        var sampling: ServerSamplingOptions
        var stream: Bool
        var model: String?
        var thinking: Bool
    }

    /// Flatten Anthropic content (string or block array) to plain text,
    /// turning `tool_use`/`tool_result` blocks into the shared sentinels.
    private static func flatten(_ content: Any?) -> String {
        if let s = content as? String { return s }
        guard let blocks = content as? [[String: Any]] else { return "" }
        var parts: [String] = []
        for b in blocks {
            switch b["type"] as? String {
            case "text":
                if let t = b["text"] as? String { parts.append(t) }
            case "tool_use":
                let name = (b["name"] as? String) ?? ""
                let input = b["input"] ?? [String: Any]()
                let args = (try? JSONSerialization.data(withJSONObject: input))
                    .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                parts.append("<tool_call>{\"name\": \"\(name)\", \"arguments\": \(args)}</tool_call>")
            case "tool_result":
                let c = b["content"]
                let txt = (c as? String) ?? flatten(c)
                parts.append("<tool_response>\(txt)</tool_response>")
            case "image":
                parts.append("[image]")
            default:
                break
            }
        }
        return parts.joined(separator: "\n")
    }

    static func parse(_ json: [String: Any]) -> Parsed {
        var messages: [[String: String]] = []

        // `system` may be a string or an array of text blocks.
        if let sys = json["system"] as? String, !sys.isEmpty {
            messages.append(["role": "system", "content": sys])
        } else if let sysBlocks = json["system"] as? [[String: Any]] {
            let t = sysBlocks.compactMap { $0["text"] as? String }
                .joined(separator: "\n")
            if !t.isEmpty { messages.append(["role": "system", "content": t]) }
        }

        for m in (json["messages"] as? [[String: Any]] ?? []) {
            let role = (m["role"] as? String) ?? "user"
            messages.append(["role": role, "content": flatten(m["content"])])
        }

        var tools: [ServerToolSpec] = []
        for t in (json["tools"] as? [[String: Any]] ?? []) {
            guard let name = t["name"] as? String else { continue }
            let schema = t["input_schema"] ?? ["type": "object"]
            let sjson = (try? JSONSerialization.data(withJSONObject: schema))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
            tools.append(ServerToolSpec(
                name: name, description: (t["description"] as? String) ?? "",
                parametersJSON: sjson))
        }

        let temp = (json["temperature"] as? NSNumber)?.floatValue ?? 0.0
        let topP = (json["top_p"] as? NSNumber)?.floatValue ?? 1.0
        let topK = (json["top_k"] as? NSNumber)?.intValue ?? 0
        var sampling = ServerSamplingOptions(
            temperature: temp, topP: topP, topK: topK,
            repetitionPenalty: 1.0, seed: nil)
        sampling.minP = 0.0

        let thinking = (json["thinking"] as? [String: Any])
            .map { ($0["type"] as? String) == "enabled" } ?? false

        return Parsed(
            messages: messages, tools: tools,
            maxTokens: (json["max_tokens"] as? NSNumber)?.intValue ?? 1024,
            sampling: sampling,
            stream: (json["stream"] as? Bool) ?? false,
            model: json["model"] as? String,
            thinking: thinking)
    }

    /// Non-streaming Anthropic `message` response.
    static func response(
        model: String, text: String,
        toolCalls: [ToolCalling.ParsedToolCall],
        thinking: String?,
        inputTokens: Int, outputTokens: Int
    ) -> [String: Any] {
        var content: [[String: Any]] = []
        if let thinking, !thinking.isEmpty {
            content.append(["type": "thinking", "thinking": thinking])
        }
        if toolCalls.isEmpty {
            content.append(["type": "text", "text": text])
        } else {
            for c in toolCalls {
                let input = (try? JSONSerialization.jsonObject(
                    with: Data(c.argumentsJSON.utf8))) ?? [String: Any]()
                content.append([
                    "type": "tool_use",
                    "id": "toolu_\(UUID().uuidString.prefix(8))",
                    "name": c.name, "input": input,
                ])
            }
        }
        return [
            "id": "msg_\(UUID().uuidString.prefix(12))",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": toolCalls.isEmpty ? "end_turn" : "tool_use",
            "stop_sequence": NSNull(),
            "usage": ["input_tokens": inputTokens, "output_tokens": outputTokens],
        ]
    }
}
