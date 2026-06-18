import Foundation
import KrillTooling

/// OpenAI Responses API compatibility so OpenAI Codex (and any client using
/// `wire_api = "responses"`) works when pointed at Krill. Codex removed
/// Chat Completions support, so the Responses surface is the only way the
/// current Codex can talk to a local server.
///
/// Pure request→internal and internal→Responses mapping (Sendable,
/// unit-testable). Transport/streaming lives in ``Server``. Tool calling
/// reuses the model-agnostic `ToolCalling` sentinel path exactly like
/// ``AnthropicCompat``: Responses `function_call` / `function_call_output`
/// input items are flattened into the same `<tool_call>` / `<tool_response>`
/// text convention the chat path uses, so one extraction implementation
/// serves every surface.
internal enum ResponsesCompat {

    struct Parsed: Sendable {
        var messages: [[String: String]]
        var tools: [ServerToolSpec]
        var maxTokens: Int
        var sampling: ServerSamplingOptions
        var stream: Bool
        var model: String?
    }

    /// Flatten Responses content (string or part array) to plain text.
    /// Parts are `input_text` / `output_text` / `text` for text and
    /// `input_image` / `image` placeholders for images.
    private static func flattenContent(_ content: Any?) -> String {
        if let s = content as? String { return s }
        guard let parts = content as? [[String: Any]] else { return "" }
        var out: [String] = []
        for p in parts {
            switch p["type"] as? String {
            case "input_text", "output_text", "text", "summary_text":
                if let t = p["text"] as? String { out.append(t) }
            case "input_image", "image":
                out.append("[image]")
            default:
                if let t = p["text"] as? String { out.append(t) }
            }
        }
        return out.joined(separator: "\n")
    }

    /// Map a Responses `role` to the internal role vocabulary the prompt
    /// templates understand (`system` / `user` / `assistant`).
    private static func normalizeRole(_ role: String?) -> String {
        switch role {
        case "developer", "system": return "system"
        case "assistant": return "assistant"
        default: return "user"
        }
    }

    /// Normalize a `function_call.arguments` value into a valid JSON object
    /// string for the shared `<tool_call>` sentinel. Responses sends it as a
    /// JSON *string*, but a blank or non-JSON value would otherwise splice
    /// malformed JSON that the extractor silently drops; a client that sends
    /// an object instead of a string is also tolerated.
    private static func normalizeArguments(_ value: Any?) -> String {
        if let s = value as? String {
            let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { return "{}" }
            if let d = trimmed.data(using: .utf8),
               (try? JSONSerialization.jsonObject(with: d)) != nil {
                return trimmed
            }
            return "{}"
        }
        if let obj = value,
           JSONSerialization.isValidJSONObject(obj),
           let d = try? JSONSerialization.data(withJSONObject: obj),
           let s = String(data: d, encoding: .utf8) {
            return s
        }
        return "{}"
    }

    /// Render a `function_call_output.output` value as text for the shared
    /// `<tool_response>` sentinel. Responses usually sends a string, but a
    /// JSON object/array or a scalar must not be dropped: the model needs
    /// the tool result.
    private static func stringifyOutput(_ value: Any?) -> String {
        if let s = value as? String { return s }
        // Content-part array -> flatten its text; if that yields nothing it is
        // an arbitrary JSON array, so fall through to serialization below.
        if let parts = value as? [[String: Any]] {
            let flat = flattenContent(parts)
            if !flat.isEmpty { return flat }
        }
        if let obj = value, JSONSerialization.isValidJSONObject(obj),
           let d = try? JSONSerialization.data(withJSONObject: obj),
           let s = String(data: d, encoding: .utf8) {
            return s
        }
        if let n = value as? NSNumber { return n.stringValue }
        return ""
    }

    static func parse(_ json: [String: Any]) -> Parsed {
        var messages: [[String: String]] = []

        // `instructions` is the Responses analogue of a system prompt.
        if let instr = json["instructions"] as? String, !instr.isEmpty {
            messages.append(["role": "system", "content": instr])
        }

        // `input` may be a plain string (single user turn) or an array of items.
        if let s = json["input"] as? String, !s.isEmpty {
            messages.append(["role": "user", "content": s])
        } else if let items = json["input"] as? [[String: Any]] {
            for item in items {
                switch item["type"] as? String {
                case "message", nil:
                    let role = normalizeRole(item["role"] as? String)
                    messages.append(["role": role,
                                     "content": flattenContent(item["content"])])
                case "function_call":
                    // Assistant-issued tool call → shared sentinel.
                    let name = (item["name"] as? String) ?? ""
                    let args = normalizeArguments(item["arguments"])
                    messages.append(["role": "assistant",
                        "content": "<tool_call>{\"name\": \"\(name)\", \"arguments\": \(args)}</tool_call>"])
                case "function_call_output":
                    // Tool result returned to the model → shared sentinel.
                    let out = stringifyOutput(item["output"])
                    messages.append(["role": "user",
                        "content": "<tool_response>\(out)</tool_response>"])
                default:
                    break
                }
            }
        }

        // Responses tools are flat: {type:"function", name, description, parameters}
        // (no nested `function` wrapper like Chat Completions). Non-function
        // tool types (web_search, etc.) are skipped gracefully.
        var tools: [ServerToolSpec] = []
        for t in (json["tools"] as? [[String: Any]] ?? []) {
            guard (t["type"] as? String) == "function" || t["type"] == nil,
                  let name = t["name"] as? String else { continue }
            let schema = t["parameters"] ?? ["type": "object"]
            let sjson = (try? JSONSerialization.data(withJSONObject: schema))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
            tools.append(ServerToolSpec(
                name: name, description: (t["description"] as? String) ?? "",
                parametersJSON: sjson))
        }

        let temp = (json["temperature"] as? NSNumber)?.floatValue ?? 0.0
        let topP = (json["top_p"] as? NSNumber)?.floatValue ?? 1.0
        var sampling = ServerSamplingOptions(
            temperature: temp, topP: topP, topK: 0,
            repetitionPenalty: 1.0, seed: nil)
        sampling.minP = 0.0

        return Parsed(
            messages: messages, tools: tools,
            maxTokens: (json["max_output_tokens"] as? NSNumber)?.intValue ?? 1024,
            sampling: sampling,
            stream: (json["stream"] as? Bool) ?? false,
            model: json["model"] as? String)
    }

    // MARK: - Output item builders (shared by streaming + non-streaming)

    /// An assistant `message` output item carrying a single `output_text` part.
    static func messageItem(id: String, text: String, status: String) -> [String: Any] {
        [
            "type": "message",
            "id": id,
            "role": "assistant",
            "status": status,
            "content": [["type": "output_text", "text": text, "annotations": [Any]()]],
        ]
    }

    /// A `function_call` output item. `arguments` stays a JSON *string*
    /// (Responses, unlike Anthropic, does not decode it to an object).
    static func functionCallItem(id: String, callId: String,
                                 name: String, arguments: String,
                                 status: String) -> [String: Any] {
        [
            "type": "function_call",
            "id": id,
            "call_id": callId,
            "name": name,
            "arguments": arguments,
            "status": status,
        ]
    }

    /// The top-level `response` object (used by `response.created`,
    /// `response.completed`, and the non-streaming body).
    static func responseObject(
        id: String, model: String, status: String,
        output: [[String: Any]], createdAt: Int,
        inputTokens: Int, outputTokens: Int
    ) -> [String: Any] {
        [
            "id": id,
            "object": "response",
            "created_at": createdAt,
            "status": status,
            "error": NSNull(),
            "incomplete_details": NSNull(),
            "model": model,
            "output": output,
            "usage": [
                "input_tokens": inputTokens,
                "input_tokens_details": ["cached_tokens": 0],
                "output_tokens": outputTokens,
                "output_tokens_details": ["reasoning_tokens": 0],
                "total_tokens": inputTokens + outputTokens,
            ],
        ]
    }

    /// Assemble the `output` array for a completed turn.
    static func outputItems(
        messageId: String, text: String,
        toolCalls: [ToolCalling.ParsedToolCall]
    ) -> [[String: Any]] {
        var output: [[String: Any]] = []
        if toolCalls.isEmpty || !text.isEmpty {
            output.append(messageItem(id: messageId, text: text, status: "completed"))
        }
        for c in toolCalls {
            output.append(functionCallItem(
                id: "fc_\(UUID().uuidString.prefix(24))",
                callId: "call_\(UUID().uuidString.prefix(24))",
                name: c.name, arguments: c.argumentsJSON, status: "completed"))
        }
        return output
    }

    /// Non-streaming Responses body.
    static func response(
        id: String, model: String, text: String,
        toolCalls: [ToolCalling.ParsedToolCall],
        createdAt: Int, inputTokens: Int, outputTokens: Int
    ) -> [String: Any] {
        let output = outputItems(messageId: "msg_\(UUID().uuidString.prefix(24))",
                                 text: text, toolCalls: toolCalls)
        return responseObject(
            id: id, model: model, status: "completed",
            output: output, createdAt: createdAt,
            inputTokens: inputTokens, outputTokens: outputTokens)
    }
}
