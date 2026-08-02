import Foundation

// Pure wire-format helpers shared by the server's streaming protocol paths.
// Module-internal visibility keeps them testable without exposing public API.

func sseChunk(id: String, content: String?, finishReason: String?) -> String {
    var delta: [String: Any] = [:]
    if let content { delta["content"] = content }
    delta["role"] = "assistant"

    var choice: [String: Any] = ["index": 0, "delta": delta]
    if let reason = finishReason {
        choice["finish_reason"] = reason
        choice["delta"] = [String: Any]()
    }

    let payload: [String: Any] = [
        "id": id,
        "object": "chat.completion.chunk",
        "created": Int(Date().timeIntervalSince1970),
        "choices": [choice]
    ]

    guard let data = try? JSONSerialization.data(withJSONObject: payload),
          let json = String(data: data, encoding: .utf8) else {
        return ""
    }
    return "data: \(json)\n\n"
}

/// OpenAI `stream_options.include_usage` terminal chunk: a `chat.completion.chunk`
/// with an empty `choices` array carrying the run's token `usage`, emitted just
/// before `data: [DONE]`. Lets streaming harnesses (opencode, the OpenAI SDK)
/// populate their context/token meter, which otherwise reads zero.
func sseUsageChunk(id: String, promptTokens: Int, completionTokens: Int) -> String {
    let payload: [String: Any] = [
        "id": id,
        "object": "chat.completion.chunk",
        "created": Int(Date().timeIntervalSince1970),
        "choices": [Any](),
        "usage": [
            "prompt_tokens": promptTokens,
            "completion_tokens": completionTokens,
            "total_tokens": promptTokens + completionTokens,
        ],
    ]
    guard let data = try? JSONSerialization.data(withJSONObject: payload),
          let json = String(data: data, encoding: .utf8) else {
        return ""
    }
    return "data: \(json)\n\n"
}

/// Escape a string for safe embedding inside a JSON string value.
/// Handles backslash, double-quote, and control characters.
func escapeJSON(_ s: String) -> String {
    var result = ""
    result.reserveCapacity(s.utf8.count)
    for c in s {
        switch c {
        case "\"": result += "\\\""
        case "\\": result += "\\\\"
        case "\n": result += "\\n"
        case "\r": result += "\\r"
        case "\t": result += "\\t"
        default:
            if c.asciiValue != nil && c.asciiValue! < 0x20 {
                result += String(format: "\\u%04x", c.asciiValue!)
            } else {
                result.append(c)
            }
        }
    }
    return result
}
