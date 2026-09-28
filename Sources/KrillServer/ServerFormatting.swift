import Foundation

// Pure wire-format helpers shared by the server's streaming protocol paths.
// Module-internal visibility keeps them testable without exposing public API.

/// - Parameter logprobs: `choices[0].logprobs` for THIS chunk
///   (docs/LOGPROBS_PLAN.md §3.3) - either a ready-made JSON object (e.g.
///   from `logprobsChoiceJSON(content:)`) for a chunk carrying content, or
///   `NSNull()` for a chunk with no token (role/finish chunks), whenever the
///   REQUEST asked for logprobs. `nil` (the default, Swift's absence of a
///   value - distinct from the JSON null `NSNull()`) omits the key entirely,
///   leaving the emitted bytes IDENTICAL to before this field existed: every
///   caller that never requested logprobs never passes this, so the off path
///   is byte-for-byte unchanged.
func sseChunk(id: String, content: String?, finishReason: String?, logprobs: Any? = nil) -> String {
    var delta: [String: Any] = [:]
    if let content { delta["content"] = content }
    delta["role"] = "assistant"

    var choice: [String: Any] = ["index": 0, "delta": delta]
    if let logprobs { choice["logprobs"] = logprobs }
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
