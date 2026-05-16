import Foundation
import NIOCore
import NIOHTTP1
import KLMSampler

internal enum ServerLimits {
    static let maxBodySize = 10 * 1024 * 1024
}

internal struct ServerSamplingOptions: Equatable, Sendable {
    let temperature: Float
    let topP: Float
    let topK: Int
    let repetitionPenalty: Float
    let seed: UInt64?
    var minP: Float = 0.0

    var samplingParams: SamplingParams {
        SamplingParams(
            temperature: temperature,
            topP: topP,
            topK: topK,
            repetitionPenalty: repetitionPenalty,
            seed: seed,
            minP: minP
        )
    }
}

/// Media payload extracted from a request, prior to base64 decoding.
/// Each string is either a raw base64 body or a `data:...;base64,...` URL.
internal struct ServerMediaPayload: Equatable, Sendable {
    /// Base64-encoded image strings (Ollama allows multiple, we cap at 1).
    var images: [String] = []
    /// Base64-encoded audio (single).
    var audio: String? = nil
    /// Audio container format hint (e.g. "wav", "mp3"). Defaults to "wav".
    var audioFormat: String? = nil

    var isEmpty: Bool { images.isEmpty && audio == nil }
}

/// A normalized tool/function definition (Sendable). `parametersJSON` is
/// the raw JSON-schema object serialized to a string so it can flow through
/// Sendable boundaries and be embedded verbatim into the tool prompt.
internal struct ServerToolSpec: Equatable, Sendable {
    let name: String
    let description: String
    let parametersJSON: String
}

/// Structured-output request (WS-D D2 / T1-1). `.json` = free-form JSON;
/// `.schema` carries a JSON-schema string the output must conform to.
internal enum ResponseFormat: Equatable, Sendable {
    case json
    case schema(String)
}

internal struct ServerChatRequest: Equatable, Sendable {
    let messages: [[String: String]]
    let stream: Bool
    let maxTokens: Int
    let sampling: ServerSamplingOptions
    let requestedModel: String?
    var media: ServerMediaPayload = ServerMediaPayload()
    var tools: [ServerToolSpec] = []
    var responseFormat: ResponseFormat? = nil
    /// `keep_alive` in seconds (WS-E). nil=default, <0=pin, 0=evict-after.
    var keepAlive: Int? = nil
    /// `num_ctx` prompt-token cap (WS-D D4). nil=model max.
    var contextLimit: Int? = nil
}

internal struct ServerCompletionRequest: Equatable, Sendable {
    let prompt: String
    let maxTokens: Int
    let sampling: ServerSamplingOptions
    let requestedModel: String?
}

internal struct ServerGenerateRequest: Equatable, Sendable {
    let prompt: String
    let system: String?
    let stream: Bool
    let maxTokens: Int
    let sampling: ServerSamplingOptions
    let requestedModel: String?
    var media: ServerMediaPayload = ServerMediaPayload()
    var responseFormat: ResponseFormat? = nil
    var keepAlive: Int? = nil
    var contextLimit: Int? = nil
}

internal enum ServerRequestError: Error, Equatable, Sendable {
    case missingField(String)
    case invalidType(field: String, expected: String)
    case invalidValue(field: String, reason: String)
    case unsupportedField(String)

    var message: String {
        switch self {
        case .missingField(let field):
            return "Missing required field '\(field)'"
        case .invalidType(let field, let expected):
            return "Field '\(field)' must be \(expected)"
        case .invalidValue(let field, let reason):
            return "Field '\(field)' is invalid: \(reason)"
        case .unsupportedField(let field):
            return "Field '\(field)' is not supported by this endpoint"
        }
    }
}

internal enum ServerParsing {
    private static let defaultOpenAIMaxTokens = 512
    private static let defaultOpenAICompletionMaxTokens = 256
    private static let defaultOllamaMaxTokens = 2048

    private static let unsupportedOpenAIChatFields: Set<String> = [
        "parallel_tool_calls",
        "functions", "function_call",
        "logprobs", "top_logprobs",
        "stop", "logit_bias",
        "stream_options"
    ]

    private static let unsupportedOpenAICompletionFields: Set<String> = [
        "suffix", "best_of", "logprobs", "echo",
        "stop", "frequency_penalty", "presence_penalty", "logit_bias"
    ]

    private static let unsupportedOllamaChatFields: Set<String> = []

    private static let unsupportedOllamaGenerateFields: Set<String> = [
        "suffix", "context", "template"
    ]

    static func jsonObject(from buffer: ByteBuffer) -> [String: Any]? {
        var buffer = buffer
        guard let bytes = buffer.readBytes(length: buffer.readableBytes) else {
            return nil
        }
        return try? JSONSerialization.jsonObject(with: Data(bytes)) as? [String: Any]
    }

    static func structuredMessages(from messages: [[String: Any]]) -> [[String: String]] {
        messages.compactMap { message in
            guard let role = message["role"] as? String,
                  let content = message["content"] as? String else {
                return nil
            }
            return ["role": role, "content": content]
        }
    }

    /// Parse OpenAI/Ollama `tools: [{type:"function", function:{name,
    /// description, parameters}}]`. Tolerant: bare `{name, description,
    /// parameters}` entries are also accepted.
    static func parseTools(from json: [String: Any]) throws -> [ServerToolSpec] {
        guard let raw = json["tools"] else { return [] }
        guard let arr = raw as? [[String: Any]] else {
            throw ServerRequestError.invalidType(field: "tools", expected: "an array of tool objects")
        }
        var specs: [ServerToolSpec] = []
        for (i, t) in arr.enumerated() {
            let fn = (t["function"] as? [String: Any]) ?? t
            guard let name = fn["name"] as? String, !name.isEmpty else {
                throw ServerRequestError.invalidValue(
                    field: "tools[\(i)].function.name", reason: "is required")
            }
            let desc = (fn["description"] as? String) ?? ""
            let params = fn["parameters"] ?? ["type": "object", "properties": [String: Any]()]
            let paramsJSON = (try? JSONSerialization.data(withJSONObject: params))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
            specs.append(ServerToolSpec(name: name, description: desc, parametersJSON: paramsJSON))
        }
        return specs
    }

    /// Rewrite tool-related conversation turns into plain text so the
    /// existing `[[String:String]]` prompt path round-trips multi-turn tool
    /// loops without every chat template needing a native `tool` role:
    ///  - assistant `tool_calls` (often `content:null`) -> assistant text
    ///    with `<tool_call>{...}</tool_call>` lines.
    ///  - role `tool` (a tool result) -> user text wrapped in
    ///    `<tool_response>...</tool_response>`.
    private static func normalizeToolTurns(in json: [String: Any]) -> [String: Any] {
        guard let msgs = json["messages"] as? [[String: Any]] else { return json }
        var out: [[String: Any]] = []
        for m in msgs {
            let role = m["role"] as? String ?? "user"
            if role == "tool" {
                let result = (m["content"] as? String) ?? ""
                let name = (m["name"] as? String) ?? ""
                out.append([
                    "role": "user",
                    "content": "<tool_response>\(name.isEmpty ? "" : "name=\(name) ")\(result)</tool_response>"
                ])
                continue
            }
            if role == "assistant", let calls = m["tool_calls"] as? [[String: Any]] {
                var parts: [String] = []
                if let c = m["content"] as? String, !c.isEmpty { parts.append(c) }
                for call in calls {
                    let fn = (call["function"] as? [String: Any]) ?? [:]
                    let name = (fn["name"] as? String) ?? ""
                    let argsAny = fn["arguments"] ?? [String: Any]()
                    let argsStr: String
                    if let s = argsAny as? String {
                        argsStr = s
                    } else if let d = try? JSONSerialization.data(withJSONObject: argsAny) {
                        argsStr = String(data: d, encoding: .utf8) ?? "{}"
                    } else { argsStr = "{}" }
                    parts.append("<tool_call>{\"name\": \"\(name)\", \"arguments\": \(argsStr)}</tool_call>")
                }
                out.append(["role": "assistant", "content": parts.joined(separator: "\n")])
                continue
            }
            out.append(m)
        }
        var copy = json
        copy["messages"] = out
        return copy
    }

    /// Ollama `format`: the string `"json"` or a JSON-schema object.
    static func parseOllamaFormat(_ raw: Any?) -> ResponseFormat? {
        guard let raw else { return nil }
        if let s = raw as? String {
            return s.lowercased() == "json" ? .json : nil
        }
        if let obj = raw as? [String: Any],
           let d = try? JSONSerialization.data(withJSONObject: obj),
           let s = String(data: d, encoding: .utf8) {
            return .schema(s)
        }
        return nil
    }

    /// OpenAI `response_format`: `{type:"json_object"}`,
    /// `{type:"json_schema", json_schema:{schema:{…}}}`, or `{type:"text"}`.
    static func parseOpenAIResponseFormat(_ raw: Any?) -> ResponseFormat? {
        guard let obj = raw as? [String: Any],
              let type = obj["type"] as? String else { return nil }
        switch type {
        case "json_object":
            return .json
        case "json_schema":
            if let js = obj["json_schema"] as? [String: Any] {
                let schema = (js["schema"] as? [String: Any]) ?? js
                if let d = try? JSONSerialization.data(withJSONObject: schema),
                   let s = String(data: d, encoding: .utf8) {
                    return .schema(s)
                }
            }
            return .json
        default:
            return nil
        }
    }

    static func openAIChatRequest(from json: [String: Any]) throws -> ServerChatRequest {
        try rejectUnsupportedFields(in: json, fields: unsupportedOpenAIChatFields)
        let tools = try parseTools(from: json)
        let extracted = try openAIMessages(from: normalizeToolTurns(in: json))
        return ServerChatRequest(
            messages: extracted.messages,
            stream: try boolValue(json["stream"], field: "stream") ?? false,
            maxTokens: try requiredTokenLimit(
                from: json,
                fields: ["max_tokens", "max_completion_tokens"],
                defaultValue: defaultOpenAIMaxTokens
            ),
            sampling: try openAISamplingOptions(from: json),
            requestedModel: try optionalString(json["model"], field: "model"),
            media: extracted.media,
            tools: tools,
            responseFormat: parseOpenAIResponseFormat(json["response_format"]),
            keepAlive: KeepAliveParse.seconds(from: json["keep_alive"])
        )
    }

    static func openAICompletionRequest(from json: [String: Any]) throws -> ServerCompletionRequest {
        try rejectUnsupportedFields(in: json, fields: unsupportedOpenAICompletionFields)
        if try boolValue(json["stream"], field: "stream") == true {
            throw ServerRequestError.unsupportedField("stream")
        }
        return ServerCompletionRequest(
            prompt: try stringValue(json["prompt"], field: "prompt"),
            maxTokens: try requiredTokenLimit(
                from: json,
                fields: ["max_tokens", "max_completion_tokens"],
                defaultValue: defaultOpenAICompletionMaxTokens
            ),
            sampling: try openAISamplingOptions(from: json),
            requestedModel: try optionalString(json["model"], field: "model")
        )
    }

    static func openAISamplingOptions(from json: [String: Any]) throws -> ServerSamplingOptions {
        var opts = ServerSamplingOptions(
            temperature: try nonNegativeFloat(json["temperature"], field: "temperature", defaultValue: 0.0),
            topP: try topPValue(json["top_p"], field: "top_p"),
            topK: try nonNegativeInt(json["top_k"], field: "top_k", defaultValue: 0),
            repetitionPenalty: 1.0,
            seed: try optionalUInt64(json["seed"], field: "seed")
        )
        opts.minP = try nonNegativeFloat(json["min_p"], field: "min_p", defaultValue: 0.0)
        return opts
    }

    static func ollamaChatRequest(from json: [String: Any]) throws -> ServerChatRequest {
        try rejectUnsupportedFields(in: json, fields: unsupportedOllamaChatFields)
        let tools = try parseTools(from: json)
        let extracted = try ollamaMessages(from: normalizeToolTurns(in: json))
        return ServerChatRequest(
            messages: extracted.messages,
            stream: try boolValue(json["stream"], field: "stream") ?? true,
            maxTokens: try ollamaTokenLimit(from: json),
            sampling: try ollamaSamplingOptions(from: json),
            requestedModel: try optionalString(json["model"], field: "model"),
            media: extracted.media,
            tools: tools,
            responseFormat: parseOllamaFormat(json["format"]),
            keepAlive: KeepAliveParse.seconds(from: json["keep_alive"]),
            contextLimit: (try? optionsObject(from: json))?["num_ctx"] as? Int ?? json["num_ctx"] as? Int
        )
    }

    static func ollamaGenerateRequest(from json: [String: Any]) throws -> ServerGenerateRequest {
        try rejectUnsupportedFields(in: json, fields: unsupportedOllamaGenerateFields)
        if try boolValue(json["raw"], field: "raw") == true {
            throw ServerRequestError.unsupportedField("raw")
        }
        var media = ServerMediaPayload()
        if let raw = json["images"] {
            guard let images = ServerMultimodal.coerceStringArray(raw) else {
                throw ServerRequestError.invalidType(field: "images", expected: "an array of base64 strings")
            }
            media.images = images
        }
        if let raw = json["audio"] {
            if let s = raw as? String {
                media.audio = s
            } else if let arr = ServerMultimodal.coerceStringArray(raw), let first = arr.first {
                if arr.count > 1 {
                    throw ServerRequestError.invalidValue(
                        field: "audio", reason: "only one audio clip per request is supported"
                    )
                }
                media.audio = first
            } else {
                throw ServerRequestError.invalidType(field: "audio", expected: "a base64 string or single-element array")
            }
        }
        if let fmt = try optionalString(json["audio_format"], field: "audio_format") {
            media.audioFormat = fmt
        }
        return ServerGenerateRequest(
            prompt: try stringValue(json["prompt"], field: "prompt"),
            system: try optionalString(json["system"], field: "system"),
            stream: try boolValue(json["stream"], field: "stream") ?? true,
            maxTokens: try ollamaTokenLimit(from: json),
            sampling: try ollamaSamplingOptions(from: json),
            requestedModel: try optionalString(json["model"], field: "model"),
            media: media,
            responseFormat: parseOllamaFormat(json["format"]),
            keepAlive: KeepAliveParse.seconds(from: json["keep_alive"]),
            contextLimit: (try? optionsObject(from: json))?["num_ctx"] as? Int ?? json["num_ctx"] as? Int
        )
    }

    static func ollamaSamplingOptions(from json: [String: Any]) throws -> ServerSamplingOptions {
        let options = try optionsObject(from: json)
        let result = ServerSamplingOptions(
            temperature: try nonNegativeFloat(
                options["temperature"] ?? json["temperature"],
                field: "temperature",
                defaultValue: 0.0
            ),
            topP: try topPValue(options["top_p"] ?? json["top_p"], field: "top_p"),
            topK: try nonNegativeInt(options["top_k"] ?? json["top_k"], field: "top_k", defaultValue: 0),
            repetitionPenalty: try positiveFloat(
                options["repeat_penalty"] ?? json["repeat_penalty"],
                field: "repeat_penalty",
                defaultValue: 1.0
            ),
            seed: try optionalUInt64(options["seed"] ?? json["seed"], field: "seed")
        )
        var withMinP = result
        withMinP.minP = try nonNegativeFloat(
            options["min_p"] ?? json["min_p"], field: "min_p", defaultValue: 0.0)
        return withMinP
    }

    /// Result of message extraction: structured messages plus any extracted media.
    internal struct ExtractedMessages {
        let messages: [[String: String]]
        let media: ServerMediaPayload
    }

    /// Parse OpenAI-style chat messages, supporting both string content and the
    /// content-block array form (`{type: text|image_url|input_audio}`).
    static func openAIMessages(from json: [String: Any]) throws -> ExtractedMessages {
        guard let rawMessages = json["messages"] else {
            throw ServerRequestError.missingField("messages")
        }
        guard let messages = rawMessages as? [[String: Any]] else {
            throw ServerRequestError.invalidType(field: "messages", expected: "an array of message objects")
        }

        var media = ServerMediaPayload()
        var structured: [[String: String]] = []
        for (index, message) in messages.enumerated() {
            guard let role = message["role"] as? String else {
                throw ServerRequestError.invalidType(field: "messages[\(index)].role", expected: "a string")
            }

            let rawContent = message["content"]
            // Plain string content — copy through unchanged.
            if let str = rawContent as? String {
                structured.append(["role": role, "content": str])
                continue
            }

            // Content-block array form.
            guard let blocks = rawContent as? [[String: Any]] else {
                throw ServerRequestError.invalidType(
                    field: "messages[\(index)].content",
                    expected: "a string or an array of content blocks"
                )
            }

            var textParts: [String] = []
            for (blockIdx, block) in blocks.enumerated() {
                let type = block["type"] as? String ?? ""
                switch type {
                case "text":
                    guard let t = block["text"] as? String else {
                        throw ServerRequestError.invalidType(
                            field: "messages[\(index)].content[\(blockIdx)].text",
                            expected: "a string"
                        )
                    }
                    textParts.append(t)
                case "image_url":
                    guard let imageURL = block["image_url"] as? [String: Any] else {
                        throw ServerRequestError.invalidType(
                            field: "messages[\(index)].content[\(blockIdx)].image_url",
                            expected: "an object with a 'url' field"
                        )
                    }
                    guard let url = imageURL["url"] as? String else {
                        throw ServerRequestError.invalidType(
                            field: "messages[\(index)].content[\(blockIdx)].image_url.url",
                            expected: "a string"
                        )
                    }
                    guard url.hasPrefix("data:") else {
                        throw ServerRequestError.invalidValue(
                            field: "messages[\(index)].content[\(blockIdx)].image_url.url",
                            reason: "only data: URLs are supported (base64-encoded images)"
                        )
                    }
                    media.images.append(url)
                case "input_audio":
                    guard let audio = block["input_audio"] as? [String: Any] else {
                        throw ServerRequestError.invalidType(
                            field: "messages[\(index)].content[\(blockIdx)].input_audio",
                            expected: "an object with 'data' and 'format' fields"
                        )
                    }
                    guard let data = audio["data"] as? String else {
                        throw ServerRequestError.invalidType(
                            field: "messages[\(index)].content[\(blockIdx)].input_audio.data",
                            expected: "a base64 string"
                        )
                    }
                    if media.audio != nil {
                        throw ServerRequestError.invalidValue(
                            field: "messages[\(index)].content[\(blockIdx)].input_audio",
                            reason: "only one audio clip per request is supported"
                        )
                    }
                    media.audio = data
                    media.audioFormat = audio["format"] as? String
                default:
                    throw ServerRequestError.invalidValue(
                        field: "messages[\(index)].content[\(blockIdx)].type",
                        reason: "unsupported content block type '\(type)'"
                    )
                }
            }
            structured.append(["role": role, "content": textParts.joined(separator: "\n")])
        }

        guard !structured.isEmpty else {
            throw ServerRequestError.invalidValue(field: "messages", reason: "must contain at least one message")
        }
        return ExtractedMessages(messages: structured, media: media)
    }

    /// Parse Ollama-style chat messages. Each message may carry `images` and `audio`
    /// fields; we collect them into the request-level media payload.
    static func ollamaMessages(from json: [String: Any]) throws -> ExtractedMessages {
        guard let rawMessages = json["messages"] else {
            throw ServerRequestError.missingField("messages")
        }
        guard let messages = rawMessages as? [[String: Any]] else {
            throw ServerRequestError.invalidType(field: "messages", expected: "an array of message objects")
        }

        var media = ServerMediaPayload()
        var structured: [[String: String]] = []
        for (index, message) in messages.enumerated() {
            guard let role = message["role"] as? String else {
                throw ServerRequestError.invalidType(field: "messages[\(index)].role", expected: "a string")
            }
            guard let content = message["content"] as? String else {
                throw ServerRequestError.invalidType(field: "messages[\(index)].content", expected: "a string")
            }
            structured.append(["role": role, "content": content])

            if let raw = message["images"] {
                guard let imgs = ServerMultimodal.coerceStringArray(raw) else {
                    throw ServerRequestError.invalidType(
                        field: "messages[\(index)].images",
                        expected: "an array of base64 strings"
                    )
                }
                media.images.append(contentsOf: imgs)
            }
            if let raw = message["audio"] {
                if let s = raw as? String {
                    if media.audio != nil {
                        throw ServerRequestError.invalidValue(
                            field: "messages[\(index)].audio",
                            reason: "only one audio clip per request is supported"
                        )
                    }
                    media.audio = s
                } else {
                    throw ServerRequestError.invalidType(
                        field: "messages[\(index)].audio",
                        expected: "a base64 string"
                    )
                }
            }
        }
        guard !structured.isEmpty else {
            throw ServerRequestError.invalidValue(field: "messages", reason: "must contain at least one message")
        }
        return ExtractedMessages(messages: structured, media: media)
    }

    private static func requiredMessages(from json: [String: Any]) throws -> [[String: String]] {
        guard let rawMessages = json["messages"] else {
            throw ServerRequestError.missingField("messages")
        }
        guard let messages = rawMessages as? [[String: Any]] else {
            throw ServerRequestError.invalidType(field: "messages", expected: "an array of message objects")
        }

        var structured: [[String: String]] = []
        for (index, message) in messages.enumerated() {
            guard let role = message["role"] as? String else {
                throw ServerRequestError.invalidType(field: "messages[\(index)].role", expected: "a string")
            }
            guard let content = message["content"] as? String else {
                throw ServerRequestError.invalidType(field: "messages[\(index)].content", expected: "a string")
            }
            structured.append(["role": role, "content": content])
        }
        guard !structured.isEmpty else {
            throw ServerRequestError.invalidValue(field: "messages", reason: "must contain at least one message")
        }
        return structured
    }

    private static func ollamaTokenLimit(from json: [String: Any]) throws -> Int {
        let options = try optionsObject(from: json)
        let optionLimit = try tokenLimit(
            from: options,
            fields: ["num_predict", "max_tokens"],
            defaultValue: nil
        )
        let topLevelLimit = try tokenLimit(
            from: json,
            fields: ["num_predict", "max_tokens"],
            defaultValue: nil
        )

        if let optionLimit, let topLevelLimit, optionLimit != topLevelLimit {
            throw ServerRequestError.invalidValue(
                field: "max_tokens",
                reason: "conflicting top-level and options token limits"
            )
        }
        return optionLimit ?? topLevelLimit ?? defaultOllamaMaxTokens
    }

    private static func tokenLimit(
        from json: [String: Any],
        fields: [String],
        defaultValue: Int?
    ) throws -> Int? {
        var found: (field: String, value: Int)?
        for field in fields where json[field] != nil {
            // num_predict == -1 means "generate until EOS" (Ollama). Map to a
            // large sentinel cap rather than rejecting it as non-positive.
            if field == "num_predict", let n = json[field] as? Int, n == -1 {
                found = (field, 1 << 20)
                continue
            }
            let value = try positiveInt(json[field], field: field)
            if let existing = found, existing.value != value {
                throw ServerRequestError.invalidValue(
                    field: field,
                    reason: "conflicts with '\(existing.field)'"
                )
            }
            found = (field, value)
        }
        return found?.value ?? defaultValue
    }

    private static func requiredTokenLimit(
        from json: [String: Any],
        fields: [String],
        defaultValue: Int
    ) throws -> Int {
        guard let value = try tokenLimit(from: json, fields: fields, defaultValue: defaultValue) else {
            return defaultValue
        }
        return value
    }

    private static func rejectUnsupportedFields(in json: [String: Any], fields: Set<String>) throws {
        if let field = fields.sorted().first(where: { json[$0] != nil }) {
            throw ServerRequestError.unsupportedField(field)
        }
    }

    private static func optionsObject(from json: [String: Any]) throws -> [String: Any] {
        guard let rawOptions = json["options"] else {
            return [:]
        }
        guard let options = rawOptions as? [String: Any] else {
            throw ServerRequestError.invalidType(field: "options", expected: "an object")
        }
        return options
    }

    private static func stringValue(_ rawValue: Any?, field: String) throws -> String {
        guard let rawValue else {
            throw ServerRequestError.missingField(field)
        }
        guard let value = rawValue as? String else {
            throw ServerRequestError.invalidType(field: field, expected: "a string")
        }
        return value
    }

    private static func optionalString(_ rawValue: Any?, field: String) throws -> String? {
        guard let rawValue else { return nil }
        guard let value = rawValue as? String else {
            throw ServerRequestError.invalidType(field: field, expected: "a string")
        }
        return value
    }

    private static func boolValue(_ rawValue: Any?, field: String) throws -> Bool? {
        guard let rawValue else { return nil }
        if let value = rawValue as? Bool {
            return value
        }
        if let number = rawValue as? NSNumber, isJSONBoolean(number) {
            return number.boolValue
        }
        throw ServerRequestError.invalidType(field: field, expected: "a boolean")
    }

    private static func positiveInt(_ rawValue: Any?, field: String) throws -> Int {
        let value = try intValue(rawValue, field: field)
        guard value > 0 else {
            throw ServerRequestError.invalidValue(field: field, reason: "must be greater than 0")
        }
        return value
    }

    private static func nonNegativeInt(_ rawValue: Any?, field: String, defaultValue: Int) throws -> Int {
        guard rawValue != nil else { return defaultValue }
        let value = try intValue(rawValue, field: field)
        guard value >= 0 else {
            throw ServerRequestError.invalidValue(field: field, reason: "must be greater than or equal to 0")
        }
        return value
    }

    private static func intValue(_ rawValue: Any?, field: String) throws -> Int {
        guard let rawValue else {
            throw ServerRequestError.missingField(field)
        }
        if let number = rawValue as? NSNumber {
            guard !isJSONBoolean(number) else {
                throw ServerRequestError.invalidType(field: field, expected: "an integer")
            }
            let value = number.doubleValue
            guard value.isFinite, value.rounded(.towardZero) == value,
                  value >= Double(Int.min), value <= Double(Int.max) else {
                throw ServerRequestError.invalidType(field: field, expected: "an integer")
            }
            return number.intValue
        }
        if let value = rawValue as? Int {
            return value
        }
        throw ServerRequestError.invalidType(field: field, expected: "an integer")
    }

    private static func optionalUInt64(_ rawValue: Any?, field: String) throws -> UInt64? {
        guard rawValue != nil else { return nil }
        let value = try intValue(rawValue, field: field)
        guard value >= 0 else {
            throw ServerRequestError.invalidValue(field: field, reason: "must be greater than or equal to 0")
        }
        return UInt64(value)
    }

    private static func nonNegativeFloat(_ rawValue: Any?, field: String, defaultValue: Float) throws -> Float {
        guard rawValue != nil else { return defaultValue }
        let value = try floatValue(rawValue, field: field)
        guard value >= 0 else {
            throw ServerRequestError.invalidValue(field: field, reason: "must be greater than or equal to 0")
        }
        return value
    }

    private static func positiveFloat(_ rawValue: Any?, field: String, defaultValue: Float) throws -> Float {
        guard rawValue != nil else { return defaultValue }
        let value = try floatValue(rawValue, field: field)
        guard value > 0 else {
            throw ServerRequestError.invalidValue(field: field, reason: "must be greater than 0")
        }
        return value
    }

    private static func topPValue(_ rawValue: Any?, field: String) throws -> Float {
        guard rawValue != nil else { return 1.0 }
        let value = try floatValue(rawValue, field: field)
        guard value > 0, value <= 1 else {
            throw ServerRequestError.invalidValue(field: field, reason: "must be greater than 0 and less than or equal to 1")
        }
        return value
    }

    private static func floatValue(_ rawValue: Any?, field: String) throws -> Float {
        guard let rawValue else {
            throw ServerRequestError.missingField(field)
        }
        if let number = rawValue as? NSNumber {
            guard !isJSONBoolean(number) else {
                throw ServerRequestError.invalidType(field: field, expected: "a number")
            }
            let value = number.doubleValue
            guard value.isFinite, value >= -Double(Float.greatestFiniteMagnitude),
                  value <= Double(Float.greatestFiniteMagnitude) else {
                throw ServerRequestError.invalidType(field: field, expected: "a finite number")
            }
            return Float(value)
        }
        if let value = rawValue as? Double {
            guard value.isFinite, value >= -Double(Float.greatestFiniteMagnitude),
                  value <= Double(Float.greatestFiniteMagnitude) else {
                throw ServerRequestError.invalidType(field: field, expected: "a finite number")
            }
            return Float(value)
        }
        if let value = rawValue as? Int {
            return Float(value)
        }
        throw ServerRequestError.invalidType(field: field, expected: "a number")
    }

    private static func isJSONBoolean(_ number: NSNumber) -> Bool {
        CFGetTypeID(number) == CFBooleanGetTypeID()
    }
}

internal enum ServerResponseHeads {
    static func ollamaStreaming(version: HTTPVersion = .http1_1) -> HTTPResponseHead {
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/x-ndjson")
        headers.add(name: "Transfer-Encoding", value: "chunked")
        return HTTPResponseHead(version: version, status: .ok, headers: headers)
    }

    static func openAIStreaming(version: HTTPVersion = .http1_1) -> HTTPResponseHead {
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "text/event-stream")
        headers.add(name: "Cache-Control", value: "no-cache")
        headers.add(name: "Connection", value: "keep-alive")
        return HTTPResponseHead(version: version, status: .ok, headers: headers)
    }
}
