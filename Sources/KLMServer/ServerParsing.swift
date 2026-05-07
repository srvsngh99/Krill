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

    var samplingParams: SamplingParams {
        SamplingParams(
            temperature: temperature,
            topP: topP,
            topK: topK,
            repetitionPenalty: repetitionPenalty,
            seed: seed
        )
    }
}

internal struct ServerChatRequest: Equatable, Sendable {
    let messages: [[String: String]]
    let stream: Bool
    let maxTokens: Int
    let sampling: ServerSamplingOptions
    let requestedModel: String?
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
        "tools", "tool_choice", "parallel_tool_calls",
        "functions", "function_call",
        "response_format", "logprobs", "top_logprobs",
        "stop", "frequency_penalty", "presence_penalty", "logit_bias",
        "stream_options"
    ]

    private static let unsupportedOpenAICompletionFields: Set<String> = [
        "suffix", "best_of", "logprobs", "echo",
        "stop", "frequency_penalty", "presence_penalty", "logit_bias"
    ]

    private static let unsupportedOllamaChatFields: Set<String> = [
        "tools", "format"
    ]

    private static let unsupportedOllamaGenerateFields: Set<String> = [
        "images", "format", "suffix", "context", "template"
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

    static func openAIChatRequest(from json: [String: Any]) throws -> ServerChatRequest {
        try rejectUnsupportedFields(in: json, fields: unsupportedOpenAIChatFields)
        let messages = try requiredMessages(from: json)
        return ServerChatRequest(
            messages: messages,
            stream: try boolValue(json["stream"], field: "stream") ?? false,
            maxTokens: try requiredTokenLimit(
                from: json,
                fields: ["max_tokens", "max_completion_tokens"],
                defaultValue: defaultOpenAIMaxTokens
            ),
            sampling: try openAISamplingOptions(from: json),
            requestedModel: try optionalString(json["model"], field: "model")
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
        ServerSamplingOptions(
            temperature: try nonNegativeFloat(json["temperature"], field: "temperature", defaultValue: 0.0),
            topP: try topPValue(json["top_p"], field: "top_p"),
            topK: try nonNegativeInt(json["top_k"], field: "top_k", defaultValue: 0),
            repetitionPenalty: 1.0,
            seed: try optionalUInt64(json["seed"], field: "seed")
        )
    }

    static func ollamaChatRequest(from json: [String: Any]) throws -> ServerChatRequest {
        try rejectUnsupportedFields(in: json, fields: unsupportedOllamaChatFields)
        let messages = try requiredMessages(from: json)
        return ServerChatRequest(
            messages: messages,
            stream: try boolValue(json["stream"], field: "stream") ?? true,
            maxTokens: try ollamaTokenLimit(from: json),
            sampling: try ollamaSamplingOptions(from: json),
            requestedModel: try optionalString(json["model"], field: "model")
        )
    }

    static func ollamaGenerateRequest(from json: [String: Any]) throws -> ServerGenerateRequest {
        try rejectUnsupportedFields(in: json, fields: unsupportedOllamaGenerateFields)
        if try boolValue(json["raw"], field: "raw") == true {
            throw ServerRequestError.unsupportedField("raw")
        }
        return ServerGenerateRequest(
            prompt: try stringValue(json["prompt"], field: "prompt"),
            system: try optionalString(json["system"], field: "system"),
            stream: try boolValue(json["stream"], field: "stream") ?? true,
            maxTokens: try ollamaTokenLimit(from: json),
            sampling: try ollamaSamplingOptions(from: json),
            requestedModel: try optionalString(json["model"], field: "model")
        )
    }

    static func ollamaSamplingOptions(from json: [String: Any]) throws -> ServerSamplingOptions {
        let options = try optionsObject(from: json)
        return ServerSamplingOptions(
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
}
