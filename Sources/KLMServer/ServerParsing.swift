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

    var samplingParams: SamplingParams {
        SamplingParams(
            temperature: temperature,
            topP: topP,
            topK: topK
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

internal enum ServerParsing {
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

    static func openAIChatRequest(from json: [String: Any]) -> ServerChatRequest {
        let messages = json["messages"] as? [[String: Any]] ?? []
        return ServerChatRequest(
            messages: structuredMessages(from: messages),
            stream: json["stream"] as? Bool ?? false,
            maxTokens: json["max_tokens"] as? Int ?? 512,
            sampling: openAISamplingOptions(from: json),
            requestedModel: json["model"] as? String
        )
    }

    static func openAISamplingOptions(from json: [String: Any]) -> ServerSamplingOptions {
        ServerSamplingOptions(
            temperature: Float(json["temperature"] as? Double ?? 0.0),
            topP: Float(json["top_p"] as? Double ?? 1.0),
            topK: json["top_k"] as? Int ?? 0
        )
    }

    static func ollamaChatRequest(from json: [String: Any]) -> ServerChatRequest {
        let messages = json["messages"] as? [[String: Any]] ?? []
        return ServerChatRequest(
            messages: structuredMessages(from: messages),
            stream: json["stream"] as? Bool ?? true,
            maxTokens: 2048,
            sampling: ollamaSamplingOptions(from: json),
            requestedModel: json["model"] as? String
        )
    }

    static func ollamaSamplingOptions(from json: [String: Any]) -> ServerSamplingOptions {
        let options = json["options"] as? [String: Any] ?? [:]
        return ServerSamplingOptions(
            temperature: Float(options["temperature"] as? Double ?? 0.0),
            topP: Float(options["top_p"] as? Double ?? 1.0),
            topK: options["top_k"] as? Int ?? 0
        )
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
