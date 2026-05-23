import Foundation

/// Thin HTTP client that talks to a locally running `krillm serve`
/// daemon. Used by `krillm run` to detect an already-running daemon
/// and route single-shot text generation through it (avoiding a fresh
/// in-process model load + Metal JIT spike per CLI invocation).
///
/// Wire contract is the daemon's own endpoints in Server.swift:
/// `GET /v1/status` (probe) and `POST /v1/chat/completions` (stream).
/// Co-located with the server so the contract evolves in one place.
public struct DaemonClient {

    public struct Status: Sendable, Equatable {
        public let modelLoaded: Bool
        public let model: String?
    }

    public struct ChatResult: Sendable {
        public let tokenCount: Int
        public let wallTimeSec: Double
    }

    public enum ChatError: Error, CustomStringConvertible {
        case httpError(Int)
        case streamTruncated
        case malformedResponse(String)

        public var description: String {
            switch self {
            case .httpError(let code):
                return "daemon returned HTTP \(code)"
            case .streamTruncated:
                return "daemon stream ended without [DONE]"
            case .malformedResponse(let detail):
                return "daemon response malformed: \(detail)"
            }
        }
    }

    /// Probe the daemon's `/v1/status` endpoint. Returns nil if
    /// unreachable, times out, or returns malformed JSON. The default
    /// 200 ms timeout keeps a failed probe from delaying the
    /// in-process fallback noticeably.
    public static func probeStatus(
        port: Int,
        timeout: TimeInterval = 0.2
    ) async -> Status? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/v1/status") else {
            return nil
        }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = timeout
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                return nil
            }
            return parseStatus(data: data)
        } catch {
            return nil
        }
    }

    /// Parse a `/v1/status` JSON payload into a `Status`. Returns nil
    /// on malformed JSON or missing `model_loaded`. Exposed internally
    /// so unit tests can exercise the parser against canned payloads
    /// without spinning a real daemon.
    static func parseStatus(data: Data) -> Status? {
        guard let object = try? JSONSerialization.jsonObject(with: data),
              let json = object as? [String: Any],
              let modelLoaded = json["model_loaded"] as? Bool else {
            return nil
        }
        let model = json["model"] as? String
        return Status(modelLoaded: modelLoaded, model: model)
    }

    /// Stream a chat completion against `/v1/chat/completions`. Each
    /// delta token is delivered via `onToken`. Returns after the
    /// daemon sends `data: [DONE]`. Throws on non-200 HTTP, an
    /// abruptly closed stream, or malformed SSE frames.
    public static func streamChat(
        port: Int,
        model: String,
        messages: [(role: String, content: String)],
        temperature: Float,
        topP: Float,
        maxTokens: Int,
        seed: UInt64?,
        onToken: @Sendable (String) -> Void
    ) async throws -> ChatResult {
        guard let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions") else {
            throw ChatError.malformedResponse("invalid URL")
        }

        var body: [String: Any] = [
            "model": model,
            "messages": messages.map { ["role": $0.role, "content": $0.content] },
            "stream": true,
            "temperature": Double(temperature),
            "top_p": Double(topP),
            "max_tokens": maxTokens
        ]
        if let seed { body["seed"] = Int(seed) }
        let bodyData = try JSONSerialization.data(withJSONObject: body)

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.httpBody = bodyData

        let start = Date()
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ChatError.malformedResponse("non-HTTP response")
        }
        guard http.statusCode == 200 else {
            throw ChatError.httpError(http.statusCode)
        }

        var tokenCount = 0
        var sawDone = false
        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst("data: ".count))
            if payload == "[DONE]" { sawDone = true; break }
            if let token = parseChunkContent(payload), !token.isEmpty {
                onToken(token)
                tokenCount += 1
            }
        }
        guard sawDone else { throw ChatError.streamTruncated }

        return ChatResult(
            tokenCount: tokenCount,
            wallTimeSec: Date().timeIntervalSince(start)
        )
    }

    /// Pull `choices[0].delta.content` out of a `chat.completion.chunk`
    /// payload. Returns nil if the shape is unexpected. Exposed for tests.
    static func parseChunkContent(_ payload: String) -> String? {
        guard let data = payload.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let json = object as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let first = choices.first,
              let delta = first["delta"] as? [String: Any] else {
            return nil
        }
        return delta["content"] as? String
    }
}
