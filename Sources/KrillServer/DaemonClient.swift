import Foundation
import KrillTooling

/// Thin HTTP client that talks to a locally running `krill serve`
/// daemon. Used by `krill run` to detect an already-running daemon
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
        /// See StreamProgress.contentChunkCount for the chunk-vs-token
        /// distinction; the daemon path does not surface a true token
        /// count today.
        public let contentChunkCount: Int
        public let wallTimeSec: Double
    }

    public enum ChatError: Error, CustomStringConvertible {
        case httpError(Int, body: String)
        case streamTruncated
        case streamErrorFrame(String)
        case malformedResponse(String)

        public var description: String {
            switch self {
            case .httpError(let code, let body):
                let bodyDetail = body.isEmpty ? "" : ": \(body)"
                return "daemon returned HTTP \(code)\(bodyDetail)"
            case .streamTruncated:
                return "daemon stream ended without [DONE]"
            case .streamErrorFrame(let detail):
                return "daemon reported error mid-stream: \(detail)"
            case .malformedResponse(let detail):
                return "daemon response malformed: \(detail)"
            }
        }
    }

    public struct StreamProgress: Sendable {
        /// Number of non-empty content chunks seen. Note: this is
        /// chunks, not tokens. The server's StreamingReasoningFilter
        /// can buffer or drop chunks (eg around `<think>` blocks)
        /// and a single chunk may carry multiple tokens, so this
        /// value is a coarse progress indicator, not a token count.
        public let contentChunkCount: Int
        public let sawDone: Bool
    }

    /// Probe the daemon's `/v1/status` endpoint. Returns nil if
    /// unreachable, times out, or returns malformed JSON. The default
    /// 200 ms timeout keeps a failed probe from delaying the
    /// in-process fallback noticeably.
    public static func probeStatus(
        port: Int,
        apiKey: String? = nil,
        timeout: TimeInterval = 0.2
    ) async -> Status? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/v1/status") else {
            return nil
        }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = timeout
        if let apiKey = ServerSecurity.normalizedAPIKey(apiKey) {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
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
        apiKey: String? = nil,
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
        // Wrap as NSNumber so UInt64 values above Int.max do not trap
        // the narrowing Int(seed) conversion. JSONSerialization preserves
        // the original value; the server validates the range and returns
        // a clear 4xx if it cannot accept a given seed.
        if let seed { body["seed"] = NSNumber(value: seed) }
        let bodyData = try JSONSerialization.data(withJSONObject: body)

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        if let apiKey = ServerSecurity.normalizedAPIKey(apiKey) {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = bodyData

        let start = Date()
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ChatError.malformedResponse("non-HTTP response")
        }
        guard http.statusCode == 200 else {
            // Drain a bounded amount of the error body into the thrown
            // error so the caller sees the server's actual reason
            // (eg `{"error":"model name mismatch ..."}`) rather than a
            // bare status code.
            let body = try await Self.collectErrorBody(bytes: bytes)
            throw ChatError.httpError(http.statusCode, body: body)
        }

        let progress = try await consumeSSE(lines: bytes.lines, onToken: onToken)
        guard progress.sawDone else { throw ChatError.streamTruncated }

        return ChatResult(
            contentChunkCount: progress.contentChunkCount,
            wallTimeSec: Date().timeIntervalSince(start)
        )
    }

    /// Drive an SSE line stream: forward content chunks to `onToken`,
    /// throw on the first `{"error": ...}` frame, return when `[DONE]`
    /// is seen (or when the sequence ends without it; the caller decides
    /// whether that is a truncation). Exposed as `internal` so tests
    /// can feed a synthetic line stream without spinning a daemon.
    static func consumeSSE<S: AsyncSequence>(
        lines: S,
        onToken: @Sendable (String) -> Void
    ) async throws -> StreamProgress where S.Element == String {
        var contentChunkCount = 0
        var sawDone = false
        for try await line in lines {
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst("data: ".count))
            if payload == "[DONE]" { sawDone = true; break }
            if let err = parseChunkError(payload) {
                throw ChatError.streamErrorFrame(err)
            }
            if let token = parseChunkContent(payload), !token.isEmpty {
                onToken(token)
                contentChunkCount += 1
            }
        }
        return StreamProgress(
            contentChunkCount: contentChunkCount,
            sawDone: sawDone
        )
    }

    /// Read up to 4 KB of an HTTP error body so the surfaced error
    /// can include the server's reason without unbounded memory or
    /// time use on a misbehaving daemon.
    private static func collectErrorBody(bytes: URLSession.AsyncBytes) async throws -> String {
        var buffer = Data()
        let cap = 4096
        for try await byte in bytes {
            buffer.append(byte)
            if buffer.count >= cap { break }
        }
        return String(data: buffer, encoding: .utf8) ?? ""
    }

    /// Detect a daemon-side SSE error frame. Server.swift emits
    /// `data: {"error": "..."}` on overflow / queue-busy paths
    /// before sending `[DONE]`. Returns the error message string if
    /// the payload carries a top-level `error` field, else nil.
    static func parseChunkError(_ payload: String) -> String? {
        guard let data = payload.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let json = object as? [String: Any] else {
            return nil
        }
        return json["error"] as? String
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
