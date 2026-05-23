import Foundation
import KLMServer

/// What the operator agent's daemon-facing tools (`server_status`,
/// `load_model`, `unload_model`) need from "the running krillm serve".
///
/// Protocol-shaped so unit tests can inject a fixture without spinning
/// a real HTTP server. The production implementation
/// (`DaemonClientAgentOps`) speaks to `127.0.0.1:$port`.
public protocol AgentDaemonOps: Sendable {
    /// `nil` when the daemon is not reachable / status is malformed.
    func status() async -> AgentDaemonStatus?

    /// Ask the running daemon to load `name`. Returns a human-readable
    /// result string the tool surfaces back to the model. Throws on
    /// HTTP / transport failure so the loop can surface it via
    /// `OperatorToolResult.isError`.
    func loadModel(_ name: String) async throws -> String

    /// Ask the running daemon to unload the currently loaded model.
    func unloadModel() async throws -> String
}

/// Subset of `/v1/status` the operator tools care about.
public struct AgentDaemonStatus: Sendable, Equatable {
    public let modelLoaded: Bool
    public let model: String?
    public let memoryMB: Int?
    public let modelUptimeSeconds: Int?

    public init(
        modelLoaded: Bool, model: String?,
        memoryMB: Int?, modelUptimeSeconds: Int?
    ) {
        self.modelLoaded = modelLoaded
        self.model = model
        self.memoryMB = memoryMB
        self.modelUptimeSeconds = modelUptimeSeconds
    }
}

/// Production `AgentDaemonOps` that talks HTTP to a local
/// `krillm serve` on `port` (default 11435, matching the
/// `KRILL_PORT` convention).
public struct DaemonClientAgentOps: AgentDaemonOps {
    private let port: Int
    private let probeTimeout: TimeInterval

    public init(port: Int = 11435, probeTimeout: TimeInterval = 0.5) {
        self.port = port
        self.probeTimeout = probeTimeout
    }

    public func status() async -> AgentDaemonStatus? {
        // Fetch the raw status payload directly so we surface
        // memory_mb / model_uptime_seconds (`DaemonClient.probeStatus`
        // returns only model + modelLoaded). Same endpoint contract,
        // same default timeout.
        guard let url = URL(string: "http://127.0.0.1:\(port)/v1/status")
        else { return nil }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = probeTimeout
        do {
            let (data, response) = try await URLSession.shared
                .data(for: request)
            guard let http = response as? HTTPURLResponse,
                  http.statusCode == 200
            else { return nil }
            guard let json = try JSONSerialization.jsonObject(
                with: data) as? [String: Any] else { return nil }
            let modelLoaded = (json["model_loaded"] as? Bool) ?? false
            let model = json["model"] as? String
            let memoryMB = json["memory_mb"] as? Int
            let modelUptime = json["model_uptime_seconds"] as? Int
            return AgentDaemonStatus(
                modelLoaded: modelLoaded,
                model: model,
                memoryMB: memoryMB,
                modelUptimeSeconds: modelUptime)
        } catch {
            return nil
        }
    }

    public func loadModel(_ name: String) async throws -> String {
        try await postJSON(
            path: "/v1/models/load",
            body: ["model": name],
            successMessage: "Loaded \(name).")
    }

    public func unloadModel() async throws -> String {
        try await postJSON(
            path: "/v1/models/unload",
            body: [:],
            successMessage: "Unloaded the current model.")
    }

    private func postJSON(
        path: String,
        body: [String: Any],
        successMessage: String
    ) async throws -> String {
        guard let url = URL(string: "http://127.0.0.1:\(port)\(path)")
        else { throw AgentDaemonError.badURL(path) }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !body.isEmpty {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } else {
            request.httpBody = Data("{}".utf8)
        }
        request.timeoutInterval = 30
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw AgentDaemonError.transport("no HTTP response")
        }
        if (200..<300).contains(http.statusCode) {
            return successMessage
        }
        let bodyText = String(data: data, encoding: .utf8) ?? ""
        throw AgentDaemonError.httpError(http.statusCode, bodyText)
    }
}

public enum AgentDaemonError: Error, CustomStringConvertible {
    case badURL(String)
    case transport(String)
    case httpError(Int, String)

    public var description: String {
        switch self {
        case .badURL(let p): return "bad daemon URL for path '\(p)'"
        case .transport(let s): return "daemon transport error: \(s)"
        case .httpError(let code, let body):
            return body.isEmpty
                ? "daemon HTTP \(code)"
                : "daemon HTTP \(code): \(body)"
        }
    }
}
