import Foundation

/// `server_status` operator tool. Probes the local daemon and
/// reports `model_loaded` + which model + memory + uptime, or "not
/// running" when the probe fails.
public struct ServerStatusTool: OperatorTool {
    public let name = "server_status"
    public let description =
        "Report whether krillm serve is running locally, which " +
        "model is loaded, memory in use, and how long it has been up."
    public let parametersJSON =
        #"{"type":"object","properties":{},"additionalProperties":false}"#

    private let ops: any AgentDaemonOps

    public init(ops: any AgentDaemonOps) {
        self.ops = ops
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        guard let status = await ops.status() else {
            return encode([
                "running": false,
                "hint": "No daemon responded; start one with `krillm serve`.",
            ])
        }
        return encode([
            "running": true,
            "model_loaded": status.modelLoaded,
            "model": status.model as Any? ?? NSNull(),
            "memory_mb": status.memoryMB as Any? ?? NSNull(),
            "model_uptime_seconds": status.modelUptimeSeconds as Any? ?? NSNull(),
        ])
    }
}

/// `load_model` operator tool. Asks the daemon to swap to `name`.
/// The daemon's `POST /v1/models/load` unloads the prior model.
public struct LoadModelTool: OperatorTool {
    public let name = "load_model"
    public let description =
        "Tell the running daemon to load (and swap to) a model by " +
        "name. Any currently loaded model is unloaded first."
    public let parametersJSON = """
{"type":"object","properties":{\
"name":{"type":"string","description":"Installed model alias to load."}\
},"required":["name"],"additionalProperties":false}
"""

    private let ops: any AgentDaemonOps

    public init(ops: any AgentDaemonOps) {
        self.ops = ops
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        guard let name = arguments["name"] as? String, !name.isEmpty else {
            throw RegistryToolError.missingArgument("name")
        }
        return try await ops.loadModel(name)
    }
}

/// `unload_model` operator tool. Asks the daemon to evict the
/// currently-loaded model (freeing RAM but leaving the daemon up).
public struct UnloadModelTool: OperatorTool {
    public let name = "unload_model"
    public let description =
        "Tell the running daemon to unload its current model and " +
        "free its RAM. The daemon itself stays up."
    public let parametersJSON =
        #"{"type":"object","properties":{},"additionalProperties":false}"#

    private let ops: any AgentDaemonOps

    public init(ops: any AgentDaemonOps) {
        self.ops = ops
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        return try await ops.unloadModel()
    }
}

/// `start_server` operator tool. Default impl returns instructions
/// (per `InstructionsOnlyServerProcess`); sub-PR C may wire a
/// process-spawning variant under `--yes`.
public struct StartServerTool: OperatorTool {
    public let name = "start_server"
    public let description =
        "Show how to start a krillm daemon. Optionally specify the " +
        "initial model to load."
    public let parametersJSON = """
{"type":"object","properties":{\
"model":{"type":"string","description":\
"Optional: model alias to load on startup."}\
},"additionalProperties":false}
"""

    private let process: any AgentServerProcess

    public init(process: any AgentServerProcess = InstructionsOnlyServerProcess()) {
        self.process = process
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        let model = arguments["model"] as? String
        return try await process.start(model: model)
    }
}

/// `stop_server` operator tool.
public struct StopServerTool: OperatorTool {
    public let name = "stop_server"
    public let description =
        "Show how to stop a running krillm daemon."
    public let parametersJSON =
        #"{"type":"object","properties":{},"additionalProperties":false}"#

    private let process: any AgentServerProcess

    public init(process: any AgentServerProcess = InstructionsOnlyServerProcess()) {
        self.process = process
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        return try await process.stop()
    }
}
