import Foundation
import KLMRegistry

/// `list_local_models` operator tool. Wraps `Registry.listModels()`.
public struct ListLocalModelsTool: OperatorTool {
    public let name = "list_local_models"
    public let description =
        "List models that are installed locally (in ~/.krillm/models). " +
        "Each entry has name, family, params, size_gb, and pulled_at."
    public let parametersJSON =
        #"{"type":"object","properties":{},"additionalProperties":false}"#

    private let registry: Registry

    public init(registry: Registry) {
        self.registry = registry
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        let models = registry.listModels()
        if models.isEmpty {
            return encode(["models": [], "count": 0])
        }
        let entries = models.map { m -> [String: Any] in
            let gb = Double(m.sizeBytes) / 1_073_741_824.0
            return [
                "name": m.name,
                "family": m.family.rawValue,
                "params": m.params,
                "quant": m.quant,
                "context": m.context,
                "size_gb": round1(gb),
                "source": m.source,
            ]
        }
        return encode(["models": entries, "count": entries.count])
    }
}

/// `model_info` operator tool. Returns rich detail on one installed
/// model: manifest fields + declared capabilities + support tier +
/// hardware-fit classification against the current machine.
public struct ModelInfoTool: OperatorTool {
    public let name = "model_info"
    public let description =
        "Show details for one installed model by name: family, " +
        "params, size, capabilities, support tier, fit on this machine."
    public let parametersJSON = """
{"type":"object","properties":{\
"name":{"type":"string","description":"Installed model alias."}\
},"required":["name"],"additionalProperties":false}
"""

    private let registry: Registry
    private let hardware: @Sendable () -> HardwareInfo

    public init(
        registry: Registry,
        hardware: @escaping @Sendable () -> HardwareInfo = HardwareInfo.current
    ) {
        self.registry = registry
        self.hardware = hardware
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        guard let name = arguments["name"] as? String, !name.isEmpty else {
            throw RegistryToolError.missingArgument("name")
        }
        guard let m = registry.getModel(name) else {
            throw RegistryToolError.notInstalled(name)
        }
        let caps = ModelCapabilities.capabilities(for: m.family)
        let tier = ModelCapabilities.supportTier(for: m.family)
        let hw = hardware()
        let fit = hw.classifyFit(modelSizeBytes: UInt64(m.sizeBytes))
        let payload: [String: Any] = [
            "name": m.name,
            "family": m.family.rawValue,
            "params": m.params,
            "quant": m.quant,
            "context": m.context,
            "size_gb": round1(Double(m.sizeBytes) / 1_073_741_824.0),
            "source": m.source,
            "capabilities": caps.map(\.rawValue).sorted(),
            "support_tier": tier.rawValue,
            "fit": fit.rawValue,
            "fit_label": fit.label,
        ]
        return encode(payload)
    }
}

/// `remove_model` operator tool. Removes the manifest + blob dir.
///
/// Confirmation lives at the agent boundary (sub-PR C's CLI), not in
/// the tool itself - operator tools never block, they execute. The
/// agent loop is expected to emit a `confirmationNeeded` event
/// BEFORE calling this tool when running interactively; `--yes`
/// short-circuits. Sub-PR B leaves the orchestration to the system
/// prompt's instructions.
public struct RemoveModelTool: OperatorTool {
    public let name = "remove_model"
    public let description =
        "Permanently remove an installed model (manifest + weights). " +
        "Frees the disk it was using. This cannot be undone."
    public let parametersJSON = """
{"type":"object","properties":{\
"name":{"type":"string","description":"Installed model alias to remove."}\
},"required":["name"],"additionalProperties":false}
"""

    private let registry: Registry

    public init(registry: Registry) {
        self.registry = registry
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        guard let name = arguments["name"] as? String, !name.isEmpty else {
            throw RegistryToolError.missingArgument("name")
        }
        guard registry.hasModel(name) else {
            throw RegistryToolError.notInstalled(name)
        }
        let priorSize = registry.getModel(name)?.sizeBytes ?? 0
        try registry.removeModel(name)
        let gb = Double(priorSize) / 1_073_741_824.0
        return String(
            format: "Removed %@ (freed %.2f GB).", name, gb)
    }
}

/// `disk_usage` operator tool. Returns total installed-model bytes,
/// the per-model breakdown, and free disk on the registry partition.
public struct DiskUsageTool: OperatorTool {
    public let name = "disk_usage"
    public let description =
        "Summarize disk usage: total bytes used by installed models, " +
        "free disk on the registry partition, and a per-model breakdown."
    public let parametersJSON =
        #"{"type":"object","properties":{},"additionalProperties":false}"#

    private let registry: Registry
    private let hardware: @Sendable () -> HardwareInfo

    public init(
        registry: Registry,
        hardware: @escaping @Sendable () -> HardwareInfo = HardwareInfo.current
    ) {
        self.registry = registry
        self.hardware = hardware
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        let models = registry.listModels()
            .sorted { $0.sizeBytes > $1.sizeBytes }
        let total = models.reduce(Int64(0)) { $0 + $1.sizeBytes }
        let breakdown = models.map { m -> [String: Any] in
            [
                "name": m.name,
                "size_gb": round1(Double(m.sizeBytes) / 1_073_741_824.0),
            ]
        }
        let payload: [String: Any] = [
            "total_used_gb": round1(Double(total) / 1_073_741_824.0),
            "free_disk_gb": round1(hardware().freeDiskGB),
            "model_count": models.count,
            "by_model": breakdown,
        ]
        return encode(payload)
    }
}

public enum RegistryToolError: Error, CustomStringConvertible, Equatable {
    case missingArgument(String)
    case notInstalled(String)

    public var description: String {
        switch self {
        case .missingArgument(let name):
            return "Missing argument '\(name)'."
        case .notInstalled(let name):
            return "Model '\(name)' is not installed. " +
                "Pull it first with pull_model."
        }
    }
}
