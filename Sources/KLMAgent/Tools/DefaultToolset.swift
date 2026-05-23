import Foundation
import KLMRegistry

/// Convenience constructor for the 13-tool default operator toolset,
/// wired against the production dependencies.
///
/// Sub-PR C's CLI uses this directly; tests typically build a smaller
/// custom registry so they can inject fixtures per tool.
public enum DefaultOperatorToolset {

    /// Build the default toolset for an agent session.
    ///
    /// - Parameters:
    ///   - registry: the local model registry (typically `Registry()`).
    ///   - catalogStore: optional remote-catalog cache. When `nil`,
    ///     `list_remote_catalog` and `recommend_model` use the
    ///     built-in `AliasMap` only.
    ///   - daemonOps: daemon HTTP client (typically `DaemonClientAgentOps`).
    ///   - puller: production puller (typically `RegistryAgentPuller`).
    ///   - server: server-process surface (default: instructions only).
    ///   - hardware: hardware snapshot provider (default: probe now).
    public static func make(
        registry: Registry,
        catalogStore: ModelCatalogStore? = nil,
        daemonOps: any AgentDaemonOps,
        puller: any AgentPuller,
        server: any AgentServerProcess = InstructionsOnlyServerProcess(),
        hardware: @escaping @Sendable () -> HardwareInfo = HardwareInfo.current
    ) -> OperatorToolRegistry {
        let tools: [any OperatorTool] = [
            HardwareInfoTool(snapshot: hardware),
            ListLocalModelsTool(registry: registry),
            ListRemoteCatalogTool(catalogStore: catalogStore),
            ModelInfoTool(registry: registry, hardware: hardware),
            RecommendModelTool(
                catalogStore: catalogStore, hardware: hardware),
            PullModelTool(puller: puller),
            RemoveModelTool(registry: registry),
            ServerStatusTool(ops: daemonOps),
            LoadModelTool(ops: daemonOps),
            UnloadModelTool(ops: daemonOps),
            StartServerTool(process: server),
            StopServerTool(process: server),
            DiskUsageTool(registry: registry, hardware: hardware),
        ]
        return OperatorToolRegistry(tools)
    }
}
