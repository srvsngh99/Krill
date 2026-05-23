import Foundation
import KLMRegistry

/// The slice of `KLMRegistry.Puller` the operator agent's
/// `pull_model` tool needs, behind a protocol so tests can inject a
/// fixture that does not hit HuggingFace.
public protocol AgentPuller: Sendable {
    /// Pull the model named `alias`. Resolves through the built-in
    /// alias map first, then the catalog store. Returns a short
    /// status string on success; throws otherwise.
    func pull(alias: String) async throws -> String
}

/// Production `AgentPuller`. Wraps `KLMRegistry.Puller` and resolves
/// names through `AliasMap` + `ModelCatalogStore`.
public struct RegistryAgentPuller: AgentPuller {
    private let registry: Registry
    private let catalogStore: ModelCatalogStore?

    public init(registry: Registry, catalogStore: ModelCatalogStore? = nil) {
        self.registry = registry
        self.catalogStore = catalogStore
    }

    public func pull(alias: String) async throws -> String {
        guard let resolved = AliasMap.resolve(alias, catalog: catalogStore)
        else {
            throw AgentPullerError.unknownModel(alias)
        }
        if registry.hasModel(resolved.name) {
            return "Model \(resolved.name) is already installed."
        }
        let puller = Puller(registry: registry)
        let manifest = try await puller.pull(resolved, force: false, progress: nil)
        let gb = Double(manifest.sizeBytes) / 1_073_741_824.0
        return String(
            format: "Pulled %@ from %@ (%.2f GB on disk).",
            manifest.name, manifest.source, gb)
    }
}

public enum AgentPullerError: Error, CustomStringConvertible, Equatable {
    case unknownModel(String)

    public var description: String {
        switch self {
        case .unknownModel(let name):
            return "Unknown model alias '\(name)'. " +
                "Run list_remote_catalog to see what is available."
        }
    }
}
