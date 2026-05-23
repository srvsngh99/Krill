import Foundation
import KLMRegistry

/// `list_remote_catalog` operator tool. Returns the curated catalog
/// the agent's `recommend_model` ranks. Defaults to the built-in
/// `AliasMap` so the tool is useful before the user has fetched a
/// remote catalog snapshot; when a `ModelCatalogStore` is supplied,
/// any matching entries from that catalog are added (built-in wins
/// on duplicate aliases, matching `AliasMap.resolve`).
///
/// Optional argument `capability` filters by declared family
/// capability (`text_generation`, `vision_input`, `audio_input`,
/// `tools`, `embeddings`, `reranker`, `moe`).
public struct ListRemoteCatalogTool: OperatorTool {
    public let name = "list_remote_catalog"
    public let description =
        "List all known model aliases the agent can pull, optionally " +
        "filtered to one capability tag."
    public let parametersJSON = """
{"type":"object","properties":{\
"capability":{"type":"string","description":\
"Optional: filter by capability tag (text_generation, vision_input, \
audio_input, tools, embeddings, reranker, moe)."}\
},"additionalProperties":false}
"""

    private let catalogStore: ModelCatalogStore?

    public init(catalogStore: ModelCatalogStore? = nil) {
        self.catalogStore = catalogStore
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        let requestedTag = arguments["capability"] as? String
        let requestedCap = requestedTag.flatMap(parseCapability)

        var entries = CatalogEntry.fromAliasMap()
        if let store = catalogStore, let remote = store.load() {
            // Built-in wins; remote-only entries are appended.
            let known = Set(entries.map(\.alias))
            for r in remote.models where !known.contains(r.alias) {
                entries.append(r)
            }
        }

        let filtered = entries.filter { entry in
            guard let required = requestedCap else { return true }
            return ModelCapabilities.capabilities(for: entry.family)
                .contains(required)
        }

        let payload: [String: Any] = [
            "count": filtered.count,
            "filtered_by_capability": requestedTag as Any? ?? NSNull(),
            "entries": filtered.map { e -> [String: Any] in
                let caps = ModelCapabilities.capabilities(for: e.family)
                return [
                    "alias": e.alias,
                    "repo": e.repo,
                    "family": e.family.rawValue,
                    "params": e.params,
                    "quant": e.quant,
                    "context": e.context,
                    "capabilities": caps.map(\.rawValue).sorted(),
                ]
            },
        ]
        return encode(payload)
    }

    private func parseCapability(_ tag: String) -> Capability? {
        switch tag.lowercased() {
        case "text_generation", "textgeneration", "completion":
            return .textGeneration
        case "vision_input", "visioninput", "vision":
            return .visionInput
        case "audio_input", "audioinput", "audio":
            return .audioInput
        case "embeddings", "embedding":
            return .embeddings
        case "tools":
            return .tools
        case "structured_output", "structuredoutput":
            return .structuredOutput
        case "moe":
            return .moe
        case "reranker":
            return .reranker
        default:
            return nil
        }
    }
}

/// `recommend_model` operator tool. Returns a ranked shortlist for
/// the requested capability. Reads the catalog through the same
/// AliasMap-plus-store path as `list_remote_catalog`, hardware
/// through `HardwareInfo.current()` (overridable for tests).
public struct RecommendModelTool: OperatorTool {
    public let name = "recommend_model"
    public let description =
        "Recommend models that fit the user's hardware for a given " +
        "capability. Returns a ranked shortlist with fit annotations."
    public let parametersJSON = """
{"type":"object","properties":{\
"capability":{"type":"string","description":\
"Capability tag the model must declare (text_generation, vision_input, \
audio_input, tools, embeddings, reranker)."},\
"limit":{"type":"integer","description":\
"How many candidates to return (default 5)."}\
},"required":["capability"],"additionalProperties":false}
"""

    private let catalogStore: ModelCatalogStore?
    private let hardware: @Sendable () -> HardwareInfo

    public init(
        catalogStore: ModelCatalogStore? = nil,
        hardware: @escaping @Sendable () -> HardwareInfo = HardwareInfo.current
    ) {
        self.catalogStore = catalogStore
        self.hardware = hardware
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        guard let capTag = arguments["capability"] as? String,
              let cap = ListRemoteCatalogTool().tryParseCapability(capTag)
        else {
            throw RecommendModelToolError.unknownCapability(
                (arguments["capability"] as? String) ?? "(missing)")
        }
        let limit = (arguments["limit"] as? Int) ?? 5

        var entries = CatalogEntry.fromAliasMap()
        if let store = catalogStore, let remote = store.load() {
            let known = Set(entries.map(\.alias))
            for r in remote.models where !known.contains(r.alias) {
                entries.append(r)
            }
        }

        let hw = hardware()
        let ranked = Recommender.recommend(
            required: [cap],
            catalog: entries,
            hardware: hw,
            limit: max(1, limit))

        let payload: [String: Any] = [
            "capability": capTag,
            "hardware": [
                "arch": hw.arch,
                "total_ram_gb": round1(hw.totalRAMGB),
            ],
            "count": ranked.count,
            "candidates": ranked.map { r -> [String: Any] in
                [
                    "alias": r.alias,
                    "family": r.family.rawValue,
                    "params": r.params,
                    "context": r.context,
                    "estimated_size_gb":
                        round1(Double(r.estimatedSizeBytes) / 1_073_741_824.0),
                    "fit": r.fit.rawValue,
                    "fit_label": r.fit.label,
                    "support_tier": r.supportTier.rawValue,
                    "score": r.score,
                ]
            },
        ]
        return encode(payload)
    }
}

public enum RecommendModelToolError: Error, CustomStringConvertible, Equatable {
    case unknownCapability(String)

    public var description: String {
        switch self {
        case .unknownCapability(let tag):
            return "Unknown capability '\(tag)'. " +
                "Valid: text_generation, vision_input, audio_input, " +
                "tools, embeddings, reranker, moe."
        }
    }
}

internal extension ListRemoteCatalogTool {
    func tryParseCapability(_ tag: String) -> Capability? {
        switch tag.lowercased() {
        case "text_generation", "textgeneration", "completion":
            return .textGeneration
        case "vision_input", "visioninput", "vision":
            return .visionInput
        case "audio_input", "audioinput", "audio":
            return .audioInput
        case "embeddings", "embedding":
            return .embeddings
        case "tools":
            return .tools
        case "structured_output", "structuredoutput":
            return .structuredOutput
        case "moe":
            return .moe
        case "reranker":
            return .reranker
        default:
            return nil
        }
    }
}
