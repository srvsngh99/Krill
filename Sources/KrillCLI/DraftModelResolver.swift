import Foundation
import KrillEngine
import KrillRegistry

/// Resolves a `--draft-model` argument into a concrete on-disk directory and
/// hands it to the engine. Centralized so `run` and `serve` agree on
/// resolution order (path > registry alias > curated pair).
enum DraftModelResolver {
    /// Load the requested draft model into `engine`. The `target` argument is
    /// the user-facing name of the target model (alias or path); it is only
    /// used when `draftSpec == "auto"` to consult the curated `draftPairs`
    /// map. Prints a one-line load summary on success; throws on failure so
    /// the caller can decide whether to abort (CLI run) or warn (serve).
    static func load(
        draftSpec: String,
        target: String,
        registry: Registry,
        engine: InferenceEngine
    ) throws {
        let resolved: URL
        let displayName: String
        if draftSpec == "auto" {
            // Try the curated pair map against the bare target name.
            let key = (target as NSString).lastPathComponent
            guard let pair = recommendedDraft(for: key) else {
                throw DraftResolutionError.noAutoPair(target: key)
            }
            guard registry.hasModel(pair) else {
                throw DraftResolutionError.aliasMissing(alias: pair)
            }
            resolved = registry.modelPath(pair)
            displayName = pair
        } else if registry.hasModel(draftSpec) {
            resolved = registry.modelPath(draftSpec)
            displayName = draftSpec
        } else {
            let url = URL(fileURLWithPath: draftSpec)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw DraftResolutionError.notFound(spec: draftSpec)
            }
            resolved = url
            displayName = url.lastPathComponent
        }
        try engine.loadDraftModel(from: resolved)
        print("Speculative decoding enabled with draft model: \(displayName)")
    }
}

enum DraftResolutionError: Error, CustomStringConvertible {
    case noAutoPair(target: String)
    case aliasMissing(alias: String)
    case notFound(spec: String)

    var description: String {
        switch self {
        case .noAutoPair(let target):
            return "no curated draft pair for target '\(target)' (see SpeculativeDecoder.draftPairs)"
        case .aliasMissing(let alias):
            return "draft alias '\(alias)' not installed; run: krill pull \(alias)"
        case .notFound(let spec):
            return "draft model '\(spec)' not found as alias or path"
        }
    }
}
