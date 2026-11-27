import Foundation
import MLX
import KrillCore
import KrillRegistry

/// Holds one loaded `DecisionModel` for the server: load, unload, and one
/// decision call at a time.
public final class DecisionEngine: @unchecked Sendable {
    private var model: DecisionModel?
    private var loadedDir: URL?
    private let state = NSLock()
    private let run = NSLock()

    public init() {}

    public var loadedModelName: String? { withState { loadedDir?.lastPathComponent } }

    @discardableResult
    public func unload() -> Bool {
        let was: Bool = withState {
            defer { model = nil; loadedDir = nil }
            return model != nil
        }
        if was { MLX.Memory.clearCache() }
        return was
    }

    /// Returns the model for `directory`; the caller runs its calls on that model, so a
    /// concurrent load of another bundle cannot swap it mid-request.
    public func load(directory: URL, forTuning: Bool = false) async throws -> DecisionModel {
        if forTuning { try DecisionModel.requireTunable(DecisionFamily.detect(directory)) }
        if let current = withState({ loadedDir?.standardizedFileURL == directory.standardizedFileURL ? model : nil }) {
            return current
        }
        MLXMemoryConfig.apply(defaultMB: EmbeddingEngine.defaultCacheLimitMB)
        let loaded = try await DecisionModel.load(directory: directory)
        withState { model = loaded; loadedDir = directory }
        return loaded
    }

    public func tune(_ model: DecisionModel, base: ModelManifest, name: String, examples: [DecisionTuneExample],
                     options: DecisionTuneOptions = .init(), registry: Registry) throws -> DecisionTuneReport {
        try serialized {
            try registry.requireAvailableModelName(name)
            try registry.ensureDirectories()
            MLXMemoryConfig.apply(defaultMB: EmbeddingEngine.defaultCacheLimitMB)
            let staging = registry.blobsDir.appendingPathComponent(".tune-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: staging) }
            let report = try model.tune(examples: examples, options: options, output: staging)
            if report.accepted { try registry.installTunedDecision(named: name, from: base, stagedBundle: staging) }
            return report
        }
    }

    public func choose(_ model: DecisionModel, _ texts: [String], options: [String], question: String?,
                       allowNone: Bool) throws -> [DecisionChoice] {
        try serialized { try model.choose(texts, options: options, question: question, allowNone: allowNone) }
    }

    public func yesNo(_ model: DecisionModel, _ texts: [String], question: String) throws -> [DecisionYesNo] {
        try serialized { try model.yesNo(texts: texts, question: question) }
    }

    public func candidateProbabilities(_ model: DecisionModel, text: String, questions: [String], options: [String],
                                       optionLabels: [String]? = nil, choiceKeys: [String]? = nil) throws
        -> (probabilities: [[Float]], inputTokens: Int)
    {
        try serialized {
            try model.candidateProbabilities(text: text, questions: questions, options: options,
                                             optionLabels: optionLabels, choiceKeys: choiceKeys)
        }
    }

    private func serialized<T>(_ body: () throws -> T) throws -> T {
        run.lock(); defer { run.unlock() }
        return try body()
    }

    private func withState<T>(_ body: () -> T) -> T {
        state.lock(); defer { state.unlock() }
        return body()
    }
}
