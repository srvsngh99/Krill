import Foundation
import MLX
import KrillCore

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
    public func load(directory: URL) async throws -> DecisionModel {
        if let current = withState({ loadedDir?.standardizedFileURL == directory.standardizedFileURL ? model : nil }) {
            return current
        }
        MLXMemoryConfig.apply(defaultMB: EmbeddingEngine.defaultCacheLimitMB)
        let loaded = try await DecisionModel.load(directory: directory)
        withState { model = loaded; loadedDir = directory }
        return loaded
    }

    public func choose(_ model: DecisionModel, _ texts: [String], options: [String], question: String?,
                       allowNone: Bool) throws -> [DecisionChoice] {
        try serialized { try model.choose(texts, options: options, question: question, allowNone: allowNone) }
    }

    public func yesNo(_ model: DecisionModel, _ texts: [String], question: String) throws -> [DecisionYesNo] {
        try serialized { try model.yesNo(texts: texts, question: question) }
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
