import Foundation
import KrillSampler

public enum EngineError: Error, CustomStringConvertible {
    case modelNotLoaded

    public var description: String {
        switch self {
        case .modelNotLoaded:
            return "No model loaded. Call load() first."
        }
    }
}

/// One row's request for ``InferenceEngine/generateBatched(_:)``.
public struct BatchGenRequest: Sendable {
    public let messages: [[String: String]]
    public let params: SamplingParams
    public let maxTokens: Int
    public let contextLimit: Int?
    public let promptTemplateOverride: String?
    public let useSpeculative: Bool?
    public let usePrefixCache: Bool

    public init(
        messages: [[String: String]],
        params: SamplingParams = .greedy,
        maxTokens: Int = TokenBudget.unlimited,
        contextLimit: Int? = nil,
        promptTemplateOverride: String? = nil,
        useSpeculative: Bool? = nil,
        usePrefixCache: Bool = true
    ) {
        self.messages = messages
        self.params = params
        self.maxTokens = maxTokens
        self.contextLimit = contextLimit
        self.promptTemplateOverride = promptTemplateOverride
        self.useSpeculative = useSpeculative
        self.usePrefixCache = usePrefixCache
    }

    /// A copy with the token ceiling replaced by a resolved one. The batch entry
    /// points resolve before admitting a row, because the batcher compares
    /// `generated >= maxTokens` directly and an unresolved sentinel would
    /// terminate the row before its first token.
    public func withMaxTokens(_ resolved: Int) -> BatchGenRequest {
        BatchGenRequest(
            messages: messages, params: params, maxTokens: resolved,
            contextLimit: contextLimit, promptTemplateOverride: promptTemplateOverride,
            useSpeculative: useSpeculative, usePrefixCache: usePrefixCache)
    }
}
