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
    /// docs/LOGPROBS_PLAN.md Phase 2: per-row opt-in for the batched/
    /// continuous decode paths. `false`/`0` (the default) costs this row
    /// nothing extra — no log-softmax is ever computed for a row that does
    /// not set this, even when other rows sharing its epoch/cohort do.
    public let wantLogprobs: Bool
    public let topLogprobs: Int

    public init(
        messages: [[String: String]],
        params: SamplingParams = .greedy,
        maxTokens: Int = TokenBudget.unlimited,
        contextLimit: Int? = nil,
        promptTemplateOverride: String? = nil,
        useSpeculative: Bool? = nil,
        usePrefixCache: Bool = true,
        wantLogprobs: Bool = false,
        topLogprobs: Int = 0
    ) {
        self.messages = messages
        self.params = params
        self.maxTokens = maxTokens
        self.contextLimit = contextLimit
        self.promptTemplateOverride = promptTemplateOverride
        self.useSpeculative = useSpeculative
        self.usePrefixCache = usePrefixCache
        self.wantLogprobs = wantLogprobs
        self.topLogprobs = topLogprobs
    }

    /// A copy with the token ceiling replaced by a resolved one. The batch entry
    /// points resolve before admitting a row, because the batcher compares
    /// `generated >= maxTokens` directly and an unresolved sentinel would
    /// terminate the row before its first token.
    public func withMaxTokens(_ resolved: Int) -> BatchGenRequest {
        BatchGenRequest(
            messages: messages, params: params, maxTokens: resolved,
            contextLimit: contextLimit, promptTemplateOverride: promptTemplateOverride,
            useSpeculative: useSpeculative, usePrefixCache: usePrefixCache,
            wantLogprobs: wantLogprobs, topLogprobs: topLogprobs)
    }
}
