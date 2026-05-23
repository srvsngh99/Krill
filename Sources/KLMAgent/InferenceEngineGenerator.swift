import Foundation
import KLMEngine
import KLMSampler

/// An `OperatorGenerator` backed by a loaded `KLMEngine.InferenceEngine`.
///
/// One instance binds to one already-loaded model; the operator agent
/// is expected to use a single small router model per session
/// (`AGENT_ROUTER_MODEL`, defined in sub-PR C). Each `generate(messages:)`
/// call runs one full turn (prefill + decode to EOS or `maxTokens`),
/// aggregates every token into the assistant text, and returns the
/// total along with the decoded-token count.
///
/// Sampling defaults to greedy: the operator agent is a tool-using
/// reasoner, not a creative writer, and determinism makes stuck-
/// detection / regression-test behavior predictable.
public final class InferenceEngineGenerator: OperatorGenerator, @unchecked Sendable {
    private let engine: InferenceEngine
    private let sampling: SamplingParams
    private let maxTokens: Int

    public init(
        engine: InferenceEngine,
        sampling: SamplingParams = .greedy,
        maxTokens: Int = 1024
    ) {
        self.engine = engine
        self.sampling = sampling
        self.maxTokens = maxTokens
    }

    public func generate(messages: [[String: String]]) async throws -> OperatorTurn {
        let (stream, _) = engine.generate(
            messages: messages,
            params: sampling,
            maxTokens: maxTokens)
        var text = ""
        var tokenCount = 0
        for await event in stream {
            if event.isEnd { break }
            text += event.text
            tokenCount += 1
        }
        return OperatorTurn(text: text, tokenCount: tokenCount)
    }
}
