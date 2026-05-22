import KLMCore

/// A single token event emitted during generation.
public struct TokenEvent: Sendable {
    /// The generated token ID.
    public let tokenId: Int

    /// The decoded text for this token.
    public let text: String

    /// Time elapsed since generation started, in seconds.
    public let elapsed: Double

    /// Whether this is the final token (EOS or max_tokens reached).
    public let isEnd: Bool

    public init(tokenId: Int, text: String, elapsed: Double, isEnd: Bool = false) {
        self.tokenId = tokenId
        self.text = text
        self.elapsed = elapsed
        self.isEnd = isEnd
    }
}

/// Speculative-decode telemetry. Populated only when the decode loop ran the
/// spec path; nil when spec was disabled, declined (non-greedy, int8 KV,
/// penalties), or unavailable (no draft model).
public struct SpeculativeStats: Sendable {
    /// Total rounds of draft+verify executed.
    public let rounds: Int
    /// Total tokens emitted by the spec path (includes bonus + rejection-replacement).
    public let acceptedTokens: Int
    /// Final adaptive K (proposals per round at the end of generation).
    public let finalK: Int
    /// Rolling acceptance rate (window=16) at the end of generation, in [0, 1].
    public let acceptanceRate: Double

    public init(rounds: Int, acceptedTokens: Int, finalK: Int, acceptanceRate: Double) {
        self.rounds = rounds
        self.acceptedTokens = acceptedTokens
        self.finalK = finalK
        self.acceptanceRate = acceptanceRate
    }
}

/// Summary statistics for a completed generation.
public struct GenerationStats: Sendable {
    /// Number of tokens in the prompt (prefill).
    public let promptTokens: Int

    /// Number of tokens generated (decode).
    public let generatedTokens: Int

    /// Time for prefill phase, in seconds.
    public let prefillTime: Double

    /// Time for decode phase, in seconds.
    public let decodeTime: Double

    /// Speculative decoding telemetry (nil when spec path did not run).
    public let speculative: SpeculativeStats?

    /// Mixture-of-experts routing telemetry (nil unless the loaded model
    /// is a native MoE runtime).
    public let moe: MoEUtilization?

    public init(
        promptTokens: Int,
        generatedTokens: Int,
        prefillTime: Double,
        decodeTime: Double,
        speculative: SpeculativeStats? = nil,
        moe: MoEUtilization? = nil
    ) {
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.prefillTime = prefillTime
        self.decodeTime = decodeTime
        self.speculative = speculative
        self.moe = moe
    }

    /// Prefill throughput.
    public var prefillTokensPerSecond: Double {
        promptTokens > 0 && prefillTime > 0 ? Double(promptTokens) / prefillTime : 0
    }

    /// Decode throughput.
    public var decodeTokensPerSecond: Double {
        generatedTokens > 0 && decodeTime > 0 ? Double(generatedTokens) / decodeTime : 0
    }

    /// Time to first token (prefill duration).
    public var ttft: Double { prefillTime }

    /// Total generation time.
    public var totalTime: Double { prefillTime + decodeTime }
}
