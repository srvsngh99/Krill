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
