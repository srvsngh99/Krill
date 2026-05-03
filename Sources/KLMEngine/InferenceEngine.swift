import Foundation
import MLX
import KLMCore
import KLMCache
import KLMTokenizer
import KLMSampler

/// Orchestrates model loading, tokenization, and the prefill + decode loop.
///
/// The engine owns the model, tokenizer, and sampler. It provides a streaming
/// generation interface via AsyncStream<TokenEvent>.
public final class InferenceEngine: @unchecked Sendable {
    private var model: LlamaForCausalLM?
    private var tokenizer: KLMTokenizer?
    private var config: LlamaConfig?
    private let modelDirectory: URL

    public init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
    }

    /// Load the model and tokenizer from disk.
    public func load() async throws {
        // Load config
        let cfg = try loadConfig(from: modelDirectory)
        self.config = cfg

        // Build model
        let llm = LlamaForCausalLM(cfg)

        // Load weights (handles quantization if config specifies it)
        try loadWeights(
            into: llm,
            from: modelDirectory,
            quantization: cfg.quantization
        )

        self.model = llm

        // Load tokenizer
        self.tokenizer = try await KLMTokenizer(from: modelDirectory)
    }

    /// Check if the model is loaded and ready.
    public var isLoaded: Bool {
        model != nil && tokenizer != nil
    }

    /// Generate tokens from a prompt, streaming results.
    ///
    /// - Parameters:
    ///   - prompt: The user's input text
    ///   - systemPrompt: Optional system prompt
    ///   - params: Sampling parameters
    ///   - maxTokens: Maximum tokens to generate
    /// - Returns: AsyncStream of TokenEvents
    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        params: SamplingParams = .greedy,
        maxTokens: Int = 512
    ) -> (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?) {
        guard let model, let tokenizer, let config else {
            let emptyStream = AsyncStream<TokenEvent> { $0.finish() }
            return (emptyStream, { nil })
        }

        // Build messages and encode
        var messages: [[String: String]] = []
        if let sys = systemPrompt {
            messages.append(["role": "system", "content": sys])
        }
        messages.append(["role": "user", "content": prompt])

        let formatted = tokenizer.applyChatTemplate(messages: messages)
        let promptTokens = tokenizer.encode(formatted)

        let sampler = Sampler(params: params)
        let eosId = tokenizer.eosTokenId
        let numLayers = config.numHiddenLayers

        // Shared stats - populated after generation completes
        let statsHolder = StatsHolder()

        // Capture model/tokenizer as nonisolated(unsafe) to suppress Sendable warnings.
        // Safe because KrillLM is single-request, single-user by design (v1.0).
        nonisolated(unsafe) let capturedModel = model
        nonisolated(unsafe) let capturedTokenizer = tokenizer

        let stream = AsyncStream<TokenEvent> { continuation in
            Task { [statsHolder] in
                let startTime = CFAbsoluteTimeGetCurrent()
                var generatedCount = 0
                var prefillDuration: Double = 0

                // Create KV caches
                let caches = makeKVCaches(numLayers: numLayers)

                // -- Prefill --
                let inputArray = MLXArray(promptTokens.map { Int32($0) })
                    .reshaped(1, promptTokens.count)
                let prefillLogits = capturedModel(inputArray, caches: caches)
                MLX.eval(prefillLogits)
                prefillDuration = CFAbsoluteTimeGetCurrent() - startTime

                // Sample first token from prefill logits (last position)
                var nextToken = sampler.sample(prefillLogits)

                // -- Decode loop --
                let decodeStart = CFAbsoluteTimeGetCurrent()

                while generatedCount < maxTokens {
                    // Check for EOS
                    if nextToken == eosId {
                        let event = TokenEvent(
                            tokenId: nextToken,
                            text: "",
                            elapsed: CFAbsoluteTimeGetCurrent() - startTime,
                            isEnd: true
                        )
                        continuation.yield(event)
                        break
                    }

                    // Decode the token text
                    let tokenText = capturedTokenizer.decode(token: nextToken)
                    let event = TokenEvent(
                        tokenId: nextToken,
                        text: tokenText,
                        elapsed: CFAbsoluteTimeGetCurrent() - startTime
                    )
                    continuation.yield(event)
                    generatedCount += 1

                    // Forward pass for next token
                    let tokenInput = MLXArray([Int32(nextToken)]).reshaped(1, 1)
                    let logits = capturedModel(tokenInput, caches: caches)
                    MLX.eval(logits)

                    nextToken = sampler.sample(logits)
                }

                let decodeDuration = CFAbsoluteTimeGetCurrent() - decodeStart

                // If we hit max tokens without EOS, emit final event
                if generatedCount >= maxTokens {
                    continuation.yield(TokenEvent(
                        tokenId: -1, text: "", elapsed: CFAbsoluteTimeGetCurrent() - startTime,
                        isEnd: true
                    ))
                }

                // Store stats
                statsHolder.stats = GenerationStats(
                    promptTokens: promptTokens.count,
                    generatedTokens: generatedCount,
                    prefillTime: prefillDuration,
                    decodeTime: decodeDuration
                )

                continuation.finish()
            }
        }

        return (stream, { statsHolder.stats })
    }

    /// Unload the model from memory and release resources.
    public func unload() {
        model = nil
        tokenizer = nil
        config = nil
    }
}

// MARK: - Internal Helpers

/// Thread-safe holder for generation statistics.
private final class StatsHolder: @unchecked Sendable {
    var stats: GenerationStats?
}
