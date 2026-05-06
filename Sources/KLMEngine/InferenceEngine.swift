import Foundation
import os
import MLX
import KLMCore
import KLMCache
import KLMTokenizer
import KLMSampler

/// Orchestrates model loading, tokenization, and the prefill + decode loop.
///
/// The engine owns the model, tokenizer, and sampler. It provides a streaming
/// generation interface via AsyncStream<TokenEvent>.
/// Supports all model families: Llama, Qwen, Mistral, Gemma, Phi.
///
/// Performance features wired in:
/// - Prefix cache: skips prefill for cached token prefixes (TTFT <100ms)
/// - Speculative decoding: draft model proposes K tokens per step (1.5-3x decode)
public final class InferenceEngine: @unchecked Sendable {
    private var loadedModel: LoadedModel?
    private var tokenizer: KLMTokenizer?
    private var modelDirectory: URL

    /// Prefix cache for reusing KV state across requests.
    public let prefixCache: PrefixCache

    /// Optional speculative decoder (loaded separately via loadDraftModel).
    private var specDecoder: SpeculativeDecoder?

    /// True while a model swap is in progress. Checked by server to return 503.
    private let _isSwapping = OSAllocatedUnfairLock(initialState: false)
    public var isSwapping: Bool { _isSwapping.withLock { $0 } }

    /// The detected model family (nil if not loaded).
    public var family: String? { loadedModel?.family }

    /// The loaded model's directory name (useful for display/status).
    public var modelName: String? { isLoaded ? modelDirectory.lastPathComponent : nil }

    /// Model identifier for cache keying.
    private var modelId: String { modelDirectory.lastPathComponent }

    /// Timestamp when the model was loaded (nil if not loaded).
    public private(set) var loadedAt: Date?

    public init(modelDirectory: URL, prefixCache: PrefixCache? = nil) {
        self.modelDirectory = modelDirectory
        self.prefixCache = prefixCache ?? PrefixCache()
    }

    /// Load the model and tokenizer from disk.
    /// Auto-detects model family from config.json.
    public func load() async throws {
        let loaded = try loadModel(from: modelDirectory)
        self.loadedModel = loaded
        self.tokenizer = try await KLMTokenizer(from: modelDirectory)
        self.loadedAt = Date()
    }

    /// Swap the current model for a new one at the given directory.
    /// Loads the new model first — if loading fails, the previous model remains active.
    public func swap(modelDirectory newDir: URL) async throws {
        _isSwapping.withLock { $0 = true }
        defer { _isSwapping.withLock { $0 = false } }

        // Load new model into temporaries before touching current state.
        let newModel = try loadModel(from: newDir)
        let newTokenizer = try await KLMTokenizer(from: newDir)

        // Success — now swap atomically.
        unload()
        self.modelDirectory = newDir
        self.loadedModel = newModel
        self.tokenizer = newTokenizer
        self.loadedAt = Date()
    }

    /// Load a draft model for speculative decoding.
    ///
    /// - Parameter draftDirectory: Path to the draft model weights
    public func loadDraftModel(from draftDirectory: URL) throws {
        guard let targetModel = loadedModel else {
            throw EngineError.modelNotLoaded
        }
        let draft = try loadModel(from: draftDirectory)
        self.specDecoder = SpeculativeDecoder(
            targetModel: targetModel,
            draftModel: draft
        )
    }

    /// Check if the model is loaded and ready.
    public var isLoaded: Bool {
        loadedModel != nil && tokenizer != nil
    }

    /// Whether speculative decoding is available.
    public var hasSpeculativeDecoding: Bool {
        specDecoder != nil
    }

    /// Generate tokens from a single prompt string (convenience wrapper).
    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        params: SamplingParams = .greedy,
        maxTokens: Int = 512,
        useSpeculative: Bool = false,
        usePrefixCache: Bool = true
    ) -> (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?) {
        var messages: [[String: String]] = []
        if let sys = systemPrompt {
            messages.append(["role": "system", "content": sys])
        }
        messages.append(["role": "user", "content": prompt])
        return generate(messages: messages, params: params, maxTokens: maxTokens,
                        useSpeculative: useSpeculative, usePrefixCache: usePrefixCache)
    }

    /// Generate tokens from a full conversation history, streaming results.
    ///
    /// - Parameters:
    ///   - messages: Array of `["role": ..., "content": ...]` messages (full chat history)
    ///   - params: Sampling parameters
    ///   - maxTokens: Maximum tokens to generate
    ///   - useSpeculative: Enable speculative decoding (requires draft model loaded)
    ///   - usePrefixCache: Enable prefix cache lookup/store
    /// - Returns: AsyncStream of TokenEvents + stats accessor
    public func generate(
        messages: [[String: String]],
        params: SamplingParams = .greedy,
        maxTokens: Int = 512,
        useSpeculative: Bool = false,
        usePrefixCache: Bool = true
    ) -> (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?) {
        guard let loadedModel, let tokenizer else {
            let emptyStream = AsyncStream<TokenEvent> { $0.finish() }
            return (emptyStream, { nil })
        }

        let formatted = tokenizer.applyChatTemplate(messages: messages)
        let promptTokens = tokenizer.encodeWithoutExtraBOS(formatted)

        guard !promptTokens.isEmpty else {
            let emptyStream = AsyncStream<TokenEvent> { continuation in
                continuation.yield(TokenEvent(tokenId: 0, text: "", elapsed: 0, isEnd: true))
                continuation.finish()
            }
            return (emptyStream, { nil })
        }

        let sampler = Sampler(params: params)
        let eosId = tokenizer.eosTokenId
        let numLayers = loadedModel.numLayers
        let forwardFn = loadedModel.forward
        let statsHolder = StatsHolder()

        // Capture state for async Task
        nonisolated(unsafe) let capturedForward = forwardFn
        nonisolated(unsafe) let capturedTokenizer = tokenizer
        nonisolated(unsafe) let capturedPrefixCache = self.prefixCache
        nonisolated(unsafe) let capturedSpecDecoder = self.specDecoder
        let capturedModelId = self.modelId
        let shouldSpec = useSpeculative && specDecoder != nil

        let stream = AsyncStream<TokenEvent> { continuation in
            Task { [statsHolder] in
                let startTime = CFAbsoluteTimeGetCurrent()
                var generatedCount = 0
                var prefillDuration: Double = 0
                var cacheHit = false

                // Create KV caches
                let caches = makeKVCaches(numLayers: numLayers)

                // -- Prefix Cache Lookup --
                // Only accept FULL prefix hits (cached tokens == prompt tokens).
                // Partial hits are unsafe: the causal mask is built for the new
                // span length only, but attention keys include the restored prefix,
                // causing shape mismatch or incorrect masking. Until cache-aware
                // mask construction is implemented, we skip partial hits.
                if usePrefixCache {
                    if let hit = capturedPrefixCache.lookup(
                        tokens: promptTokens, modelId: capturedModelId
                    ), !hit.keys.isEmpty, hit.prefixLength == promptTokens.count {
                        // Full exact hit — restore KV for all prompt tokens.
                        for (i, cache) in caches.enumerated() {
                            if i < hit.keys.count, let k = hit.keys[i].first,
                               i < hit.values.count, let v = hit.values[i].first {
                                cache.restore(keys: k, values: v)
                            }
                        }
                        cacheHit = true
                    }
                }

                // -- Prefill --
                // On a full cache hit we already have KV for the entire prompt.
                // We truncate the last position and re-forward that single token
                // to get logits without duplicating a KV entry.
                // On a miss we forward the entire prompt normally.
                let tokensToProcess: [Int]
                if cacheHit {
                    let trimmedLength = max(0, promptTokens.count - 1)
                    for cache in caches {
                        cache.truncate(to: trimmedLength)
                    }
                    tokensToProcess = [promptTokens.last!]
                } else {
                    tokensToProcess = promptTokens
                }

                let inputArray = MLXArray(tokensToProcess.map { Int32($0) })
                    .reshaped(1, tokensToProcess.count)
                let prefillLogits = capturedForward(inputArray, caches)
                MLX.eval(prefillLogits)
                prefillDuration = CFAbsoluteTimeGetCurrent() - startTime

                // -- Store in prefix cache (write-behind) --
                // We cache KV for the full prompt (all tokens that have been
                // forwarded). On the next request with the same prefix, the
                // restored KV will cover all prompt tokens, and the "full cache
                // hit" path above trims the last token and re-forwards it.
                if usePrefixCache && !cacheHit && promptTokens.count >= 32 {
                    var snapshotKeys: [[MLXArray]] = []
                    var snapshotValues: [[MLXArray]] = []
                    for cache in caches {
                        if let snap = cache.snapshot() {
                            snapshotKeys.append([snap.keys])
                            snapshotValues.append([snap.values])
                        }
                    }
                    if !snapshotKeys.isEmpty {
                        capturedPrefixCache.store(
                            tokens: promptTokens,
                            modelId: capturedModelId,
                            keys: snapshotKeys,
                            values: snapshotValues
                        )
                    }
                }

                // Sample first token
                var nextToken = sampler.sample(prefillLogits)

                // -- Decode loop --
                let decodeStart = CFAbsoluteTimeGetCurrent()

                if shouldSpec, let specDec = capturedSpecDecoder {
                    // === Speculative Decoding Path ===
                    let draftCaches = makeKVCaches(numLayers: specDec.totalRounds == 0 ? numLayers : numLayers)

                    // Emit the first token sampled from prefill logits — specDec.step
                    // only returns tokens *after* lastToken, so we must yield it here.
                    if nextToken == eosId {
                        continuation.yield(TokenEvent(
                            tokenId: nextToken, text: "",
                            elapsed: CFAbsoluteTimeGetCurrent() - startTime, isEnd: true))
                    } else {
                        let firstText = capturedTokenizer.decode(token: nextToken)
                        continuation.yield(TokenEvent(
                            tokenId: nextToken, text: firstText,
                            elapsed: CFAbsoluteTimeGetCurrent() - startTime))
                        generatedCount += 1
                    }

                    while generatedCount < maxTokens && nextToken != eosId {
                        // Speculative step: get multiple tokens at once
                        let accepted = specDec.step(
                            lastToken: nextToken,
                            targetCaches: caches,
                            draftCaches: draftCaches
                        )

                        for token in accepted {
                            if token == eosId {
                                continuation.yield(TokenEvent(
                                    tokenId: token, text: "",
                                    elapsed: CFAbsoluteTimeGetCurrent() - startTime, isEnd: true))
                                break
                            }

                            let text = capturedTokenizer.decode(token: token)
                            continuation.yield(TokenEvent(
                                tokenId: token, text: text,
                                elapsed: CFAbsoluteTimeGetCurrent() - startTime))
                            generatedCount += 1

                            if generatedCount >= maxTokens { break }
                        }

                        nextToken = accepted.last ?? eosId
                        if nextToken == eosId || generatedCount >= maxTokens { break }
                    }
                } else {
                    // === Standard Decode Path ===
                    while generatedCount < maxTokens {
                        if nextToken == eosId {
                            continuation.yield(TokenEvent(
                                tokenId: nextToken, text: "",
                                elapsed: CFAbsoluteTimeGetCurrent() - startTime, isEnd: true))
                            break
                        }

                        let tokenText = capturedTokenizer.decode(token: nextToken)
                        continuation.yield(TokenEvent(
                            tokenId: nextToken, text: tokenText,
                            elapsed: CFAbsoluteTimeGetCurrent() - startTime))
                        generatedCount += 1

                        let tokenInput = MLXArray([Int32(nextToken)]).reshaped(1, 1)
                        let logits = capturedForward(tokenInput, caches)
                        MLX.eval(logits)
                        nextToken = sampler.sample(logits)
                    }
                }

                let decodeDuration = CFAbsoluteTimeGetCurrent() - decodeStart

                if generatedCount >= maxTokens {
                    continuation.yield(TokenEvent(
                        tokenId: -1, text: "",
                        elapsed: CFAbsoluteTimeGetCurrent() - startTime, isEnd: true))
                }

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
        loadedModel = nil
        tokenizer = nil
        specDecoder = nil
        loadedAt = nil
    }
}

// MARK: - Errors

public enum EngineError: Error, CustomStringConvertible {
    case modelNotLoaded

    public var description: String {
        switch self {
        case .modelNotLoaded: return "No model loaded. Call load() first."
        }
    }
}

// MARK: - Internal Helpers

/// Thread-safe holder for generation statistics.
private final class StatsHolder: @unchecked Sendable {
    var stats: GenerationStats?
}
