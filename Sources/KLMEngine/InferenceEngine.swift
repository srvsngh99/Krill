import Foundation
import os
import MLX
import KLMCore
import KLMCache
import KLMTokenizer
import KLMSampler
import KLMRegistry

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

    /// Display name of the loaded draft model (alias or directory name).
    /// Nil when no draft is loaded.
    public private(set) var draftModelName: String?

    /// When true, requests default to `useSpeculative: true` once a draft
    /// model is loaded. Set by `loadDraftModel()`. Generates that explicitly
    /// pass `useSpeculative: false` still skip the spec path (used by tests
    /// and by paths that must match a non-spec reference).
    private var autoUseSpec: Bool = false

    /// True while a model swap is in progress. Checked by server to return 503.
    private let _isSwapping = OSAllocatedUnfairLock(initialState: false)
    public var isSwapping: Bool { _isSwapping.withLock { $0 } }

    /// The detected model family (nil if not loaded).
    public var family: String? { loadedModel?.family }

    /// The loaded model's directory name (useful for display/status).
    public var modelName: String? { isLoaded ? modelDirectory.lastPathComponent : nil }

    /// Filesystem path to the loaded model directory.
    /// Returns nil if no model is currently loaded.
    public var modelDirectoryPath: String? { isLoaded ? modelDirectory.path : nil }

    /// The loaded model's declared capability set. Combines the family's
    /// static registry entry (`ModelCapabilities.capabilities(for:)`)
    /// with checkpoint-level facts that only the loader knows (e.g. a
    /// `gemma4_text` checkpoint declares family=gemma4 but ships no
    /// `vision_config`, so vision/audio must be revoked).
    public var capabilities: Set<Capability> {
        guard let loaded = loadedModel,
              let family = ModelFamily(rawValue: loaded.family) else {
            return []
        }
        var caps = ModelCapabilities.capabilities(for: family)
        // Multimodal capabilities also require the checkpoint to carry
        // the corresponding sub-config: a text-only Gemma 4 dump has
        // `multimodalForward == nil` and must NOT advertise vision or
        // audio even though its family does in general. This keeps
        // server pre-generation media gating honest.
        if loaded.multimodalForward == nil {
            caps.remove(.visionInput)
            caps.remove(.audioInput)
        }
        return caps
    }

    /// Whether the loaded model can natively handle image input via the
    /// engine. Sourced from `capabilities.contains(.visionInput)`.
    public var supportsNativeImage: Bool {
        capabilities.contains(.visionInput)
    }

    /// Whether the loaded model can handle audio input. Audio runs
    /// exclusively on the native Swift+MLX USM path (the mlx-vlm bridge
    /// was retired in WS6 Step 4 after native was validated and
    /// benchmarked faster than Ollama; see
    /// `docs/NATIVE_GEMMA4_AUDIO_PLAN.md`). Sourced from
    /// `capabilities.contains(.audioInput)`.
    public var supportsAudio: Bool {
        capabilities.contains(.audioInput)
    }

    /// True when the loaded model can run audio. Audio is always native now;
    /// this is exactly `supportsAudio`. Retained as stable API for the CLI
    /// and server, which gate audio acceptance on it.
    public var canUseNativeAudio: Bool { supportsAudio }

    /// A generation stream that surfaces a hard pre-generation failure (e.g.
    /// native audio decode error) so it is **never swallowed**. The message
    /// is a NON-terminal content event followed by a SEPARATE terminal
    /// event: consumers that break on `isEnd` before reading `text` (CLI,
    /// non-streaming server loops) still receive the message in the prior
    /// event, and streaming consumers emit it as a content chunk before the
    /// final chunk. (PR #21 rereview P1b.)
    static func mediaErrorStream(
        _ message: String
    ) -> (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?) {
        let stream = AsyncStream<TokenEvent> { continuation in
            continuation.yield(TokenEvent(
                tokenId: 0, text: message, elapsed: 0, isEnd: false))
            continuation.yield(TokenEvent(
                tokenId: 0, text: "", elapsed: 0, isEnd: true))
            continuation.finish()
        }
        return (stream, { nil })
    }

    /// Model identifier for cache keying.
    private var modelId: String { modelDirectory.lastPathComponent }

    /// The set of token ids that terminate generation. Reads
    /// `generation_config.json`'s `eos_token_id` (scalar or array) from the
    /// model directory and unions it with the tokenizer's single EOS so
    /// multi-stop models (e.g. Gemma 4: `[1, 106, 50]`) halt correctly
    /// instead of leaking `<end_of_turn>`. Falls back to just the tokenizer
    /// EOS if the file is absent or unparseable.
    static func stopTokenIds(modelDirectory: URL, tokenizerEOS: Int) -> Set<Int> {
        var ids: Set<Int> = [tokenizerEOS]
        let url = modelDirectory.appendingPathComponent("generation_config.json")
        if let data = try? Data(contentsOf: url),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let eos = obj["eos_token_id"] {
            if let single = eos as? Int {
                ids.insert(single)
            } else if let many = eos as? [Int] {
                ids.formUnion(many)
            }
        }
        return ids
    }

    /// Timestamp when the model was loaded (nil if not loaded).
    public private(set) var loadedAt: Date?

    /// KV cache dtype: "fp16" (default) or "int8" (quantized). Read from arg or
    /// `KRILL_KV_CACHE_DTYPE` env var.
    private let kvCacheDtype: String

    public init(modelDirectory: URL, prefixCache: PrefixCache? = nil, kvCacheDtype: String? = nil) {
        self.modelDirectory = modelDirectory
        self.prefixCache = prefixCache ?? PrefixCache()
        if let dtype = kvCacheDtype {
            self.kvCacheDtype = dtype
        } else {
            self.kvCacheDtype = ProcessInfo.processInfo.environment["KRILL_KV_CACHE_DTYPE"] ?? "fp16"
        }
    }

    /// Whether this engine is configured to use int8 quantized KV cache.
    public var usesQuantizedKVCache: Bool { kvCacheDtype == "int8" }

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

    /// Load a draft model for speculative decoding. Once loaded, subsequent
    /// `generate()` calls will use the spec path by default (greedy-only;
    /// the engine falls back to standard decode for non-greedy / penalized
    /// requests).
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
        self.draftModelName = draftDirectory.lastPathComponent
        self.autoUseSpec = true
    }

    /// Check if the model is loaded and ready.
    public var isLoaded: Bool {
        loadedModel != nil && tokenizer != nil
    }

    /// Whether speculative decoding is available.
    public var hasSpeculativeDecoding: Bool {
        specDecoder != nil
    }

    /// Toggle the auto-spec opt-in. When false, even with a draft loaded,
    /// `generate()` will not use spec unless the caller explicitly passes
    /// `useSpeculative: true`. Used by parity tests that need a non-spec
    /// reference output from the same engine instance.
    public func setAutoUseSpec(_ enabled: Bool) {
        self.autoUseSpec = enabled && specDecoder != nil
    }

    /// True if the engine will currently take the spec path on a greedy,
    /// non-penalized, fp16-cache request.
    public var willUseSpeculativeByDefault: Bool { autoUseSpec }

    /// Prepend `<|image|>` and/or `<|audio|>` placeholder runs to the first
    /// user message's content. The vision encoder produces N soft tokens, so
    /// we insert N copies of `<|image|>` for `injectEmbeddings` to fill.
    /// Returns messages unchanged when no media is provided.
    func injectMediaPlaceholders(
        into messages: [[String: String]],
        imageData: Data?,
        audioData: Data?,
        audioTokenCount: Int = 1
    ) -> [[String: String]] {
        guard imageData != nil || audioData != nil else { return messages }
        var prefix = ""
        if let img = imageData {
            prefix += String(repeating: "<|image|>", count: computeImageTokenCount(imageData: img))
        }
        if audioData != nil {
            // Native audio scatters `audioTokenCount` encoder frames into
            // exactly that many `<|audio|>` positions (mirrors the image
            // path). Bridge fallback uses a single placeholder.
            prefix += String(repeating: "<|audio|>", count: max(1, audioTokenCount))
        }
        var result = messages
        if let firstUserIndex = result.firstIndex(where: { $0["role"] == "user" }) {
            let existing = result[firstUserIndex]["content"] ?? ""
            result[firstUserIndex]["content"] = prefix + existing
        } else {
            result.append(["role": "user", "content": prefix])
        }
        return result
    }

    /// Compute the number of soft tokens the vision encoder will produce for an image.
    /// This depends on the preprocessed image size.
    func computeImageTokenCount(imageData: Data) -> Int {
        guard let tensor = try? preprocessImage(imageData) else { return 256 }
        let H = tensor.dim(2)
        let W = tensor.dim(3)
        let patchSize = 16
        let poolingKernel = 3
        let pH = H / patchSize
        let pW = W / patchSize
        let numPatches = pH * pW
        return numPatches / (poolingKernel * poolingKernel)
    }

    /// Generate tokens from a single prompt string (convenience wrapper).
    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        params: SamplingParams = .greedy,
        maxTokens: Int = 512,
        useSpeculative: Bool? = nil,
        usePrefixCache: Bool = true,
        imageData: Data? = nil,
        audioData: Data? = nil,
        contextLimit: Int? = nil
    ) -> (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?) {
        var messages: [[String: String]] = []
        if let sys = systemPrompt {
            messages.append(["role": "system", "content": sys])
        }
        messages.append(["role": "user", "content": prompt])
        // Placeholder run is injected by generate(messages:) so the chat path
        // and the prompt path agree on tokenization.
        return generate(messages: messages, params: params, maxTokens: maxTokens,
                        useSpeculative: useSpeculative, usePrefixCache: usePrefixCache,
                        imageData: imageData, audioData: audioData,
                        contextLimit: contextLimit)
    }

    /// Generate tokens from a full conversation history, streaming results.
    ///
    /// - Parameters:
    ///   - messages: Array of `["role": ..., "content": ...]` messages (full chat history)
    ///   - params: Sampling parameters
    ///   - maxTokens: Maximum tokens to generate
    ///   - useSpeculative: Tri-state spec opt-in. `nil` defers to the engine
    ///     default (`autoUseSpec`, set by `loadDraftModel`). `false` is an
    ///     explicit opt-out honored even when a draft is loaded; `true` is
    ///     an explicit opt-in still subject to greedy / non-int8 / no-penalty
    ///     guards.
    ///   - usePrefixCache: Enable prefix cache lookup/store
    /// - Returns: AsyncStream of TokenEvents + stats accessor
    public func generate(
        messages: [[String: String]],
        params: SamplingParams = .greedy,
        maxTokens: Int = 512,
        useSpeculative: Bool? = nil,
        usePrefixCache: Bool = true,
        imageData: Data? = nil,
        audioData: Data? = nil,
        contextLimit: Int? = nil
    ) -> (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?) {
        guard let loadedModel, let tokenizer else {
            let emptyStream = AsyncStream<TokenEvent> { $0.finish() }
            return (emptyStream, { nil })
        }

        // Inject media placeholders into the first user message before
        // tokenization so the chat path matches the prompt path. Gemma 4's
        // multimodal forward replaces embeddings at placeholder token positions
        // (image_token_id=258880, audio_token_id=258881); without these tokens
        // the encoder output has nowhere to land and image/audio requests
        // become silently text-only.
        // Native audio: decode the container (WAV/mp3/flac/ogg/m4a) ->
        // log-mel + validity once, up front, so the `<|audio|>` placeholder
        // count matches the encoder frame count.
        // Audio is always native when the model is audio-capable.
        let nativeAudioChosen = audioData != nil && canUseNativeAudio
            && loadedModel.family == "gemma4"
        let nativeAudio: AudioPreprocessor.Features?
        if nativeAudioChosen, let aud = audioData {
            do {
                nativeAudio = try AudioPreprocessor.features(fromAudio: aud)
            } catch {
                // The native path was selected for this audio request but
                // decoding failed. Fail loudly: never silently drop the audio
                // and answer the prompt as if it were text-only.
                return Self.mediaErrorStream(
                    "Error: native audio decode failed: \(error)")
            }
        } else {
            nativeAudio = nil
        }

        let preparedMessages = injectMediaPlaceholders(
            into: messages, imageData: imageData, audioData: audioData,
            audioTokenCount: nativeAudio?.numTokens ?? 1)

        // Use direct token ID path for Gemma4 to avoid decode→re-encode
        // round-trip that loses special tokens (105, 106, 107).
        var promptTokensBuilt: [Int]
        if loadedModel.family == "gemma4" {
            promptTokensBuilt = tokenizer.formatGemma4TokenIds(messages: preparedMessages)
        } else {
            let formatted = tokenizer.applyChatTemplate(messages: preparedMessages)
            promptTokensBuilt = tokenizer.encodeWithoutExtraBOS(formatted)
        }

        // WS-D D4: honor a context-length cap (num_ctx / KRILL_CONTEXT_LENGTH).
        // Keep the most recent `contextLimit` tokens so the latest turn and
        // any tool results survive; warn rather than silently overflow.
        if let limit = contextLimit, limit > 0, promptTokensBuilt.count > limit {
            let dropped = promptTokensBuilt.count - limit
            promptTokensBuilt = Array(promptTokensBuilt.suffix(limit))
            FileHandle.standardError.write(Data(
                "[KrillLM] num_ctx=\(limit): prompt truncated, dropped \(dropped) leading token(s).\n".utf8))
        }
        let promptTokens = promptTokensBuilt

        guard !promptTokens.isEmpty else {
            let emptyStream = AsyncStream<TokenEvent> { continuation in
                continuation.yield(TokenEvent(tokenId: 0, text: "", elapsed: 0, isEnd: true))
                continuation.finish()
            }
            return (emptyStream, { nil })
        }

        let sampler = Sampler(params: params)
        let eosId = tokenizer.eosTokenId
        // Stop on ANY of the model's declared end tokens, not just the
        // tokenizer's single `eos_token`. Gemma 4's generation_config.json
        // declares `eos_token_id: [1, 106, 50]` — the model emits 106
        // (`<end_of_turn>`) to end a turn, not `<eos>` (1). Checking only
        // `eosId` let `<turn|>` leak into native output; the mlx-vlm oracle
        // stops correctly because it honors the full list.
        let stopIds = Self.stopTokenIds(
            modelDirectory: modelDirectory, tokenizerEOS: eosId)
        let numLayers = loadedModel.numLayers
        let forwardFn = loadedModel.forward
        let multimodalFn = loadedModel.multimodalForward
        let statsHolder = StatsHolder()

        // Capture state for async Task
        nonisolated(unsafe) let capturedForward = forwardFn
        nonisolated(unsafe) let capturedMultimodalForward = multimodalFn
        nonisolated(unsafe) let capturedTokenizer = tokenizer
        nonisolated(unsafe) let capturedPrefixCache = self.prefixCache
        let capturedSpecDecoder = self.specDecoder
        let capturedModelId = self.modelId
        // int8 KV is gated to Gemma 4: other model loaders downcast caches to
        // [KVCache] in their forward closures (see ModelLoader.swift), so a
        // [QuantizedKVCache] would crash at first attention. Speculative
        // decoding also assumes fp16 target/draft caches.
        let useInt8KV: Bool = {
            guard self.usesQuantizedKVCache else { return false }
            if loadedModel.family != "gemma4" {
                FileHandle.standardError.write(Data(
                    "[KrillLM] warning: int8 KV cache is supported for Gemma 4 only; falling back to fp16 for family=\(loadedModel.family).\n".utf8))
                return false
            }
            return true
        }()
        // Penalty/mirostat sampling needs a sequential per-step recent-token
        // window, which the multi-token speculative path cannot honor; fall
        // back to the standard decode loop when penalties are active.
        //
        // Spec also requires greedy semantics (the decoder verifies against
        // its own internal greedy sampler); non-greedy without Leviathan
        // rejection sampling would silently diverge from the per-request
        // sampler. Until rejection sampling is implemented, restrict to
        // pure greedy.
        let greedyRequest = params.temperature <= 0 && params.topP >= 1.0
            && params.topK <= 0 && params.minP <= 0
        // `useSpeculative` is tri-state:
        //   nil   -> use the engine's auto opt-in (set true by
        //            `loadDraftModel`; can be cleared with
        //            `setAutoUseSpec(false)`),
        //   false -> explicit opt-out, honored even with a draft loaded,
        //   true  -> explicit opt-in (still subject to the guards below).
        // The explicit-false case is what tests and parity reference paths
        // need: they want to call generate() on the same engine instance
        // and get the non-spec output for comparison.
        let wantsSpec = useSpeculative ?? autoUseSpec
        let shouldSpec = wantsSpec && specDecoder != nil && !useInt8KV
            && !params.penaltiesActive && greedyRequest
        let capturedImageData = imageData
        nonisolated(unsafe) let capturedAudioMel = nativeAudio?.mel
        nonisolated(unsafe) let capturedAudioMask = nativeAudio?.validMask

        // int8 KV + prefix cache coexist via the quantized snapshot path
        // (PrefixCache.lookupQuantized / storeQuantized).
        let effectiveUsePrefixCache = usePrefixCache
        // Bind the prefix-cache key to all non-text conditioning. Without this,
        // a prompt+image request can mis-hit a prior entry computed under a
        // different image (or no image at all) and serve stale KV state.
        let mediaHash: String? = {
            guard imageData != nil || audioData != nil else { return nil }
            var parts: [String] = []
            if let img = imageData { parts.append("img:" + VisionEncoderCache.key(forImageBytes: img)) }
            if let aud = audioData {
                let digest = VisionEncoderCache.key(forImageBytes: aud) // SHA-256 of bytes
                parts.append("aud:" + digest)
            }
            return parts.joined(separator: "|")
        }()

        let stream = AsyncStream<TokenEvent> { continuation in
            Task { [statsHolder] in
                let startTime = CFAbsoluteTimeGetCurrent()
                var generatedCount = 0
                var prefillDuration: Double = 0
                var cacheHit = false

                // Create KV caches. fp16 path uses concrete KVCache so the prefix-cache
                // path (snapshot/restore/truncate) works; int8 uses QuantizedKVCache
                // and its parallel quantized snapshot/restore path.
                let fp16Caches: [KVCache]? = useInt8KV ? nil : makeKVCaches(numLayers: numLayers)
                let int8Caches: [QuantizedKVCache]? = useInt8KV ? makeQuantizedKVCaches(numLayers: numLayers) : nil
                let caches: [KVCacheProtocol] = useInt8KV ? int8Caches! : fp16Caches!

                // -- Prefix Cache Lookup --
                // Only accept FULL prefix hits (cached tokens == prompt tokens).
                // Partial hits are unsafe: the causal mask is built for the new
                // span length only, but attention keys include the restored prefix,
                // causing shape mismatch or incorrect masking. Until cache-aware
                // mask construction is implemented, we skip partial hits.
                if effectiveUsePrefixCache {
                    if let fp16Caches {
                        if let hit = capturedPrefixCache.lookup(
                            tokens: promptTokens, modelId: capturedModelId, mediaHash: mediaHash
                        ), !hit.keys.isEmpty, hit.prefixLength == promptTokens.count {
                            for (i, cache) in fp16Caches.enumerated() {
                                if i < hit.keys.count, let k = hit.keys[i].first,
                                   i < hit.values.count, let v = hit.values[i].first {
                                    cache.restore(keys: k, values: v)
                                }
                            }
                            cacheHit = true
                        }
                    } else if let int8Caches {
                        // Gemma 4 KV sharing leaves a suffix of caches empty
                        // (shared layers reuse the donor's K/V via sharedCache),
                        // so a hit may carry fewer snapshots than caches. Match
                        // the fp16 contract: hit.layers cover caches[0..count].
                        if let hit = capturedPrefixCache.lookupQuantized(
                            tokens: promptTokens, modelId: capturedModelId, mediaHash: mediaHash
                        ), !hit.layers.isEmpty,
                           hit.layers.count <= int8Caches.count,
                           hit.prefixLength == promptTokens.count {
                            for i in 0 ..< hit.layers.count {
                                int8Caches[i].restoreQuantized(hit.layers[i])
                            }
                            cacheHit = true
                        }
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
                    if let fp16Caches {
                        for cache in fp16Caches { cache.truncate(to: trimmedLength) }
                    } else if let int8Caches {
                        for cache in int8Caches { cache.truncate(to: trimmedLength) }
                    }
                    tokensToProcess = [promptTokens.last!]
                } else {
                    tokensToProcess = promptTokens
                }

                let inputArray = MLXArray(tokensToProcess.map { Int32($0) })
                    .reshaped(1, tokensToProcess.count)

                // Multimodal prefill: route image and/or native audio through
                // the Swift encoder pipeline. Image and audio can be present
                // together (single native pass, no bridge).
                let prefillLogits: MLXArray
                if let mmForward = capturedMultimodalForward,
                   !cacheHit,
                   capturedImageData != nil || capturedAudioMel != nil {
                    do {
                        var pixelValues: MLXArray? = nil
                        var imageHash: String? = nil
                        if let imgData = capturedImageData {
                            pixelValues = try preprocessImage(imgData)
                            // Only key the vision cache for pure-image
                            // requests (combined audio bypasses the cache).
                            imageHash = capturedAudioMel == nil
                                ? VisionEncoderCache.key(forImageBytes: imgData)
                                : nil
                        }
                        prefillLogits = mmForward(
                            inputArray, caches, pixelValues,
                            capturedAudioMel, capturedAudioMask, imageHash)
                    } catch {
                        // Fall back to text-only if preprocessing fails
                        prefillLogits = capturedForward(inputArray, caches)
                    }
                } else {
                    prefillLogits = capturedForward(inputArray, caches)
                }
                MLX.eval(prefillLogits)
                prefillDuration = CFAbsoluteTimeGetCurrent() - startTime

                // -- Store in prefix cache (write-behind) --
                // We cache KV for the full prompt (all tokens that have been
                // forwarded). On the next request with the same prefix, the
                // restored KV will cover all prompt tokens, and the "full cache
                // hit" path above trims the last token and re-forwards it.
                if effectiveUsePrefixCache && !cacheHit && promptTokens.count >= 8 {
                    if let fp16Caches {
                        var snapshotKeys: [[MLXArray]] = []
                        var snapshotValues: [[MLXArray]] = []
                        for cache in fp16Caches {
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
                                values: snapshotValues,
                                mediaHash: mediaHash
                            )
                        }
                    } else if let int8Caches {
                        // Mirror the fp16 path: append snapshots in cache order,
                        // skipping layers that have no own state (KV-shared
                        // suffix in Gemma 4). The first empty cache marks the
                        // end of the prefix we persist — caches beyond that are
                        // always empty for this checkpoint family.
                        var snapshots: [QuantizedKVSnapshot] = []
                        for cache in int8Caches {
                            if let snap = cache.quantizedSnapshot() {
                                snapshots.append(snap)
                            }
                        }
                        if !snapshots.isEmpty {
                            capturedPrefixCache.storeQuantized(
                                tokens: promptTokens,
                                modelId: capturedModelId,
                                snapshots: snapshots,
                                mediaHash: mediaHash
                            )
                        }
                    }
                }

                // Sample first token. For the spec-decoding path we need a host
                // Int up front; for the standard path we keep the token as a
                // lazy 1-element MLXArray so we can chain it directly into the
                // next forward without paying a GPU↔CPU sync per step.
                var nextToken: Int
                var nextTokenArr: MLXArray
                if shouldSpec {
                    nextToken = sampler.sample(prefillLogits)
                    nextTokenArr = MLXArray(Int32(nextToken))
                } else {
                    nextTokenArr = sampler.sampleArray(prefillLogits)
                    asyncEval(nextTokenArr)
                    nextToken = nextTokenArr.item(Int.self)
                }

                // Prefill the draft model on the FULL prompt BEFORE we
                // start the decode-time clock. Draft prefill is one-time
                // setup per generation (analogous to target prefill,
                // already in `prefillDuration`); attributing it to decode
                // would inflate per-token decode cost and depress tok/s
                // unfairly versus the non-spec baseline.
                //
                // Critical: must use `promptTokens`, NOT `inputArray`. On
                // a prefix-cache hit the target's `tokensToProcess` is
                // trimmed to the last prompt token (its KV was restored
                // from snapshot for everything else); the draft has no
                // such snapshot, so feeding it the trimmed input would
                // leave its cache covering 1 token vs the target's full
                // L, collapsing acceptance on every warm-server request.
                let draftCaches: [KVCache]?
                if shouldSpec, let specDec = capturedSpecDecoder, let draftModel = specDec.draft {
                    let dCaches = makeKVCaches(numLayers: draftModel.numLayers)
                    let draftInput = MLXArray(promptTokens.map { Int32($0) })
                        .reshaped(1, promptTokens.count)
                    let draftPrefillLogits = draftModel.forward(
                        draftInput, dCaches)
                    MLX.eval(draftPrefillLogits)
                    specDec.reset()
                    draftCaches = dCaches
                } else {
                    draftCaches = nil
                }
                // Roll draft prefill into the prefill time so decode tok/s
                // is comparable with the baseline path.
                prefillDuration = CFAbsoluteTimeGetCurrent() - startTime

                // -- Decode loop --
                let decodeStart = CFAbsoluteTimeGetCurrent()

                if shouldSpec, let specDec = capturedSpecDecoder, let draftCaches {

                    // Emit the first token sampled from prefill logits — specDec.step
                    // only returns tokens *after* lastToken, so we must yield it here.
                    if stopIds.contains(nextToken) {
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

                    while generatedCount < maxTokens && !stopIds.contains(nextToken) {
                        // Speculative step: get multiple tokens at once.
                        // Spec path requires fp16 caches; shouldSpec already excludes int8.
                        let accepted = specDec.step(
                            lastToken: nextToken,
                            targetCaches: fp16Caches!,
                            draftCaches: draftCaches
                        )

                        for token in accepted {
                            if stopIds.contains(token) {
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
                        if stopIds.contains(nextToken) || generatedCount >= maxTokens { break }
                    }
                } else {
                    // === Standard Decode Path ===
                    // WS-D D3: only when the request opts into penalties /
                    // mirostat do we maintain a recent-token window; the
                    // default path below is byte-for-byte unchanged.
                    let trackHistory = sampler.needsHistory
                    var recent: [Int] = trackHistory
                        ? Array(promptTokens.suffix(512)) : []

                    // Two-deep on-GPU pipeline: the just-sampled token stays as
                    // a lazy MLXArray and is fed directly into the next forward.
                    // We schedule the next forward + sample, then sync only on
                    // the *current* token's host int for decode/yield/EOS. By
                    // the time we sync, the GPU has already started executing
                    // forward N+1, so kernel launch and host string work overlap
                    // with kernel execution.
                    while generatedCount < maxTokens {
                        if stopIds.contains(nextToken) {
                            continuation.yield(TokenEvent(
                                tokenId: nextToken, text: "",
                                elapsed: CFAbsoluteTimeGetCurrent() - startTime, isEnd: true))
                            break
                        }

                        // Reuse the on-GPU sampled token as forward input —
                        // avoids the per-step `MLXArray([Int32(nextToken)])`
                        // allocation and keeps the dependency graph on-device.
                        let tokenInput = nextTokenArr.reshaped(1, 1)
                        let logits = capturedForward(tokenInput, caches)
                        let nextTokenArr2: MLXArray
                        if trackHistory {
                            recent.append(nextToken)
                            nextTokenArr2 = sampler.sampleArray(logits, recent: recent)
                        } else {
                            nextTokenArr2 = sampler.sampleArray(logits)
                        }

                        // Schedule both the forward and the next sample to run
                        // on the GPU in the background while we decode/yield.
                        asyncEval(nextTokenArr2)

                        // While GPU is busy with iteration N+1, decode and yield N.
                        let yieldedToken = nextToken
                        let tokenText = capturedTokenizer.decode(token: yieldedToken)
                        continuation.yield(TokenEvent(
                            tokenId: yieldedToken, text: tokenText,
                            elapsed: CFAbsoluteTimeGetCurrent() - startTime))
                        generatedCount += 1

                        // Sync once per step on a single 1-element int32 array.
                        nextTokenArr = nextTokenArr2
                        nextToken = nextTokenArr.item(Int.self)
                    }
                }

                let decodeDuration = CFAbsoluteTimeGetCurrent() - decodeStart

                if generatedCount >= maxTokens {
                    continuation.yield(TokenEvent(
                        tokenId: -1, text: "",
                        elapsed: CFAbsoluteTimeGetCurrent() - startTime, isEnd: true))
                }

                let specStats: SpeculativeStats?
                if shouldSpec, let specDec = capturedSpecDecoder, specDec.totalRounds > 0 {
                    specStats = SpeculativeStats(
                        rounds: specDec.totalRounds,
                        acceptedTokens: specDec.totalAccepted,
                        finalK: specDec.currentK,
                        acceptanceRate: specDec.acceptanceRate
                    )
                } else {
                    specStats = nil
                }

                statsHolder.stats = GenerationStats(
                    promptTokens: promptTokens.count,
                    generatedTokens: generatedCount,
                    prefillTime: prefillDuration,
                    decodeTime: decodeDuration,
                    speculative: specStats
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
        draftModelName = nil
        autoUseSpec = false
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
