import Foundation
import os
import MLX
import KLMCore
import KLMCache
import KLMTokenizer
import KLMSampler
import KLMGrammar
import KLMRegistry
#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

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

    /// Grammar-constrained decoding caches (parity plan §8). All guarded by
    /// `jsonMaskLock`. The per-model token-piece table (every vocab id decoded
    /// to its string) is decoded once and shared by every mask. `cachedJSONMask`
    /// is the Stage A any-JSON-value mask; `cachedSchemaMask` memoizes the most
    /// recent Stage B schema mask keyed by (schema string, stop set) so a
    /// repeated schema request does not recompile/rebuild.
    private var cachedPieces: [String]?
    private var cachedJSONMask: JSONTokenMask?
    private var cachedSchemaMask: (schema: String, mask: any GrammarLogitMask)?
    /// Memoized most-recent Stage C regex mask, keyed by (pattern, stop set).
    private var cachedRegexMask: (pattern: String, mask: any GrammarLogitMask)?
    private let jsonMaskLock = NSLock()

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

    /// Internal access for the Stage B batched decoder (BatchedDecode.swift),
    /// which lives in a separate file and so cannot see the `private` stored
    /// model/tokenizer directly.
    var loadedModelForBatching: LoadedModel? { loadedModel }
    var tokenizerEOS: Int? { tokenizer?.eosTokenId }
    /// Tokenize raw text for the Stage B batched-decode tests.
    func encodeForBatchTest(_ text: String) -> [Int]? { tokenizer?.encode(text) }

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

    /// Continuous batched-decode driver (follow-up #8, Stage C1). Lazily
    /// created on the first eligible `submitBatched` call against the current
    /// model and torn down on `unload()`, so a model swap never decodes against
    /// a stale model's captured forward closures. Guarded by `batcherLock`.
    private var continuousBatcher: ContinuousBatcher?
    private let batcherLock = NSLock()

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

    /// Run a tiny dry forward to pre-warm compile caches, kernel
    /// JIT, and lazy graph state so the first user request does not
    /// pay the one-time costs. Honors `KRILL_SKIP_WARMUP=1` to opt
    /// out (CI / cold-startup sensitive use). Safe to call after
    /// `load()` on any family - the call returns silently if the
    /// model is not loaded yet, and any error is swallowed (warmup
    /// is best-effort by design; an unrelated failure here must not
    /// block the server from accepting real requests).
    ///
    /// What this actually pre-warms:
    /// - `MLX.compile` block-forward caches that PR #48 builds
    ///   lazily on first call. For VL families, this means the 32
    ///   vision-block compiled forwards; for text-only families,
    ///   the JIT-compiled fused SwiGLU + any future text-block
    ///   compile slots.
    /// - The custom Metal kernel binaries (fused SwiGLU JIT).
    /// - Tokenizer / weight access page-cache.
    ///
    /// Behaviour is family-aware: VL models warm with a tiny
    /// synthetic image so the vision tower's compile cache fills;
    /// other families warm with text only. Cost is well under one
    /// second on the families this build ships.
    public func warmup() async {
        if let v = ProcessInfo.processInfo.environment["KRILL_SKIP_WARMUP"],
           !v.isEmpty, v != "0", v.lowercased() != "false" {
            return
        }
        guard loadedModel != nil, tokenizer != nil else { return }
        let messages: [[String: String]] = [
            ["role": "user", "content": "Warmup."],
        ]
        // For vision-capable families, attach a tiny synthetic gray
        // PNG so the vision tower's MLX.compile cache fills here
        // instead of on the user's first image request. The data is
        // discarded with the stream. Best effort: a platform without
        // CoreGraphics (or any PNG-encoding failure) just falls
        // through to a text-only warmup.
        var imageData: Data? = nil
        if supportsNativeImage {
            imageData = Self.warmupImagePNG()
        }
        let (stream, _) = generate(
            messages: messages, params: .greedy, maxTokens: 1,
            usePrefixCache: false, imageData: imageData)
        for await event in stream {
            if event.isEnd { break }
        }
    }

    /// Build a tiny solid-gray 224x224 PNG for the VL warmup. The
    /// content does not matter - only the shape does, since warmup
    /// is purely about pre-compiling the vision tower. Returns nil
    /// on platforms without CoreGraphics or on encoder failure.
    private static func warmupImagePNG() -> Data? {
        #if canImport(CoreGraphics) && canImport(ImageIO)
        let w = 224, h = 224
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.setFillColor(CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        guard let image = ctx.makeImage() else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, "public.png" as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return out as Data
        #else
        return nil
        #endif
    }

    /// Swap the current model for a new one at the given directory.
    /// Loads the new model first — if loading fails, the previous model remains active.
    public func swap(modelDirectory newDir: URL) async throws {
        _isSwapping.withLock { $0 = true }

        let newModel: LoadedModel
        let newTokenizer: KLMTokenizer
        do {
            // Load new model into temporaries before touching current state.
            newModel = try loadModel(from: newDir)
            newTokenizer = try await KLMTokenizer(from: newDir)
        } catch {
            // Failed before any state changed - clear the lock and
            // rethrow. The previous model remains active.
            _isSwapping.withLock { $0 = false }
            throw error
        }

        // Success — now swap atomically.
        unload()
        self.modelDirectory = newDir
        self.loadedModel = newModel
        self.tokenizer = newTokenizer
        self.loadedAt = Date()
        // Release the swap lock BEFORE warmup so the server can
        // accept requests on the new model while warmup runs.
        // Warmup is a pre-compile / pre-JIT optimization, not a
        // correctness requirement; new requests routed through
        // the same compile-on-demand path naturally elide the
        // remaining warmup work.
        _isSwapping.withLock { $0 = false }
        await warmup()
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
    /// Render a created model's Modelfile `TEMPLATE` override (an Ollama
    /// Go text/template) against the chat messages. Returns the rendered
    /// prompt string, or `nil` if the template fails to parse/evaluate --
    /// in which case the caller falls back to the model's built-in chat
    /// template so an exotic override never hard-fails a request.
    private func renderWithOverrideTemplate(
        _ template: String, messages: [[String: String]]
    ) -> String? {
        do {
            let context = OllamaTemplateContext.build(messages: messages)
            return try GoTemplate.render(template, context)
        } catch {
            FileHandle.standardError.write(Data((
                "[KrillLM] TEMPLATE override render failed (\(error)); "
                + "falling back to the built-in chat template.\n").utf8))
            return nil
        }
    }

    /// The per-model token-piece table (every vocab id decoded to its string),
    /// decoded once and cached. Caller must hold `jsonMaskLock`. `nil` when the
    /// tokenizer cannot report a vocab size.
    private func pieceTableLocked() -> [String]? {
        if let cachedPieces { return cachedPieces }
        guard let tokenizer, let vocabSize = tokenizer.vocabSize, vocabSize > 0 else {
            return nil
        }
        let start = CFAbsoluteTimeGetCurrent()
        var pieces = [String](repeating: "", count: vocabSize)
        for id in 0 ..< vocabSize { pieces[id] = tokenizer.decode(token: id) }
        cachedPieces = pieces
        FileHandle.standardError.write(Data((
            "[KrillLM] built grammar token-piece table (vocab=\(vocabSize)) in "
            + String(format: "%.0fms", (CFAbsoluteTimeGetCurrent() - start) * 1000)
            + "\n").utf8))
        return pieces
    }

    /// Build/reuse the grammar logit mask for the requested `format`. Returns
    /// `nil` (mask disabled, request relies on guided-prompt + post-extraction)
    /// when the tokenizer cannot report a vocab size. For `.jsonSchema`, an
    /// uncompilable schema falls back to the plain JSON-validity mask.
    private func grammarMask(for format: OutputFormat, stopIds: Set<Int>) -> (any GrammarLogitMask)? {
        jsonMaskLock.lock(); defer { jsonMaskLock.unlock() }
        guard let pieces = pieceTableLocked() else { return nil }

        switch format {
        case .json:
            return jsonValidityMaskLocked(pieces: pieces, stopIds: stopIds)

        case .jsonSchema(let schema):
            // Memoize the most recent (schema, stopIds) so a repeated schema
            // request reuses the compiled automaton + built mask.
            if let cached = cachedSchemaMask, cached.schema == schema,
               cached.mask.stopIdSet == stopIds {
                return cached.mask
            }
            guard let grammar = SchemaGrammar.compile(schema) else {
                FileHandle.standardError.write(Data((
                    "[KrillLM] JSON schema did not compile; falling back to the "
                    + "plain JSON-validity mask.\n").utf8))
                return jsonValidityMaskLocked(pieces: pieces, stopIds: stopIds)
            }
            let mask = GrammarTokenMask(automaton: grammar, pieces: pieces, stopIds: stopIds)
            cachedSchemaMask = (schema, mask)
            return mask

        case .regex(let pattern):
            if let cached = cachedRegexMask, cached.pattern == pattern,
               cached.mask.stopIdSet == stopIds {
                return cached.mask
            }
            guard let grammar = RegexGrammar.compile(pattern) else {
                FileHandle.standardError.write(Data((
                    "[KrillLM] regex did not compile; decoding unconstrained "
                    + "for this request.\n").utf8))
                return nil
            }
            let mask = GrammarTokenMask(automaton: grammar, pieces: pieces, stopIds: stopIds)
            cachedRegexMask = (pattern, mask)
            return mask
        }
    }

    /// The Stage A any-JSON-value mask. Caller must hold `jsonMaskLock`.
    /// Rebuilt when the request's stop set differs from the cached one (a
    /// Modelfile `PARAMETER stop` can vary it per request; the mask gates EOS
    /// on exactly that set).
    private func jsonValidityMaskLocked(pieces: [String], stopIds: Set<Int>) -> JSONTokenMask {
        if let cached = cachedJSONMask, cached.stopIdSet == stopIds { return cached }
        let mask = JSONTokenMask(pieces: pieces, stopIds: stopIds)
        cachedJSONMask = mask
        return mask
    }

    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        params: SamplingParams = .greedy,
        maxTokens: Int = 512,
        useSpeculative: Bool? = nil,
        usePrefixCache: Bool = true,
        imageData: Data? = nil,
        audioData: Data? = nil,
        contextLimit: Int? = nil,
        promptTemplateOverride: String? = nil,
        format: OutputFormat? = nil
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
                        contextLimit: contextLimit,
                        promptTemplateOverride: promptTemplateOverride,
                        format: format)
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
        contextLimit: Int? = nil,
        promptTemplateOverride: String? = nil,
        format: OutputFormat? = nil
    ) -> (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?) {
        guard let loadedModel, let tokenizer else {
            let emptyStream = AsyncStream<TokenEvent> { $0.finish() }
            return (emptyStream, { nil })
        }

        // Qwen 2.5-VL native runtime. Its 3D-mRoPE decode loop needs
        // a per-step positional offset that the generic decode path
        // does not thread (after an image span the KV-cache length
        // is not the next mRoPE position), so every VL request -
        // image or text-only - routes to the dedicated driver.
        if let vlModel = loadedModel.module as? Qwen25VLForConditionalGeneration {
            return generateQwen25VL(
                model: vlModel, tokenizer: tokenizer, messages: messages,
                params: params, maxTokens: maxTokens, imageData: imageData,
                usePrefixCache: usePrefixCache)
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
        if let overrideTemplate = promptTemplateOverride,
           !overrideTemplate.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           let rendered = renderWithOverrideTemplate(
               overrideTemplate, messages: preparedMessages) {
            // A created model's Modelfile `TEMPLATE` override. Ollama
            // TEMPLATEs are Go text/templates and embed their own special
            // tokens as literal text, so encode the rendered string
            // without injecting an extra BOS (the template owns the
            // structure). `renderWithOverrideTemplate` returns nil on a
            // parse/eval failure, falling through to the built-in path so
            // an exotic Modelfile never hard-fails a request.
            promptTokensBuilt = tokenizer.encodeWithoutExtraBOS(rendered)
        } else if loadedModel.family == "gemma4" {
            promptTokensBuilt = tokenizer.formatGemma4TokenIds(messages: preparedMessages)
        } else if let direct = tokenizer.applyChatTemplateTokens(messages: preparedMessages) {
            // Same loss-avoidance principle as the Gemma 4 branch, but
            // for every other family whose tokenizer can render via the
            // swift-transformers Jinja engine (embedded template or the
            // newer separate `chat_template.jinja`): keep the IDs the
            // library already produced instead of decoding to text and
            // re-encoding, which silently drops ChatML / FIM / tool
            // special tokens on Qwen 3 Coder and similar checkpoints
            // and rots generation into pure special-token loops.
            promptTokensBuilt = direct
        } else {
            // Last resort: the wrapper's manual Llama 3 fallback path,
            // for checkpoints with neither an embedded nor an external
            // template. Lossy round-trip but the alternative is no
            // chat formatting at all.
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
        // Grammar-constrained decoding (parity plan §8): when a structured
        // output format is requested, build/reuse the token-level logit mask
        // (any-JSON-value for `.json`, schema-constrained for `.jsonSchema`).
        // EOS/stop tokens are passed so the mask can forbid them until the
        // value is complete. nil format → no work.
        let grammarMask: (any GrammarLogitMask)? = format.map { self.grammarMask(for: $0, stopIds: stopIds) } ?? nil
        let numLayers = loadedModel.numLayers
        let forwardFn = loadedModel.forward
        let prefillForwardFn = loadedModel.prefillForward
        let multimodalFn = loadedModel.multimodalForward
        let multimodalPrefillFn = loadedModel.multimodalPrefillForward
        let statsHolder = StatsHolder()

        // Bundle every value the Task closure needs into a single
        // @unchecked Sendable struct. The closure then captures just
        // {statsHolder, captures}, sidestepping the Swift 6.2.4 SIL
        // SendNonSendable pass crash that fires when many individual
        // nonisolated(unsafe) values are captured under -O. Inside
        // the closure body we re-bind each field to its original
        // `capturedX` name as a closure-local `let`, so the body
        // diff stays empty.
        //
        // Field rationale (was on the per-field declarations):
        //   prefillForward: when present, returns logits sliced to
        //     the last position. Used on prefill (sampling reads the
        //     last position only) so the [1, L, hidden] -> [1, L,
        //     vocab] matmul drops to a single position. Bit-exact
        //     for the sampled token; decode-step forwards keep using
        //     `forward` (single-token input makes the slice a no-op).
        //   multimodalPrefillForward: optional last-token-only
        //     multimodal prefill (Gemma 4 wires this; same
        //     `lastTokenOnly` win as the text path).
        //   moeModel: Native MoE runtime when this is one; used to
        //     scope and read expert-utilization telemetry around the
        //     generation. nil for every dense family, so the
        //     reset/read calls become no-ops.
        struct Captures: @unchecked Sendable {
            typealias ForwardFn = (MLXArray, [KVCacheProtocol]?) -> MLXArray
            typealias MultimodalForwardFn = (
                MLXArray, [KVCacheProtocol]?, MLXArray?, MLXArray?, MLXArray?, String?
            ) -> MLXArray
            let forward: ForwardFn
            let prefillForward: ForwardFn?
            let multimodalForward: MultimodalForwardFn?
            let multimodalPrefillForward: MultimodalForwardFn?
            let tokenizer: KLMTokenizer
            let prefixCache: PrefixCache
            let specDecoder: SpeculativeDecoder?
            let modelId: String
            let imageData: Data?
            let audioMel: MLXArray?
            let audioMask: MLXArray?
            let moeModel: Qwen3MoEForCausalLM?
            let grammarMask: (any GrammarLogitMask)?
        }
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
        // Grammar masking advances a per-token JSON automaton in the
        // standard serial loop; the multi-token speculative path cannot
        // honor a per-step mask, so a format request disables spec.
        let shouldSpec = wantsSpec && specDecoder != nil && !useInt8KV
            && !params.penaltiesActive && greedyRequest && grammarMask == nil
        let captures = Captures(
            forward: forwardFn,
            prefillForward: prefillForwardFn,
            multimodalForward: multimodalFn,
            multimodalPrefillForward: multimodalPrefillFn,
            tokenizer: tokenizer,
            prefixCache: self.prefixCache,
            specDecoder: self.specDecoder,
            modelId: self.modelId,
            imageData: imageData,
            audioMel: nativeAudio?.mel,
            audioMask: nativeAudio?.validMask,
            moeModel: loadedModel.module as? Qwen3MoEForCausalLM,
            grammarMask: grammarMask
        )

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
            Task { [statsHolder, captures] in
                // Re-bind each Captures field to the closure-body
                // local name the rest of this Task body uses. Doing
                // this here (rather than referencing `captures.x`
                // throughout) keeps the body diff against the
                // pre-refactor version effectively empty.
                let capturedForward = captures.forward
                let capturedPrefillForward = captures.prefillForward
                let capturedMultimodalForward = captures.multimodalForward
                let capturedMultimodalPrefillForward = captures.multimodalPrefillForward
                let capturedTokenizer = captures.tokenizer
                let capturedPrefixCache = captures.prefixCache
                let capturedSpecDecoder = captures.specDecoder
                let capturedModelId = captures.modelId
                let capturedImageData = captures.imageData
                let capturedAudioMel = captures.audioMel
                let capturedAudioMask = captures.audioMask
                let capturedMoEModel = captures.moeModel
                let capturedGrammarMask = captures.grammarMask

                let startTime = CFAbsoluteTimeGetCurrent()
                // Scope expert-utilization telemetry to this generation
                // (no-op for dense models).
                capturedMoEModel?.resetMoEUtilizationStats()
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
                        // Prefer the last-token-only multimodal
                        // prefill closure when the family wired it.
                        let mmPrefill = capturedMultimodalPrefillForward
                            ?? mmForward
                        prefillLogits = mmPrefill(
                            inputArray, caches, pixelValues,
                            capturedAudioMel, capturedAudioMask, imageHash)
                    } catch {
                        // Fall back to text-only if preprocessing fails -
                        // honor `prefillForward` here too.
                        prefillLogits = (capturedPrefillForward ?? capturedForward)(inputArray, caches)
                    }
                } else {
                    // Text-only prefill. Prefer the `prefillForward`
                    // last-token-only path when the family wired it;
                    // otherwise fall back to the full forward. On a
                    // prefix-cache hit the input is 1 token, so the
                    // slice is a no-op either way.
                    prefillLogits = (capturedPrefillForward ?? capturedForward)(inputArray, caches)
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
                // Grammar-constrained decoding cursor for this generation
                // (any-JSON-value or schema-constrained). nil when no format
                // was requested or the mask was disabled. Validate the mask
                // width against the real logits width once (config vocab_size
                // can be padded relative to the lm_head); disable on mismatch.
                var grammarSession: (any GrammarLogitSession)?
                if let m = capturedGrammarMask {
                    let logitsVocab = prefillLogits.dim(prefillLogits.ndim - 1)
                    if m.vocabSize != logitsVocab {
                        FileHandle.standardError.write(Data((
                            "[KrillLM] grammar mask disabled: mask vocab "
                            + "\(m.vocabSize) != logits vocab \(logitsVocab).\n").utf8))
                    } else {
                        grammarSession = m.makeSession()
                    }
                }

                var nextToken: Int
                var nextTokenArr: MLXArray
                if shouldSpec {
                    nextToken = sampler.sample(prefillLogits)
                    nextTokenArr = MLXArray(Int32(nextToken))
                } else {
                    let prefillMask = grammarSession?.currentMask()
                    nextTokenArr = sampler.sampleArray(prefillLogits, mask: prefillMask)
                    asyncEval(nextTokenArr)
                    nextToken = nextTokenArr.item(Int.self)
                }
                // Advance the grammar by the first (prefill) token. If the
                // chosen token's piece does not extend the grammar (a mask
                // fail-open step let an off-grammar token through, or
                // single-id detok diverged from streaming detok), disable the
                // mask for the rest of this generation rather than freeze the
                // cursor and emit a stuck mask every step. `coerce` remains the
                // validity safety net.
                if let s = grammarSession, !s.advance(token: nextToken) {
                    grammarSession = nil
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
                    // Draft prefill's logits are never sampled or
                    // returned - it exists only to fill the draft
                    // KV cache. The cache is populated by the
                    // attention layers above the lm_head, so slicing
                    // the head output to the last position is bit
                    // exact and skips the full vocab matmul. Fall
                    // back to `forward` for drafts that have not
                    // wired `prefillForward` yet.
                    let draftPrefillFn = draftModel.prefillForward ?? draftModel.forward
                    let draftPrefillLogits = draftPrefillFn(
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
                        // Grammar mask for the NEXT token, given everything
                        // accepted so far (reflected in the grammar cursor).
                        let stepMask = grammarSession?.currentMask()
                        let nextTokenArr2: MLXArray
                        if trackHistory {
                            recent.append(nextToken)
                            nextTokenArr2 = sampler.sampleArray(logits, recent: recent, mask: stepMask)
                        } else {
                            nextTokenArr2 = sampler.sampleArray(logits, mask: stepMask)
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
                        // Advance the grammar by the freshly accepted token so
                        // the next iteration masks from the correct state. On a
                        // non-extending token, disable the mask for the rest of
                        // this generation (see the prefill site above) instead
                        // of freezing and re-emitting a stuck mask each step.
                        if let s = grammarSession, !s.advance(token: nextToken) {
                            grammarSession = nil
                        }
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
                    speculative: specStats,
                    moe: capturedMoEModel?.moeUtilization()
                )
                continuation.finish()
            }
        }

        return (stream, { statsHolder.stats })
    }

    /// Native Qwen 2.5-VL generation: preprocess the image, render
    /// the ChatML prompt with the `<|image_pad|>` run, and drive
    /// `Qwen25VLRuntime` (prefill + 3D-mRoPE-correct decode).
    /// Returns the same `(stream, stats)` contract as `generate`.
    private func generateQwen25VL(
        model: Qwen25VLForConditionalGeneration,
        tokenizer: KLMTokenizer,
        messages: [[String: String]],
        params: SamplingParams,
        maxTokens: Int,
        imageData: Data?,
        usePrefixCache: Bool = true
    ) -> (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?) {
        // Preprocess the image (if any) into the per-patch batch and
        // its post-spatial-merge grid. The grid sizes the
        // `<|image_pad|>` placeholder run and drives mRoPE.
        var pixelValues: MLXArray? = nil
        var gridMerged: (Int, Int)? = nil
        var imagePadCount = 0
        if let imgData = imageData {
            do {
                let prepped = try Qwen25VLImagePreprocessor.preprocess(
                    imgData, vision: model.config.vision)
                pixelValues = prepped.patches
                gridMerged = (prepped.gridHMerged, prepped.gridWMerged)
                imagePadCount = prepped.gridHMerged * prepped.gridWMerged
            } catch {
                // Never silently answer an image prompt as text-only.
                return Self.mediaErrorStream(
                    "Error: Qwen 2.5-VL image preprocessing failed: \(error)")
            }
        }

        let promptTokens = tokenizer.formatQwen25VLTokenIds(
            messages: messages,
            imagePadCount: imagePadCount,
            imageTokenId: model.config.imageTokenId,
            visionStartTokenId: model.config.visionStartTokenId,
            visionEndTokenId: model.config.visionEndTokenId)
        guard !promptTokens.isEmpty else {
            let empty = AsyncStream<TokenEvent> { c in
                c.yield(TokenEvent(tokenId: 0, text: "", elapsed: 0, isEnd: true))
                c.finish()
            }
            return (empty, { nil })
        }

        let stopIds = Self.stopTokenIds(
            modelDirectory: modelDirectory, tokenizerEOS: tokenizer.eosTokenId)
        // Hash the image bytes once so the model can consult its
        // per-instance vision-encoder cache and skip the vision tower
        // on a same-image follow-up (mirrors the Gemma 4 path).
        let mediaHash: String? = imageData.map {
            "img:" + VisionEncoderCache.key(forImageBytes: $0)
        }
        let statsHolder = StatsHolder()
        // Same Swift-6.2.4 SIL-pass workaround as `generate()`:
        // bundle every value the Task closure needs into a single
        // @unchecked Sendable struct and capture just that, then
        // re-bind to the original `capturedX` names inside the
        // closure so the body diff stays empty.
        //
        // Field rationale (was on the per-field declarations):
        //   prefixCache: the VL path consults the same PrefixCache
        //     the dense path uses; a same-image, same-prompt
        //     follow-up restores KV for the whole prompt and
        //     forwards just the last token, bypassing the 36-layer
        //     text prefill (~125 ms). The mediaHash includes the
        //     image bytes so a different image misses safely.
        //     Honor the request's `usePrefixCache` flag - the
        //     warmup forward passes `false` so its synthetic-image
        //     entry does not pollute the cache.
        struct Captures: @unchecked Sendable {
            let model: Qwen25VLForConditionalGeneration
            let tokenizer: KLMTokenizer
            let pixels: MLXArray?
            let grid: (Int, Int)?
            let prompt: [Int]
            let stops: Set<Int>
            let params: SamplingParams
            let max: Int
            let mediaHash: String?
            let prefixCache: PrefixCache?
            let modelId: String
        }
        let captures = Captures(
            model: model,
            tokenizer: tokenizer,
            pixels: pixelValues,
            grid: gridMerged,
            prompt: promptTokens,
            stops: stopIds,
            params: params,
            max: maxTokens,
            mediaHash: mediaHash,
            prefixCache: usePrefixCache ? self.prefixCache : nil,
            modelId: self.modelId
        )

        let stream = AsyncStream<TokenEvent> { continuation in
            Task { [statsHolder, captures] in
                let capturedModel = captures.model
                let capturedTokenizer = captures.tokenizer
                let capturedPixels = captures.pixels
                let capturedGrid = captures.grid
                let capturedPrompt = captures.prompt
                let capturedStops = captures.stops
                let capturedParams = captures.params
                let capturedMax = captures.max
                let capturedMediaHash = captures.mediaHash
                let capturedPrefixCache = captures.prefixCache
                let capturedModelId = captures.modelId

                let startTime = CFAbsoluteTimeGetCurrent()
                let output = Qwen25VLRuntime.generate(
                    model: capturedModel,
                    promptTokens: capturedPrompt,
                    pixelValues: capturedPixels,
                    imageGridMerged: capturedGrid,
                    maxTokens: capturedMax,
                    stopIds: capturedStops,
                    params: capturedParams,
                    mediaHash: capturedMediaHash,
                    prefixCache: capturedPrefixCache,
                    modelId: capturedModelId,
                    onToken: { token in
                        // Stream content tokens only. The terminal
                        // (isEnd) event is emitted below - AFTER the
                        // stats are published - so a consumer that
                        // breaks on the first isEnd event always
                        // observes populated GenerationStats. A stop
                        // token carries no text, so nothing is lost
                        // by not emitting it here.
                        guard !capturedStops.contains(token) else { return }
                        continuation.yield(TokenEvent(
                            tokenId: token,
                            text: capturedTokenizer.decode(token: token),
                            elapsed: CFAbsoluteTimeGetCurrent() - startTime))
                    })
                // Publish stats BEFORE the terminal event. The server
                // reads the stats accessor as soon as it sees isEnd;
                // setting stats first keeps the (stream, stats)
                // contract race-free.
                statsHolder.stats = GenerationStats(
                    promptTokens: capturedPrompt.count,
                    generatedTokens: output.tokens.count,
                    prefillTime: output.prefillSeconds,
                    decodeTime: output.decodeSeconds)
                // One terminal event: the stop token id if generation
                // ended on a stop, else -1 (maxTokens reached).
                let sawStop = output.tokens.last.map {
                    capturedStops.contains($0)
                } ?? false
                continuation.yield(TokenEvent(
                    tokenId: sawStop ? (output.tokens.last ?? -1) : -1,
                    text: "",
                    elapsed: CFAbsoluteTimeGetCurrent() - startTime,
                    isEnd: true))
                continuation.finish()
            }
        }
        return (stream, { statsHolder.stats })
    }

    /// Unload the model from memory and release resources.
    public func unload() {
        // Tear down the continuous batcher first so its background decode loop
        // stops and finishes every in-flight / waiting stream before the model
        // it captured is dropped (its forward closures would otherwise dangle).
        batcherLock.lock()
        let batcher = continuousBatcher
        continuousBatcher = nil
        batcherLock.unlock()
        batcher?.stop()
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

// MARK: - Batched concurrent decode (live streaming entry)

/// One row's request for ``InferenceEngine/generateBatched(_:)`` — the
/// text-only subset of `generate(...)` arguments the Stage B batched decode
/// path supports.
public struct BatchGenRequest: Sendable {
    public let messages: [[String: String]]
    public let params: SamplingParams
    public let maxTokens: Int
    public let contextLimit: Int?
    public let promptTemplateOverride: String?
    /// Carried so the serial fallback (single-row cohort / ineligible model)
    /// reproduces the request's exact non-batched behavior. The true batched
    /// path (R > 1) bypasses both by design (Stage B: fp16 KV, no prefix
    /// cache, no spec).
    public let useSpeculative: Bool?
    public let usePrefixCache: Bool
    public init(messages: [[String: String]],
                params: SamplingParams = .greedy,
                maxTokens: Int = 512,
                contextLimit: Int? = nil,
                promptTemplateOverride: String? = nil,
                useSpeculative: Bool? = nil,
                usePrefixCache: Bool = true) {
        self.messages = messages
        self.params = params
        self.maxTokens = maxTokens
        self.contextLimit = contextLimit
        self.promptTemplateOverride = promptTemplateOverride
        self.useSpeculative = useSpeculative
        self.usePrefixCache = usePrefixCache
    }
}

/// Tiny thread-safe live-stream counter: when every row's consumer has gone
/// away the shared decode Task can stop early. `NSLock`-backed to avoid
/// pulling in another dependency just for one counter.
private final class BatchLiveCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var n: Int
    init(_ n: Int) { self.n = n }
    func decrementReachedZero() -> Bool {
        lock.lock(); defer { lock.unlock() }
        n -= 1
        return n <= 0
    }
}

extension InferenceEngine {
    /// Everything the batched-decode Task needs, bundled into one
    /// `@unchecked Sendable` struct so the Task closure captures a single
    /// value rather than ~12 individual nonisolated ones — the same workaround
    /// `generate(...)`'s `Captures` uses for the Swift 6.2.4 SIL
    /// SendNonSendable crash under `-O`.
    private struct BatchedCaptures: @unchecked Sendable {
        /// Prefix-cache-aware per-row prefill (tokens, fresh caches, the row's
        /// usePrefixCache flag) -> prefill logits. The SAME protocol-typed
        /// closures the continuous batcher uses, so the static cohort and the
        /// live batcher consult the shared prefix cache identically and share the
        /// fp16/int8 dispatch. `useQuantizedKV` selects the per-row cache factory.
        let prefillRow: ([Int], [KVCacheProtocol], Bool) -> MLXArray
        let batchedForward: (MLXArray, [KVCacheProtocol], MLXArray, [Int]) -> MLXArray
        let numLayers: Int
        let useQuantizedKV: Bool
        let tokenizer: KLMTokenizer
        let stopIds: Set<Int>
        let promptRows: [[Int]]
        /// Per-row prefix-cache opt-in, mirroring each `BatchGenRequest`.
        let usePrefixCache: [Bool]
        let samplers: [Sampler]
        let perRowMax: [Int]
        let conts: [AsyncStream<TokenEvent>.Continuation]
        let statsHolders: [StatsHolder]
    }

    /// Decode several ragged-length, **text-only** prompts for a plain-causal
    /// family (Llama 3.x / Qwen 2.5-3 dense) in ONE batched forward per step,
    /// streaming each row's tokens to its own `AsyncStream`. The correctness
    /// core — per-row RoPE offsets, left-padded KV stacking, and the additive
    /// pad mask — is the verified Stage B path (`batchedGreedyDecode` /
    /// `teacherForcedBatchedVsSerialMaxDiff`); this adds per-row sampling,
    /// per-row stop/maxTokens, and live streaming on top of it.
    ///
    /// Returns one `(stream, stats)` per request, in request order. Falls back
    /// to per-row serial `generate(...)` when the batched path cannot apply
    /// (unsupported family, nothing loaded, an empty prompt, or a single row)
    /// so the entry is always total-correct regardless of how it is called.
    ///
    /// Scope: fp16 KV, no speculative decode; finished rows stay in the batch
    /// (they simply stop emitting) until every row completes. The shared prefix
    /// cache IS consulted per row at prefill (Stage C4), honoring each row's
    /// `usePrefixCache` - a full hit restores that prompt's KV and forwards only
    /// the last token, identical to a cold prefill, so the stacked decode is
    /// byte-for-byte the same. Shrinking the batch mid-flight and continuous
    /// admission are the continuous batcher (`submitBatched`), not this static
    /// cohort.
    public func generateBatched(_ requests: [BatchGenRequest])
        -> [(stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?)]
    {
        // A fallback row is NOT a batched row - it is the request running
        // alone on the serial path - so it honors that request's own
        // useSpeculative / usePrefixCache rather than forcing them off. This
        // keeps a single-row cohort (or an ineligible model) byte-identical to
        // calling generate() directly. (The true R>1 batched loop below now
        // consults the shared prefix cache per row (Stage C4); it still never
        // runs speculative decode, per the batched-path scope.)
        func serialFallback()
            -> [(stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?)]
        {
            requests.map { r in
                generate(messages: r.messages, params: r.params,
                         maxTokens: r.maxTokens, useSpeculative: r.useSpeculative,
                         usePrefixCache: r.usePrefixCache, contextLimit: r.contextLimit,
                         promptTemplateOverride: r.promptTemplateOverride)
            }
        }

        guard !requests.isEmpty else { return [] }
        guard requests.count > 1,
              let model = loadedModelForBatching,
              model.batchedDecodeForward != nil,
              let tokenizer, let eosId = tokenizerEOS
        else { return serialFallback() }

        // Per-row prompt tokens, built with the SAME text-path logic as
        // generate() so a batched row reproduces that prompt's solo run.
        var promptRows: [[Int]] = []
        promptRows.reserveCapacity(requests.count)
        for r in requests {
            let toks = buildBatchedPromptTokens(
                messages: r.messages, contextLimit: r.contextLimit,
                promptTemplateOverride: r.promptTemplateOverride, tokenizer: tokenizer)
            guard !toks.isEmpty else { return serialFallback() }
            promptRows.append(toks)
        }

        let R = requests.count
        let stopIds = Self.stopTokenIds(
            modelDirectory: modelDirectory, tokenizerEOS: eosId)

        var conts: [AsyncStream<TokenEvent>.Continuation] = []
        var streams: [AsyncStream<TokenEvent>] = []
        conts.reserveCapacity(R)
        streams.reserveCapacity(R)
        for _ in 0 ..< R {
            var captured: AsyncStream<TokenEvent>.Continuation!
            let s = AsyncStream<TokenEvent> { captured = $0 }
            conts.append(captured)
            streams.append(s)
        }
        let statsHolders = (0 ..< R).map { _ in StatsHolder() }

        let caps = BatchedCaptures(
            prefillRow: protocolPrefillRow(model),
            batchedForward: protocolBatchedForward(model),
            numLayers: model.numLayers,
            useQuantizedKV: useQuantizedBatched(model),
            tokenizer: tokenizer,
            stopIds: stopIds,
            promptRows: promptRows,
            usePrefixCache: requests.map { $0.usePrefixCache },
            samplers: requests.map { Sampler(params: $0.params) },
            perRowMax: requests.map { $0.maxTokens },
            conts: conts,
            statsHolders: statsHolders)

        // The decode loop runs off the caller's (event-loop) context.
        let task = Task(priority: .userInitiated) { Self.runBatchedDecode(caps) }
        let live = BatchLiveCounter(R)
        for c in conts {
            c.onTermination = { _ in if live.decrementReachedZero() { task.cancel() } }
        }

        return (0 ..< R).map { r in
            (streams[r], { [statsHolders] in statsHolders[r].stats })
        }
    }

    /// Build the prefix-cache-aware per-row prefill closure for the batched
    /// path (Stage C4). Mirrors the dense `generate()` prefill exactly, fp16
    /// only: on a FULL prefix hit (`usePrefixCache` set and the cached prefix
    /// covers the whole prompt) it restores each layer's KV, trims the last
    /// position, and forwards just that token - so the resulting per-row caches
    /// are identical to a cold prefill and the stacked decode is byte-for-byte
    /// the same. On a miss it prefills cold and stores write-behind (>= 8
    /// tokens). Gemma 4 KV-shared layers leave a suffix of caches empty, so the
    /// store skips nil snapshots and the restore only fills the layers the hit
    /// carries - matching the dense path. `mediaHash` is nil: the batched path
    /// is text-only (image/audio requests are routed serially upstream).
    private func makeBatchedPrefillRow(
        _ model: LoadedModel
    ) -> ([Int], [KVCache], Bool) -> MLXArray {
        let rawPrefill = model.prefillForward ?? model.forward
        let pc = self.prefixCache
        let pcModelId = self.modelId
        return { tokens, caches, usePrefixCache in
            var cacheHit = false
            if usePrefixCache,
               let hit = pc.lookup(tokens: tokens, modelId: pcModelId, mediaHash: nil),
               !hit.keys.isEmpty, hit.prefixLength == tokens.count {
                for (i, cache) in caches.enumerated() {
                    if i < hit.keys.count, let k = hit.keys[i].first,
                       i < hit.values.count, let v = hit.values[i].first {
                        cache.restore(keys: k, values: v)
                    }
                }
                cacheHit = true
            }

            let toProcess: [Int]
            if cacheHit {
                let trimmed = max(0, tokens.count - 1)
                for cache in caches { cache.truncate(to: trimmed) }
                toProcess = [tokens.last!]
            } else {
                toProcess = tokens
            }

            let input = MLXArray(toProcess.map { Int32($0) }).reshaped(1, toProcess.count)
            let logits = rawPrefill(input, caches)
            MLX.eval(logits)   // realize the cache fills before snapshotting for store

            if usePrefixCache, !cacheHit, tokens.count >= 8 {
                var snapshotKeys: [[MLXArray]] = []
                var snapshotValues: [[MLXArray]] = []
                for cache in caches {
                    if let snap = cache.snapshot() {
                        snapshotKeys.append([snap.keys])
                        snapshotValues.append([snap.values])
                    }
                }
                if !snapshotKeys.isEmpty {
                    pc.store(tokens: tokens, modelId: pcModelId,
                             keys: snapshotKeys, values: snapshotValues, mediaHash: nil)
                }
            }
            return logits
        }
    }

    /// int8 analogue of ``makeBatchedPrefillRow``: the prefix-cache-aware per-row
    /// prefill closure for the QUANTIZED batched path (Stage C4). Mirrors the
    /// int8 serial prefill exactly (`lookupQuantized` / `restoreQuantized` /
    /// `storeQuantized`): a full hit restores each layer's uint8 KV (avoiding a
    /// dequant->requant round trip), trims the last position, and forwards just
    /// that token - so the resulting per-row quantized caches are identical to a
    /// cold int8 prefill and the stacked decode is byte-for-byte the same; a
    /// miss prefills cold and stores write-behind (>= 8 tokens). Gemma 4
    /// KV-shared layers leave a suffix of caches empty, so the store skips nil
    /// snapshots and the restore fills only the layers the hit carries.
    private func makeBatchedPrefillRowQuantized(
        _ model: LoadedModel
    ) -> ([Int], [QuantizedKVCache], Bool) -> MLXArray {
        let rawPrefill = model.prefillForward ?? model.forward
        let pc = self.prefixCache
        let pcModelId = self.modelId
        return { tokens, caches, usePrefixCache in
            var cacheHit = false
            if usePrefixCache,
               let hit = pc.lookupQuantized(tokens: tokens, modelId: pcModelId, mediaHash: nil),
               !hit.layers.isEmpty, hit.layers.count <= caches.count,
               hit.prefixLength == tokens.count {
                for i in 0 ..< hit.layers.count {
                    caches[i].restoreQuantized(hit.layers[i])
                }
                cacheHit = true
            }

            let toProcess: [Int]
            if cacheHit {
                let trimmed = max(0, tokens.count - 1)
                for cache in caches { cache.truncate(to: trimmed) }
                toProcess = [tokens.last!]
            } else {
                toProcess = tokens
            }

            let input = MLXArray(toProcess.map { Int32($0) }).reshaped(1, toProcess.count)
            let logits = rawPrefill(input, caches)
            MLX.eval(logits)

            if usePrefixCache, !cacheHit, tokens.count >= 8 {
                var snapshots: [QuantizedKVSnapshot] = []
                for cache in caches {
                    if let snap = cache.quantizedSnapshot() { snapshots.append(snap) }
                }
                if !snapshots.isEmpty {
                    pc.storeQuantized(tokens: tokens, modelId: pcModelId,
                                      snapshots: snapshots, mediaHash: nil)
                }
            }
            return logits
        }
    }

    /// Whether the engine should batch this model with int8 KV: int8 is
    /// configured AND a quantized batched forward is wired (Gemma 4). Other
    /// families have no quantized forward, so they batch fp16 - matching their
    /// serial behavior (they also fall back to fp16 KV serially).
    func useQuantizedBatched(_ model: LoadedModel) -> Bool {
        usesQuantizedKVCache && model.batchedDecodeForwardQuantized != nil
    }

    /// Adapt the chosen fp16/int8 prefill closure to the protocol-typed shape
    /// the batcher holds. The batcher allocates caches from the matching factory
    /// (`useQuantizedKV`), so the downcast is always valid.
    private func protocolPrefillRow(
        _ model: LoadedModel
    ) -> ([Int], [KVCacheProtocol], Bool) -> MLXArray {
        if useQuantizedBatched(model) {
            let q = makeBatchedPrefillRowQuantized(model)
            return { toks, caches, upc in q(toks, caches.map { $0 as! QuantizedKVCache }, upc) }
        }
        let f = makeBatchedPrefillRow(model)
        return { toks, caches, upc in f(toks, caches.map { $0 as! KVCache }, upc) }
    }

    /// Adapt the chosen fp16/int8 batched forward to the protocol-typed shape.
    private func protocolBatchedForward(
        _ model: LoadedModel
    ) -> (MLXArray, [KVCacheProtocol], MLXArray, [Int]) -> MLXArray {
        if useQuantizedBatched(model), let qf = model.batchedDecodeForwardQuantized {
            return { inp, caches, mask, offs in qf(inp, caches.map { $0 as! QuantizedKVCache }, mask, offs) }
        }
        let ff = model.batchedDecodeForward!
        return { inp, caches, mask, offs in ff(inp, caches.map { $0 as! KVCache }, mask, offs) }
    }

    /// Submit ONE request to the engine's persistent continuous batcher
    /// (follow-up #8, Stage C1) and get back its own `(stream, stats)`. Unlike
    /// `generateBatched` (a static cohort decoded to completion), the batcher
    /// admits this row into the *running* batch between steps and drops finished
    /// rows, so concurrent requests share one decode loop with rolling
    /// admission. The batcher is created lazily on the first eligible call
    /// against the current model (`maxRows` = `KRILL_NUM_PARALLEL`, `windowMs` =
    /// the cold-start gather window) and torn down on `unload()`.
    ///
    /// Returns nil when the loaded model is not batch-eligible (unknown family,
    /// nothing loaded, or an empty prompt) so the caller falls back to serial
    /// `generate(...)`. The batched families are text-only Llama 3.x / Qwen
    /// 2.5-3 dense, Gemma 4 (dense + MoE) and Qwen3 MoE; fp16 KV; the shared
    /// prefix cache IS consulted per row (Stage C4); no speculative decode.
    public func submitBatched(
        _ req: BatchGenRequest, maxRows: Int, windowMs: Int
    ) -> (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?)? {
        guard let model = loadedModelForBatching,
              model.batchedDecodeForward != nil,
              let tokenizer, let eosId = tokenizerEOS
        else { return nil }
        let promptTokens = buildBatchedPromptTokens(
            messages: req.messages, contextLimit: req.contextLimit,
            promptTemplateOverride: req.promptTemplateOverride, tokenizer: tokenizer)
        guard !promptTokens.isEmpty else { return nil }

        let stopIds = Self.stopTokenIds(
            modelDirectory: modelDirectory, tokenizerEOS: eosId)

        batcherLock.lock()
        if continuousBatcher == nil {
            let deps = ContinuousBatcher.Deps(
                prefillRow: protocolPrefillRow(model),
                batchedForward: protocolBatchedForward(model),
                numLayers: model.numLayers,
                useQuantizedKV: useQuantizedBatched(model),
                decode: { tokenizer.decode(token: $0) },
                stopIds: stopIds)
            continuousBatcher = ContinuousBatcher(
                deps: deps, maxRows: maxRows, windowMs: windowMs)
        }
        let batcher = continuousBatcher!
        batcherLock.unlock()
        return batcher.submit(promptTokens: promptTokens, req: req)
    }

    /// Build one row's prompt tokens. This is the non-Gemma-4 (plain-causal)
    /// subset of `generate(...)`'s prompt-prep: a created model's `TEMPLATE`
    /// override, else the tokenizer's chat template (token-id path, no lossy
    /// decode→re-encode), else the manual fallback — then the same num_ctx
    /// suffix truncation. The batched path is text-only, so generate()'s
    /// media-placeholder injection (a no-op without media) and its Gemma 4 / VL
    /// branches (never batch-eligible) are intentionally omitted; for a
    /// text-only Llama/Qwen prompt this yields the identical token stream.
    private func buildBatchedPromptTokens(
        messages: [[String: String]], contextLimit: Int?,
        promptTemplateOverride: String?, tokenizer: KLMTokenizer
    ) -> [Int] {
        var built: [Int]
        if let overrideTemplate = promptTemplateOverride,
           !overrideTemplate.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           let rendered = renderWithOverrideTemplate(overrideTemplate, messages: messages) {
            built = tokenizer.encodeWithoutExtraBOS(rendered)
        } else if let direct = tokenizer.applyChatTemplateTokens(messages: messages) {
            built = direct
        } else {
            built = tokenizer.encodeWithoutExtraBOS(
                tokenizer.applyChatTemplate(messages: messages))
        }
        if let limit = contextLimit, limit > 0, built.count > limit {
            built = Array(built.suffix(limit))
        }
        return built
    }

    /// The shared batched decode loop: per-row prefill, left-padded KV stack,
    /// then one batched forward per step advancing every row by a token, with
    /// per-row sampling / stop / maxTokens and live per-row streaming. Mirrors
    /// the serial standard-decode loop's emit semantics (check the current
    /// token, emit-or-end, then forward to the next) so each row equals its
    /// solo run.
    private static func runBatchedDecode(_ c: BatchedCaptures) {
        let R = c.promptRows.count
        let startTime = CFAbsoluteTimeGetCurrent()

        // Per-row prefill (B=1 path) → per-row cache + first token. The
        // prefix-cache-aware closure restores a full hit's KV (then forwards
        // only the last token, so the cache matches a cold prefill) or prefills
        // cold and stores write-behind, honoring this row's usePrefixCache. It
        // evaluates the logits internally before returning.
        var perRowCaches: [[KVCacheProtocol]] = []
        var lengths: [Int] = []
        var current: [Int] = []
        perRowCaches.reserveCapacity(R)
        lengths.reserveCapacity(R)
        current.reserveCapacity(R)
        for (i, tokens) in c.promptRows.enumerated() {
            let caches: [KVCacheProtocol] = c.useQuantizedKV
                ? makeQuantizedKVCaches(numLayers: c.numLayers)
                : makeKVCaches(numLayers: c.numLayers)
            let logits = c.prefillRow(tokens, caches, c.usePrefixCache[i])
            perRowCaches.append(caches)
            lengths.append(tokens.count)
            current.append(c.samplers[i].sample(logits))
        }
        let prefillDuration = CFAbsoluteTimeGetCurrent() - startTime

        let sMax = lengths.max() ?? 0
        let caches = stackCachesLeftPaddedAny(perRow: perRowCaches, lengths: lengths)
        let kDtype = caches.first?.snapshot()?.keys.dtype ?? .float16

        var generated = [Int](repeating: 0, count: R)
        var done = [Bool](repeating: false, count: R)
        var recent: [[Int]] = (0 ..< R).map {
            c.samplers[$0].needsHistory ? Array(c.promptRows[$0].suffix(512)) : []
        }
        let decodeStart = CFAbsoluteTimeGetCurrent()

        // Emit `current[r]` for one row, applying the serial loop's
        // top-of-iteration semantics: maxTokens terminal, stop-token terminal,
        // or a normal text token (with an inline maxTokens terminal when the
        // just-emitted token was the last one).
        func emit(_ r: Int, now: Double) {
            guard !done[r] else { return }
            if generated[r] >= c.perRowMax[r] {
                c.conts[r].yield(TokenEvent(tokenId: -1, text: "", elapsed: now, isEnd: true))
                done[r] = true
                return
            }
            if c.stopIds.contains(current[r]) {
                c.conts[r].yield(TokenEvent(
                    tokenId: current[r], text: "", elapsed: now, isEnd: true))
                done[r] = true
                return
            }
            let text = c.tokenizer.decode(token: current[r])
            c.conts[r].yield(TokenEvent(tokenId: current[r], text: text, elapsed: now))
            generated[r] += 1
            if generated[r] >= c.perRowMax[r] {
                c.conts[r].yield(TokenEvent(tokenId: -1, text: "", elapsed: now, isEnd: true))
                done[r] = true
            }
        }

        var step = 0
        let maxStep = (c.perRowMax.max() ?? 0) + 4   // safety bound
        while !done.allSatisfy({ $0 }) {
            if Task.isCancelled { break }
            let now = CFAbsoluteTimeGetCurrent() - startTime
            for r in 0 ..< R { emit(r, now: now) }
            if done.allSatisfy({ $0 }) { break }

            // One batched forward advances every row by one token. Done rows
            // stay in the batch (their output is ignored) — the v1 no-shrink
            // simplification — so the stacked layout / offsets stay aligned.
            let rowOffsets = (0 ..< R).map { lengths[$0] + step }
            let totalLen = sMax + step + 1
            let mask = buildBatchedDecodeMask(
                lengths: lengths, sMax: sMax, totalLen: totalLen, dtype: kDtype)
            let input = MLXArray(current.map { Int32($0) }).reshaped(R, 1)
            let logits = c.batchedForward(input, caches, mask, rowOffsets)
            MLX.eval(logits)
            for r in 0 ..< R {
                let rowLogits = logits[r ..< (r + 1)]
                if c.samplers[r].needsHistory {
                    if !done[r] { recent[r].append(current[r]) }
                    current[r] = c.samplers[r].sample(rowLogits, recent: recent[r])
                } else {
                    current[r] = c.samplers[r].sample(rowLogits)
                }
            }
            step += 1
            if step > maxStep { break }
        }
        let decodeDuration = CFAbsoluteTimeGetCurrent() - decodeStart

        let end = CFAbsoluteTimeGetCurrent() - startTime
        for r in 0 ..< R {
            // A row that did not reach a natural terminal (cancel / safety
            // bound) still gets a terminal event so consumers always see one.
            if !done[r] {
                c.conts[r].yield(TokenEvent(tokenId: -1, text: "", elapsed: end, isEnd: true))
            }
            c.statsHolders[r].stats = GenerationStats(
                promptTokens: lengths[r], generatedTokens: generated[r],
                prefillTime: prefillDuration, decodeTime: decodeDuration)
            c.conts[r].finish()
        }
    }
}

// MARK: - Internal Helpers

/// Thread-safe holder for generation statistics.
private final class StatsHolder: @unchecked Sendable {
    var stats: GenerationStats?
}
