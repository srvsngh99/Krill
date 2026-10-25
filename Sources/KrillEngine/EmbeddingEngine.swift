import Foundation
import MLX
import MLXNN
import KrillCore
import KrillTokenizer

/// Loads and runs a *dedicated* sentence-embedding model (BERT-style
/// encoder: sentence-transformers / BGE / MiniLM / E5). Separate from
/// `InferenceEngine` so embeddings do not require - or disturb - a loaded
/// chat model, and so a RAG client can embed while chat is unloaded.
public final class EmbeddingEngine: @unchecked Sendable {
    var model: (any SentenceEmbeddingEncoder)?
    var tokenizer: KrillTokenizer?
    /// Code-point BPE used by EmbeddingGemma 2 (swift-transformers' BPE seeds
    /// merges with grapheme clusters, which breaks Indic scripts; see
    /// `CodePointBPETokenizer`). Set only for that model.
    var cpTokenizer: CodePointBPETokenizer?
    var loadedDir: URL?
    var maxTokens: Int = 512
    private var pooling = EmbeddingPooling.fromEnv()
    /// Decoder-LLM embedders (last-token pooling) append an EOS so the pooled
    /// final position has attended over the whole input; BERT encoders do not.
    private var appendEOS = false
    private var eosTokenId = 0
    /// sentence-transformers `prompts` table of the loaded model (task -> prefix).
    var prompts: EmbeddingPromptTable?
    /// Matryoshka dimensions the loaded model supports (nil = not an MRL model).
    var mrlDimensions: [Int]?
    let lock = NSLock()
    /// EmbeddingGemma 2 modality towers are loaded LAZILY, on the first request
    /// that carries media, so text-only use keeps the text-only footprint.
    private var visionTower: EG2VisionTower?
    private var audioTower: EG2AudioTower?
    private var eg2Dtype: DType = .float32
    /// Per-forward padded-token budget default of the loaded EG2 build (see `modelDefaultBudget`).
    private var eg2DefaultBudget = EmbeddingEngine.defaultPaddedTokenBudget
    private var eg2Tokens = EG2ModalityTokens.checkpointDefaults
    let towerLock = NSLock()
    /// Neural Engine (Core ML) backend, set instead of `model` when the loaded directory is an
    /// ANE package (`krill_ane.json`). Text inputs up to the package's `max_tokens` run here;
    /// longer inputs and media go to `fallback`, an ordinary MLX EmbeddingGemma 2 engine that
    /// is created lazily and unloaded with this engine. See EmbeddingEngine+ANE.swift.
    var ane: ANEEmbedder?
    var aneFallbackDir: URL?
    /// The package's tokenizer when it is Krill's own (every package but EmbeddingGemma 2). Also seeds the MLX fallback.
    var aneTokenizer: KrillTokenizer?
    var fallback: EmbeddingEngine?
    let fallbackLock = NSLock()

    public init() {}

    public var loadedModelName: String? {
        withLock { loadedDir?.lastPathComponent }
    }

    public func isLoaded(directory: URL) -> Bool {
        withLock {
            loadedDir?.standardizedFileURL == directory.standardizedFileURL
                && (model != nil || ane != nil)
        }
    }

    func withLock<T>(_ body: () -> T) -> T {
        lock.lock(); defer { lock.unlock() }
        return body()
    }

    private func install(model: any SentenceEmbeddingEncoder, tokenizer: KrillTokenizer?,
                         cpTokenizer: CodePointBPETokenizer? = nil,
                         directory: URL, maxTokens: Int,
                         pooling: EmbeddingPooling, appendEOS: Bool, eosTokenId: Int,
                         prompts: EmbeddingPromptTable?, mrlDimensions: [Int]?) {
        withLock {
            self.prompts = prompts
            self.mrlDimensions = mrlDimensions
            self.model = model
            self.tokenizer = tokenizer
            self.cpTokenizer = cpTokenizer
            self.loadedDir = directory
            self.maxTokens = maxTokens
            self.pooling = pooling
            self.appendEOS = appendEOS
            self.eosTokenId = eosTokenId
            self.visionTower = nil
            self.audioTower = nil
            self.ane = nil
            self.aneFallbackDir = nil
            self.aneTokenizer = nil
        }
        dropFallback()
    }

    /// Load (or hot-swap to) the embedding model in `directory`.
    ///
    /// `fallbackDirectory` only matters for a Neural Engine package (`krill_ane.json`): the
    /// MLX EmbeddingGemma 2 build that serves what the package cannot (text over its largest
    /// bucket, images, audio, video). Default: the sibling directory named by the package manifest.
    public func load(directory: URL, fallbackDirectory: URL? = nil) async throws {
        if isLoaded(directory: directory) {
            retargetFallback(fallbackDirectory)
            return
        }
        if ANEPackageManifest.exists(in: directory) {
            let m = try ANEPackageManifest.load(directory: directory)
            let tok: KrillTokenizer? = m.tokenizer == ANEPackageManifest.tokenizerKrill
                ? try await KrillTokenizer(from: directory) : nil
            try loadANE(directory: directory, fallbackDirectory: fallbackDirectory, krillTokenizer: tok)
            return
        }

        let configURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let mt = Self.modelType(from: data) ?? ""
        // EmbeddingGemma 2 uses its own code-point BPE; skip the (slow, 32 MB)
        // swift-transformers load for it.
        let tok: KrillTokenizer? = mt == "embedding_gemma2"
            ? nil : try await KrillTokenizer(from: directory)
        try loadResolved(directory: directory, data: data, mt: mt, tok: tok)
    }

    /// Synchronous load of an MLX EmbeddingGemma 2 build (the Neural Engine package's fallback is
    /// loaded lazily from inside the synchronous `embed`). Other model types need the async `load`.
    func loadEG2Sync(directory: URL) throws {
        if isLoaded(directory: directory) { return }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard Self.modelType(from: data) == "embedding_gemma2" else {
            throw EmbeddingError.unavailable("Neural Engine fallback must be an EmbeddingGemma 2 build: \(directory.lastPathComponent)")
        }
        try loadResolved(directory: directory, data: data, mt: "embedding_gemma2", tok: nil)
    }

    func loadResolved(directory: URL, data: Data, mt: String, tok: KrillTokenizer?) throws {
        // Bound MLX's Metal buffer-recycling pool. The embed path never went through
        // `loadModel` (which applies this for chat models), so the pool was left at MLX's
        // default (the device working set, ~16 GB on a 24 GB Mac). Embedding requests
        // allocate intermediates sized by (batch x padded length); every NEW shape needs
        // buffers the pool cannot reuse, so the pool grew to 17 GB over 25 requests while
        // MLX's active memory stayed at 0.2 GB. With the cap, freed buffers go back to the OS.
        MLXMemoryConfig.apply(defaultMB: Self.defaultCacheLimitMB)

        var cpTok: CodePointBPETokenizer? = nil

        let model: any SentenceEmbeddingEncoder
        let maxTokens: Int
        let pooling: EmbeddingPooling
        var appendEOS = false
        var mrl: [Int]? = nil

        if mt == "embedding_gemma2" {
            // EmbeddingGemma 2 (text path): bidirectional Gemma-4-derived encoder
            // with a 512->768 projection, mean pooling, MRL dims. Strict-bound
            // (see loadEmbeddingGemma2); vision/audio tensors are skipped.
            // float32 compute by default; float16 is unsafe for this model.
            let dtype: DType = Self.envDtype() ?? .float32
            cpTok = try CodePointBPETokenizer(directory: directory)
            let loaded = try loadEmbeddingGemma2(directory: directory, dtype: dtype)
            withLock {
                eg2Dtype = dtype; eg2Tokens = loaded.model.config.modalityTokens
                eg2DefaultBudget = Self.modelDefaultBudget(quantized: loaded.report.quantizedText > 0)
            }
            model = loaded.model
            maxTokens = EmbeddingGemma2Config.maxContext
            pooling = .mean
            mrl = EmbeddingGemma2Config.mrlDimensions
        } else if mt == "nomic_bert",
           let v2 = try? JSONDecoder().decode(NomicBertV2Config.self, from: data), v2.isMoE {
            // nomic-embed-text-v2-moe: same `nomic_bert` model_type as v1.5 but a
            // top-2 mixture of experts on every 2nd layer (XLM-R vocab). Detected
            // by the MoE config fields. Keys match the checkpoint; strict verify
            // guards mismatch. fp32 weights, mean-pooled.
            let m = NomicBertV2MoEModel(v2)
            try loadWeights(into: m, from: directory, quantization: nil,
                            keyPrefix: nil, strictVerify: true)
            eval(m)
            model = m
            maxTokens = v2.maxTokens
            pooling = Self.envPooling() ?? .mean
        } else if mt == "nomic_bert" {
            // nomic-embed-text: a RoPE encoder (fused Wqkv + SwiGLU), distinct
            // from vanilla BERT/RoBERTa. Checkpoint keys already match the module
            // keys (no `bert.`/`roberta.` prefix); strict verify guards mismatch.
            let config = try JSONDecoder().decode(NomicBertConfig.self, from: data)
            let m = NomicBertEmbeddingModel(config)
            try loadWeights(into: m, from: directory, quantization: nil,
                            keyPrefix: nil, strictVerify: true)
            eval(m)
            model = m
            maxTokens = config.maxTokens
            pooling = Self.envPooling() ?? .mean
        } else if mt == "mpnet" {
            // MPNet: relative-attention-bias encoder, no token-type embeddings,
            // RoBERTa-style offset positions. The checkpoint ships a `pooler` and
            // a `position_ids` buffer this encoder does not use; drop them so a
            // strict-verify update sees an exact key match.
            let config = try JSONDecoder().decode(MPNetConfig.self, from: data)
            let m = MPNetEmbeddingModel(config)
            try loadWeights(into: m, from: directory, quantization: nil, keyPrefix: nil,
                            keyRewrite: { weights in
                                for key in weights.keys
                                where key.hasPrefix("pooler.") || key == "embeddings.position_ids" {
                                    weights.removeValue(forKey: key)
                                }
                            }, strictVerify: true)
            eval(m)
            model = m
            maxTokens = config.maxTokens
            pooling = Self.envPooling() ?? .mean
        } else if mt == "new" {
            // GTE-v1.5 ("NewModel"): RoPE encoder with biased fused qkv, GeGLU
            // MLP, post-norm, no token-type. CLS-pooled. Keys match 1:1.
            let config = try JSONDecoder().decode(GTEConfig.self, from: data)
            let m = GTEEmbeddingModel(config)
            try loadWeights(into: m, from: directory, quantization: nil,
                            keyPrefix: nil, strictVerify: true)
            eval(m)
            model = m
            maxTokens = config.maxTokens
            pooling = Self.envPooling()
                ?? Self.sentenceTransformerPooling(directory: directory) ?? .cls
        } else if mt == "modernbert" {
            // ModernBERT: pre-norm RoPE encoder with alternating global/local
            // attention (per-layer theta + sliding window), GeGLU, weight-only
            // norms, no biases. Keys map 1:1 (layer 0 has no attn_norm).
            let config = try JSONDecoder().decode(ModernBertConfig.self, from: data)
            let m = ModernBertEmbeddingModel(config)
            try loadWeights(into: m, from: directory, quantization: nil,
                            keyPrefix: nil, strictVerify: true)
            // ModernBERT ships fp16 weights but its activations overflow fp16
            // (GeGLU intermediates run large); upcast to fp32 for a stable,
            // reference-matching forward.
            m.update(parameters: m.parameters().mapValues { $0.asType(.float32) })
            eval(m)
            model = m
            maxTokens = config.maxTokens
            pooling = Self.envPooling()
                ?? Self.sentenceTransformerPooling(directory: directory) ?? .cls
        } else if Self.causalEmbedderTypes.contains(mt) {
            // Decoder-LLM embedder (gte-Qwen2, e5-mistral, ...): reuse the
            // already-validated causal backbone via the shared loader, then pool
            // its final hidden state. Last-token pooling appends an EOS upstream
            // so the pooled position has attended over the whole sequence.
            let loaded = try loadModel(from: directory)
            guard let enc = loaded.module as? (any SentenceEmbeddingEncoder) else {
                throw EmbeddingError.unsupported(mt)
            }
            // fp16 decoder backbones (e5-mistral, SFR, ...) carry massive
            // residual-stream activations that overflow fp16 (-> inf -> the
            // final RMSNorm divides to an all-zero vector, so every embedding
            // comes back zero). A full fp32 upcast would double a 7B past this
            // host's RAM, so upcast only embed_tokens: that seeds the residual
            // stream in fp32 and MLX promotes the fp16 weight matmuls to fp32
            // from there, keeping the stream fp32 end to end for one extra
            // embedding table. No-op when the backbone is already fp32.
            let upcast = loaded.module.parameters().flattened()
                .filter { $0.0.contains("embed_tokens") && $0.1.dtype == .float16 }
                .map { ($0.0, $0.1.asType(.float32)) }
            if !upcast.isEmpty {
                loaded.module.update(parameters: ModuleParameters.unflattened(upcast))
            }
            model = enc
            pooling = Self.envPooling()
                ?? Self.sentenceTransformerPooling(directory: directory) ?? .lastToken
            appendEOS = (pooling == .lastToken)
            // Cap context to keep a single embed forward bounded (these backbones
            // advertise 100k+ positions); agent-scale chunks sit far under it.
            maxTokens = min(Self.maxPositionEmbeddings(from: data) ?? 8192, 8192)
        } else if Self.isJinaBert(data) {
            // jina-embeddings-v2: model_type is "bert" but it uses ALiBi (no
            // positional embeddings) and a GLU MLP, so it must NOT route to the
            // vanilla BERT loader. Drop the unused `pooler.*`; upcast fp16 -> fp32.
            let config = try JSONDecoder().decode(JinaBertConfig.self, from: data)
            let m = JinaBertEmbeddingModel(config)
            try loadWeights(into: m, from: directory, quantization: nil, keyPrefix: nil,
                            keyRewrite: { weights in
                                for key in weights.keys where key.hasPrefix("pooler.") {
                                    weights.removeValue(forKey: key)
                                }
                            }, strictVerify: true)
            m.update(parameters: m.parameters().mapValues { $0.asType(.float32) })
            eval(m)
            model = m
            maxTokens = config.maxTokens
            pooling = Self.envPooling()
                ?? Self.sentenceTransformerPooling(directory: directory) ?? .mean
        } else {
            let config = try JSONDecoder().decode(BertEmbeddingConfig.self, from: data)
            let m = BertEmbeddingModel(config)
            // BertModel checkpoints may prefix keys with `bert.`/`roberta.`.
            let raw = try loadWeightArrays(from: directory)
            let prefix: String? =
                raw.keys.contains { $0.hasPrefix("bert.") } ? "bert."
                : raw.keys.contains { $0.hasPrefix("roberta.") } ? "roberta."
                : nil
            try loadWeights(into: m, from: directory, quantization: nil, keyPrefix: prefix)
            eval(m)
            model = m
            maxTokens = config.maxPositionEmbeddings
            pooling = Self.envPooling() ?? .mean
        }

        install(model: model, tokenizer: tok, cpTokenizer: cpTok, directory: directory,
                maxTokens: maxTokens, pooling: pooling,
                appendEOS: appendEOS, eosTokenId: tok?.eosTokenId ?? cpTok?.eosId ?? 0,
                prompts: EmbeddingPromptTable.load(directory: directory),
                mrlDimensions: mrl)
    }

    /// `KRILL_EMBED_DTYPE=float32|bfloat16` for EmbeddingGemma 2. float16 is
    /// deliberately not accepted (NaN / silent degradation).
    private static func envDtype() -> DType? {
        switch ProcessInfo.processInfo.environment["KRILL_EMBED_DTYPE"]?.lowercased() {
        case "float32", "fp32", "f32": return .float32
        case "bfloat16", "bf16": return .bfloat16
        default: return nil
        }
    }

    /// Peek the `model_type` field from a raw config.json to select the encoder
    /// architecture without committing to a full config decode.
    static func modelType(from configData: Data) -> String? {
        struct Peek: Decodable {
            let modelType: String?
            enum CodingKeys: String, CodingKey { case modelType = "model_type" }
        }
        return (try? JSONDecoder().decode(Peek.self, from: configData))?.modelType
    }

    /// jina-embeddings-v2 declares `model_type: "bert"` but uses ALiBi; route it
    /// to the JinaBERT encoder rather than the vanilla BERT loader. Detected by
    /// `position_embedding_type == "alibi"` or a `JinaBert*` architecture.
    private static func isJinaBert(_ configData: Data) -> Bool {
        struct Peek: Decodable {
            let positionEmbeddingType: String?
            let architectures: [String]?
            enum CodingKeys: String, CodingKey {
                case positionEmbeddingType = "position_embedding_type"
                case architectures
            }
        }
        guard let p = try? JSONDecoder().decode(Peek.self, from: configData) else { return false }
        if p.positionEmbeddingType?.lowercased() == "alibi" { return true }
        return p.architectures?.contains { $0.lowercased().contains("jinabert") } ?? false
    }

    public struct EmbedResult: Sendable {
        public let vectors: [[Float]]
        public let promptTokens: Int
        /// Which backend produced the vectors: `gpu` (MLX), `ane` (Neural Engine) or
        /// `mixed` (a Neural Engine request that sent some items to MLX).
        public var servedBy: EmbedServedBy = .gpu
        public init(vectors: [[Float]], promptTokens: Int, servedBy: EmbedServedBy = .gpu) {
            self.vectors = vectors; self.promptTokens = promptTokens; self.servedBy = servedBy
        }
    }

    /// Embed a batch of texts. Generic encoders run each text independently
    /// (batch=1) so no padding mask is needed and the forward stays exact.
    /// EmbeddingGemma 2 batches by length with a key-padding mask.
    ///
    /// `options.task` / `options.instruction` prefix every text;
    /// `options.dimensions` truncates (MRL models) and re-normalises. Invalid
    /// combinations throw `EmbeddingError.invalidOption` (a client error).
    public func embed(_ texts: [String],
                      options: EmbeddingRequestOptions = EmbeddingRequestOptions()) throws -> EmbedResult {
        lock.lock()
        let model = self.model
        let tokenizer = self.tokenizer
        let cpTokenizer = self.cpTokenizer
        let cap = EmbeddingRequestOptions.effectiveTokenCap(
            modelCap: self.maxTokens, requestCap: options.maxLength)
        let pooling = self.pooling
        let appendEOS = self.appendEOS
        let eos = self.eosTokenId
        let prompts = self.prompts
        let mrl = self.mrlDimensions
        let budget = Self.paddedTokenBudget(modelDefault: self.eg2DefaultBudget)
        let aneEmbedder = self.ane
        lock.unlock()

        if let aneEmbedder, let tk = aneTextTokenizer() {
            return try embedANE(aneEmbedder, tokenizer: tk, texts: texts, options: options,
                                cap: cap, prompts: prompts, mrl: mrl)
        }
        guard let model else { throw EmbeddingError.notLoaded }

        let prefix = try Self.resolvePrefix(options, prompts: prompts, mrl: mrl)

        var vectors: [[Float]] = []
        var totalTokens = 0

        if let eg = model as? EmbeddingGemma2Model {
            guard let cpTokenizer else { throw EmbeddingError.notLoaded }
            return try embedGemma2(eg, tokenizer: cpTokenizer, texts: texts.map { prefix + $0 },
                                   cap: cap, dimensions: options.dimensions,
                                   tokenBudget: budget)
        }
        guard let tokenizer else { throw EmbeddingError.notLoaded }

        vectors.reserveCapacity(texts.count)
        for text in texts {
            var ids = tokenizer.encode(prefix + text)
            if ids.isEmpty { ids = [tokenizer.bosTokenId] }
            if appendEOS {
                // Last-token decoder embedders pool the EOS position. Normalize
                // to exactly one trailing EOS (some tokenizers, e.g. e5-mistral,
                // already append it; others, e.g. gte-Qwen2, do not), reserving a
                // slot for it when truncating.
                if ids.last == eos { ids.removeLast() }
                if ids.count > cap - 1 { ids = Array(ids.prefix(cap - 1)) }
                ids.append(eos)
            } else if ids.count > cap {
                ids = Array(ids.prefix(cap))
            }
            totalTokens += ids.count

            let tokens = MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
            let hidden = model.lastHiddenState(tokens)
            vectors.append(
                poolSentenceEmbedding(hidden, pooling: pooling, normalize: true))
        }

        // Non-MRL models: `dimensions` is ignored, exactly as before this field
        // was supported (existing models' behaviour must not change).
        return EmbedResult(vectors: vectors, promptTokens: totalTokens)
    }

    /// Resolve the prefix (task table or literal instruction) and validate
    /// `dimensions` against the loaded model.
    static func resolvePrefix(_ options: EmbeddingRequestOptions,
                                      prompts: EmbeddingPromptTable?, mrl: [Int]?) throws -> String {
        var prefix = options.instruction ?? ""
        if let task = options.task {
            guard let table = prompts else {
                throw EmbeddingError.invalidOption(
                    "this model defines no task prompts; use 'instruction' for a literal prefix")
            }
            guard let p = table.prefix(for: task) else {
                throw EmbeddingError.invalidOption(
                    "unknown task '\(task)'; valid tasks: \(table.taskNames.joined(separator: ", "))")
            }
            prefix = p
        }
        if let d = options.dimensions, let mrl, !mrl.contains(d) {
            throw EmbeddingError.invalidOption(
                "unsupported 'dimensions' \(d); this model supports \(mrl.map(String.init).joined(separator: ", "))")
        }
        return prefix
    }

    /// Embed inputs that may carry media parts (see `EmbeddingInput`). All-text
    /// input takes EXACTLY the pre-existing `embed(_:options:)` path for every
    /// model. Media is accepted only by EmbeddingGemma 2 (others: HTTP 400).
    ///
    /// Text-only items inside a mixed request are batched like before (and keep
    /// the 8,192-token truncation); items with media run one at a time, are
    /// never truncated (an input over the context is a 400), and the task prefix
    /// applies to their text only, placed first. `promptTokens` counts every
    /// token including soft tokens and markers.
    public func embed(inputs: [EmbeddingInput],
                      options: EmbeddingRequestOptions = EmbeddingRequestOptions()) throws -> EmbedResult {
        if inputs.allSatisfy({ !$0.hasMedia }) {
            return try embed(inputs.map { $0.joinedText }, options: options)
        }
        if withLock({ ane != nil }) { return try embedANEInputs(inputs, options: options) }
        lock.lock()
        let model = self.model
        let cpTokenizer = self.cpTokenizer
        let prompts = self.prompts
        let mrl = self.mrlDimensions
        let tokens = self.eg2Tokens
        let budget = Self.paddedTokenBudget(modelDefault: self.eg2DefaultBudget)
        lock.unlock()
        guard let model else { throw EmbeddingError.notLoaded }
        guard let eg = model as? EmbeddingGemma2Model, let cp = cpTokenizer else {
            throw EmbeddingError.invalidOption(
                "this model accepts text input only; send images to a multimodal embedder such as embeddinggemma-2")
        }
        let prefix = try Self.resolvePrefix(options, prompts: prompts, mrl: mrl)
        let builder = EG2SequenceBuilder(tokenizer: cp, tokens: tokens)

        var out = [[Float]](repeating: [], count: inputs.count)
        var total = 0
        // 1. text-only items: batched exactly like the text path.
        let textIdx = inputs.indices.filter { !inputs[$0].hasMedia }
        if !textIdx.isEmpty {
            let r = try embedGemma2(eg, tokenizer: cp,
                                    texts: textIdx.map { prefix + inputs[$0].joinedText },
                                    cap: EmbeddingRequestOptions.effectiveTokenCap(
                                        modelCap: EmbeddingGemma2Config.maxContext, requestCap: options.maxLength),
                                    dimensions: options.dimensions,
                                    tokenBudget: budget)
            for (k, i) in textIdx.enumerated() { out[i] = r.vectors[k] }
            total += r.promptTokens
        }
        // 2. media items, one at a time.
        for i in inputs.indices where inputs[i].hasMedia {
            let (vec, n) = try embedMediaItem(inputs[i], model: eg, builder: builder,
                                              prefix: prefix, dimensions: options.dimensions)
            out[i] = vec
            total += n
        }
        logPeakMemory()
        MLX.Memory.clearCache()
        return EmbedResult(vectors: out, promptTokens: total)
    }

    /// One media item: preprocess every part, build the token sequence, run the
    /// towers, scatter their soft tokens, run the backbone, mean-pool.
    private func embedMediaItem(_ item: EmbeddingInput, model: EmbeddingGemma2Model,
                                builder: EG2SequenceBuilder, prefix: String,
                                dimensions: Int?) throws -> ([Float], Int) {
        var segments: [EG2Segment] = []
        var decoded: [EG2RGBImage] = []
        var pendingWaves: [[Float]] = []
        var pendingVideos: [(src: EG2VideoSource, indices: [Int])] = []
        for part in item.parts {
            switch part {
            case .text(let t):
                segments.append(.text(t))
            case .image(let data):
                // Decode and size first (cheap); the pixel work only starts once the
                // whole prompt is known to fit the context.
                do {
                    let rgb = try EG2ImagePreprocessor.decode(data)
                    segments.append(.media(EG2MediaBlock(
                        .image, softTokensPerBlock: try EG2ImagePreprocessor.softTokens(for: rgb))))
                    decoded.append(rgb)
                } catch { throw EmbeddingError.invalidOption("image: \(error)") }
            case .audio(let data, let format):
                // Decode and count first (cheap); mel + tower only run once the whole
                // prompt is known to fit.
                do {
                    let wave = try EG2AudioPreprocessor.decode(data, formatHint: format)
                    let n = try EG2AudioPreprocessor.softTokens(forWaveform: wave)
                    segments.append(.media(EG2MediaBlock(.audio, softTokensPerBlock: n)))
                    pendingWaves.append(wave)
                } catch { throw EmbeddingError.invalidOption("audio: \(error)") }
            case .video(let data, let format):
                // Container read + frame count only (no decode); the sampled frames are
                // decoded after the whole prompt is known to fit.
                do {
                    let src = try EG2VideoSource(data: data, formatHint: format)
                    let idx = EG2VideoSampler.frameIndices(
                        totalFrames: src.info.totalFrames, fps: src.info.fps, duration: src.info.duration)
                    guard !idx.isEmpty else { throw EG2VideoError.noFrames }
                    let n = try EG2ImagePreprocessor.softTokens(
                        width: src.info.width, height: src.info.height,
                        maxSoftTokens: EG2ImagePreprocessor.videoFrameSoftTokens)
                    segments.append(.media(EG2MediaBlock(.video, softTokensPerBlock: n, blocks: idx.count)))
                    pendingVideos.append((src, idx))
                } catch { throw EmbeddingError.invalidOption("video: \(error)") }
            }
        }
        let seq: EG2Sequence
        do { seq = try builder.build(segments, prefix: prefix) }
        catch { throw EmbeddingError.invalidOption("\(error)") }

        var features: [EG2Modality: [MLXArray]] = [:]
        if !decoded.isEmpty || !pendingVideos.isEmpty {
            let tower = try ensureVisionTower()  // images and video share one tower
            if !decoded.isEmpty {
                do {
                    // Evaluate each image's soft tokens right away (numerically identical): left lazy,
                    // every image/frame's tower intermediates would stay alive in one graph until
                    // the final pooled.eval(), so the transient peak grew with the item count.
                    features[.image] = try decoded.map {
                        let t = tower.softTokens(try EG2ImagePreprocessor.prepare($0)); t.eval(); return t
                    }
                } catch { throw EmbeddingError.invalidOption("image: \(error)") }
            }
            if !pendingVideos.isEmpty {
                do {
                    // One `[n, hidden]` array per frame, in order (the builder emits one span per frame).
                    var perFrame: [MLXArray] = []
                    for v in pendingVideos {
                        // Each sampled frame is resized + patchified as it is decoded.
                        let patches = try v.src.frames(at: v.indices) {
                            try EG2ImagePreprocessor.prepare($0, maxSoftTokens: EG2ImagePreprocessor.videoFrameSoftTokens)
                        }
                        perFrame += patches.map { let t = tower.softTokens($0); t.eval(); return t }
                    }
                    features[.video] = perFrame
                } catch { throw EmbeddingError.invalidOption("video: \(error)") }
            }
        }
        if !pendingWaves.isEmpty {
            let tower = try ensureAudioTower()
            do {
                features[.audio] = try pendingWaves.map {
                    let t = try tower.softTokens(try EG2AudioPreprocessor.prepare(waveform: $0)); t.eval(); return t
                }
            } catch { throw EmbeddingError.invalidOption("audio: \(error)") }
        }
        let embeds: MLXArray
        do { embeds = try model.mergedEmbeddings(seq, features: features) }
        catch { throw EmbeddingError.invalidOption("\(error)") }
        let pooled = model.pooled(inputsEmbeds: embeds, lengths: [seq.count])
        pooled.eval()
        let full = pooled.asArray(Float.self)
        guard full.allSatisfy({ $0.isFinite }) else { throw EmbeddingError.nonFinite }
        let v = EmbeddingMath.l2Normalize(full)
        return (dimensions.map { EmbeddingMath.truncate(v, to: $0) } ?? v, seq.count)
    }

    /// Load the vision tower on first use (strict binding). Throws a client
    /// error when the checkpoint has no vision tensors.
    private func ensureVisionTower() throws -> EG2VisionTower {
        towerLock.lock(); defer { towerLock.unlock() }
        if let t = visionTower { return t }
        lock.lock(); let dir = loadedDir; let dtype = eg2Dtype; lock.unlock()
        guard let dir else { throw EmbeddingError.notLoaded }
        do {
            let t = try loadEG2VisionTower(directory: dir, dtype: dtype).tower
            visionTower = t
            logPeakMemory(label: "after vision tower load")
            return t
        } catch EG2VisionLoadError.noVisionTower {
            throw EmbeddingError.invalidOption("this checkpoint has no vision tower; images are not supported")
        }
    }

    /// Load the audio tower on first use (strict binding), separately from vision.
    private func ensureAudioTower() throws -> EG2AudioTower {
        towerLock.lock(); defer { towerLock.unlock() }
        if let t = audioTower { return t }
        lock.lock(); let dir = loadedDir; let dtype = eg2Dtype; lock.unlock()
        guard let dir else { throw EmbeddingError.notLoaded }
        do {
            let t = try loadEG2AudioTower(directory: dir, dtype: dtype).tower
            audioTower = t
            return t
        } catch EG2AudioLoadError.noAudioTower {
            throw EmbeddingError.invalidOption("this checkpoint has no audio tower; audio is not supported")
        }
    }

    /// Drop the vision tower only (text tower stays resident); frees its weights and
    /// MLX's buffer cache. The next image/video request reloads it. Returns true if resident.
    @discardableResult
    public func unloadVisionTower() -> Bool {
        towerLock.lock()
        let was = visionTower != nil
        visionTower = nil
        towerLock.unlock()
        if was { MLX.Memory.clearCache() }
        return (fallbackEngine()?.unloadVisionTower() ?? false) || was
    }

    /// Drop the audio tower only (see ``unloadVisionTower()``).
    @discardableResult
    public func unloadAudioTower() -> Bool {
        towerLock.lock()
        let was = audioTower != nil
        audioTower = nil
        towerLock.unlock()
        if was { MLX.Memory.clearCache() }
        return (fallbackEngine()?.unloadAudioTower() ?? false) || was
    }

    /// True once the lazily loaded audio tower is resident (tests / diagnostics).
    public var isAudioTowerLoaded: Bool {
        towerLock.lock(); defer { towerLock.unlock() }
        return audioTower != nil || (fallbackEngine()?.isAudioTowerLoaded ?? false)
    }

    /// True once the lazily loaded vision tower is resident (tests / diagnostics).
    public var isVisionTowerLoaded: Bool {
        towerLock.lock(); defer { towerLock.unlock() }
        return visionTower != nil || (fallbackEngine()?.isVisionTowerLoaded ?? false)
    }

    private func logPeakMemory(label: String = "") {
        if ProcessInfo.processInfo.environment["KRILL_EMBED_LOG_MEM"] != nil {
            // MLX unified-memory high-water mark (RSS does not see it).
            let mb = Double(Memory.peakMemory) / 1_048_576
            FileHandle.standardError.write(Data(String(
                format: "EmbeddingGemma2: mlx peak memory %.0f MB (active %.0f MB, cache %.0f MB) \(label)\n",
                mb, Double(Memory.activeMemory) / 1_048_576, Double(Memory.cacheMemory) / 1_048_576).utf8))
        }
    }

    /// Padded-batch EmbeddingGemma 2 path. Texts are sorted by length and run
    /// in chunks bounded by a padded-token budget; results return in input
    /// order. Every vector is NaN/Inf-guarded.
    private func embedGemma2(_ model: EmbeddingGemma2Model, tokenizer: CodePointBPETokenizer,
                             texts: [String], cap: Int, dimensions: Int?,
                             tokenBudget: Int) throws -> EmbedResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        var encoded: [[Int32]] = texts.map { t in
            var ids = tokenizer.encode(t)
            // Truncate keeping the trailing <eos>, like HF `truncation=True`.
            if ids.count > cap, let eos = tokenizer.eosId { ids = Array(ids.prefix(cap - 1)) + [eos] }
            else if ids.count > cap { ids = Array(ids.prefix(cap)) }
            return ids.map { Int32($0) }
        }
        let tokenizeMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let total = encoded.reduce(0) { $0 + $1.count }
        let order = encoded.indices.sorted { encoded[$0].count < encoded[$1].count }
        var out = [[Float]](repeating: [], count: texts.count)

        let chunks = Self.planChunks(sortedLengths: order.map { encoded[$0].count },
                                     budget: tokenBudget, maxBatch: Self.gemma2MaxBatch)
        for chunk in chunks {
            let idx = Array(order[chunk])
            let T = idx.map { encoded[$0].count }.max() ?? 1
            var flat = [Int32](); flat.reserveCapacity(idx.count * T)
            for k in idx { flat += encoded[k] + [Int32](repeating: 0, count: T - encoded[k].count) }
            let tokens = MLXArray(flat).reshaped(idx.count, T)
            let pooled = model.pooled(tokens, lengths: idx.map { encoded[$0].count })
            pooled.eval()
            let rows = pooled.asArray(Float.self)
            let w = model.config.embeddingDim
            for (r, k) in idx.enumerated() {
                let full = Array(rows[(r * w) ..< ((r + 1) * w)])
                guard full.allSatisfy({ $0.isFinite }) else { throw EmbeddingError.nonFinite }
                let v = EmbeddingMath.l2Normalize(full)
                out[k] = dimensions.map { EmbeddingMath.truncate(v, to: $0) } ?? v
            }
        }
        encoded.removeAll()
        if ProcessInfo.processInfo.environment["KRILL_EMBED_LOG_MEM"] != nil {
            FileHandle.standardError.write(Data(String(
                format: "EmbeddingGemma2: %d texts, %d tokens, tokenize %.0f ms, total %.0f ms\n",
                texts.count, total, tokenizeMs, (CFAbsoluteTimeGetCurrent() - t0) * 1000).utf8))
        }
        logPeakMemory()
        return EmbedResult(vectors: out, promptTokens: total)
    }

    private static let gemma2MaxBatch = 32

    /// Default per-forward budget of padded tokens (sequences x padded length).
    /// A request is split into sub-batches that each stay within it; one
    /// sequence longer than the budget still runs alone. Peak MLX transient
    /// memory scales with this number, not with the request size. Each item's
    /// vector is independent of the split (key-padding mask + length-aware pooling).
    public static let defaultPaddedTokenBudget = 512

    /// MLX buffer-cache cap used by the embedding path when
    /// `KRILL_MLX_CACHE_LIMIT_MB` is unset. Embedding forwards allocate
    /// transients that never repeat a shape, so a large recycling pool only
    /// inflates the footprint (measured: the 256 MB chat default adds ~280 MB to the
    /// lifetime peak at the default budget). Chat models keep 256.
    public static let defaultCacheLimitMB = 64
    public static let batchTokensEnvVar = "KRILL_EMBED_MAX_BATCH_TOKENS"

    /// `KRILL_EMBED_MAX_BATCH_TOKENS` (positive integer) or the default.
    public static func paddedTokenBudget(
        modelDefault: Int = defaultPaddedTokenBudget,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        if let raw = environment[batchTokensEnvVar]?.trimmingCharacters(in: .whitespaces),
           let n = Int(raw), n > 0 { return n }
        return modelDefault
    }

    /// Default budget for the full-precision (non-quantized) EmbeddingGemma 2 build, the
    /// bulk-embedding build whose weights already put it well past 1 GB (fp32 compute
    /// copy of the bf16 weights): a larger forward is ~15% faster at +0.2 GB peak
    /// (measured, M4 Pro: 28.4 -> 32.6 passages/s). Quantized builds stay at 512, where
    /// throughput is flat in the budget and the lifetime peak stays under 1 GB.
    public static let defaultPaddedTokenBudgetFullPrecision = 2048

    /// Budget default for a loaded EmbeddingGemma 2 build.
    static func modelDefaultBudget(quantized: Bool) -> Int {
        quantized ? defaultPaddedTokenBudget : defaultPaddedTokenBudgetFullPrecision
    }

    /// Split `lengths` (sorted ascending by the caller) into consecutive
    /// chunks of indices where count x longest <= budget and count <= maxBatch.
    /// A chunk always holds at least one item. Pure; unit-tested.
    static func planChunks(sortedLengths lengths: [Int], budget: Int, maxBatch: Int) -> [Range<Int>] {
        var out: [Range<Int>] = []
        var i = 0
        while i < lengths.count {
            var j = i
            while j < lengths.count, (j - i + 1) <= maxBatch,
                  (j - i + 1) * lengths[j] <= budget || j == i { j += 1 }
            out.append(i ..< j)
            i = j
        }
        return out
    }

    /// Drop the loaded model, tokenizers and lazily loaded towers, then return
    /// MLX's buffer cache to the OS so the process footprint falls back to the
    /// bare-process baseline. The next `embed`/`load` reloads transparently.
    /// Returns true if a model was resident.
    @discardableResult
    public func unload() -> Bool {
        let was: Bool = withLock {
            // A Neural Engine embedder is released, not torn down: a request that already
            // holds it finishes on its own reference and Core ML frees the buckets after it.
            let had = model != nil || ane != nil
            model = nil; tokenizer = nil; cpTokenizer = nil; loadedDir = nil
            prompts = nil; mrlDimensions = nil
            ane = nil; aneFallbackDir = nil; aneTokenizer = nil
            return had
        }
        let hadFallback = dropFallback()
        towerLock.lock(); visionTower = nil; audioTower = nil; towerLock.unlock()
        if was || hadFallback { MLX.Memory.clearCache() }
        // The tokenizer's vocabulary tables and Core ML's buffers were malloc'd; hand the freed
        // pages back so a resting Neural Engine server really returns to a small footprint.
        if was { malloc_zone_pressure_relief(nil, 0) }
        return was
    }

    public var isResident: Bool { withLock { model != nil || ane != nil } }

    // MARK: - Decoder-LLM embedder detection

    /// Causal base architectures that can be repurposed as sentence embedders.
    /// Limited to the dense families that conform to `SentenceEmbeddingEncoder`
    /// (see `DecoderEmbedder.swift`), so an unsupported backbone is rejected at
    /// the gate (400) rather than admitted and failing in `load` (500). Add a
    /// family here only once its `*ForCausalLM` conforms (e.g. Gemma for
    /// bge-multilingual-gemma2). Qwen3 MoE has model_type `qwen3_moe`, so it does
    /// not match `qwen3` here.
    private static let causalEmbedderTypes: Set<String> = [
        "qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2",
    ]

    /// True when `directory` holds a decoder-LLM embedder: a causal base arch
    /// plus a sentence-transformers `1_Pooling/config.json`. Used by the server
    /// to admit such models through the embeddings endpoint (their family is
    /// `.qwen`/`.mistral`/... not `.bert`).
    public static func isDecoderEmbedder(directory: URL) -> Bool {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              let mt = modelType(from: data), causalEmbedderTypes.contains(mt) else {
            return false
        }
        return FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("1_Pooling/config.json").path)
    }

    /// Explicit `KRILL_EMBED_POOLING` override, or nil when unset (so each model
    /// keeps its natural default: mean for BERT, last-token for decoder embedders).
    private static func envPooling() -> EmbeddingPooling? {
        guard let v = ProcessInfo.processInfo.environment["KRILL_EMBED_POOLING"] else {
            return nil
        }
        return EmbeddingPooling.from(v)
    }

    private static func maxPositionEmbeddings(from data: Data) -> Int? {
        struct Peek: Decodable {
            let maxPos: Int?
            enum CodingKeys: String, CodingKey { case maxPos = "max_position_embeddings" }
        }
        return (try? JSONDecoder().decode(Peek.self, from: data))?.maxPos
    }

    /// Read the pooling mode from a sentence-transformers `1_Pooling/config.json`.
    private static func sentenceTransformerPooling(directory: URL) -> EmbeddingPooling? {
        let url = directory.appendingPathComponent("1_Pooling/config.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        if json["pooling_mode_lasttoken"] as? Bool == true { return .lastToken }
        if json["pooling_mode_cls_token"] as? Bool == true { return .cls }
        if json["pooling_mode_mean_tokens"] as? Bool == true { return .mean }
        return nil
    }
}

public enum EmbeddingError: Error, CustomStringConvertible {
    case notLoaded
    case unsupported(String)
    /// A request option the loaded model cannot honour (HTTP 400).
    case invalidOption(String)
    /// The forward produced NaN/Inf (HTTP 500; never returned as a vector).
    case nonFinite
    /// A backend the request needs is not available (e.g. Core ML off-platform, or the MLX
    /// fallback model of a Neural Engine package is not installed). HTTP 500 with this message.
    case unavailable(String)

    public var description: String {
        switch self {
        case .notLoaded:
            return "No embedding model loaded"
        case .unsupported(let mt):
            return "Model type '\(mt)' is not a supported embedder"
        case .invalidOption(let m):
            return m
        case .unavailable(let m):
            return m
        case .nonFinite:
            return "embedding produced non-finite values (NaN/Inf); refusing to return a degraded vector"
        }
    }
}
