import Foundation
import KrillCore
import KrillTokenizer

// Neural Engine (Core ML) backend of EmbeddingEngine. A directory holding `krill_ane.json`
// loads here instead of through the MLX path. Text up to the package's `max_tokens` is
// embedded by `ANEEmbedder`; text over the largest bucket and every image/audio/video input
// go to an ordinary MLX EmbeddingGemma 2 engine (`fallback`), loaded lazily on first need.

extension EmbeddingEngine {
    /// Load an ANE package. `embedder` is injectable for tests; production opens Core ML.
    func loadANE(directory: URL, fallbackDirectory: URL?, embedder injected: ANEEmbedder? = nil) throws {
        let manifest = try ANEPackageManifest.load(directory: directory)
        let embedder: ANEEmbedder
        if let injected {
            embedder = injected
        } else {
            #if canImport(CoreML)
            embedder = try ANEEmbedder.coreML(directory: directory, manifest: manifest)
            #else
            throw EmbeddingError.unavailable(
                "the Neural Engine embedder needs Core ML (macOS on Apple Silicon); set embed_device = gpu")
            #endif
        }
        let cp = try CodePointBPETokenizer(directory: directory)
        let siblings = directory.deletingLastPathComponent()
        let fb = fallbackDirectory ?? siblings.appendingPathComponent(manifest.fallbackModel)
        let promptTable = EmbeddingPromptTable.load(directory: directory)
        withLock {
            self.model = nil
            self.tokenizer = nil
            self.cpTokenizer = cp
            self.loadedDir = directory
            self.maxTokens = EmbeddingGemma2Config.maxContext
            self.prompts = promptTable
            self.mrlDimensions = EmbeddingGemma2Config.mrlDimensions
            self.ane = embedder
            self.aneFallbackDir = fb
        }
        dropFallback()
    }

    /// The MLX build the loaded ANE package falls back to. Changing it drops the engine built from the old one.
    func retargetFallback(_ dir: URL?) {
        guard let dir else { return }
        let changed: Bool = withLock {
            guard ane != nil, aneFallbackDir?.standardizedFileURL != dir.standardizedFileURL else { return false }
            aneFallbackDir = dir
            return true
        }
        if changed { dropFallback() }
    }

    func fallbackEngine() -> EmbeddingEngine? {
        fallbackLock.lock(); defer { fallbackLock.unlock() }
        return fallback
    }

    /// Release the fallback engine (and its MLX weights). True if one was resident.
    @discardableResult
    func dropFallback() -> Bool {
        fallbackLock.lock()
        let f = fallback
        fallback = nil
        fallbackLock.unlock()
        guard let f else { return false }
        _ = f.unload()
        return true
    }

    /// The fallback engine, loading it on first use.
    func ensureFallback() throws -> EmbeddingEngine {
        fallbackLock.lock(); defer { fallbackLock.unlock() }
        if let fallback { return fallback }
        guard let dir = withLock({ aneFallbackDir }) else { throw EmbeddingError.notLoaded }
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path) else {
            throw EmbeddingError.unavailable(
                "the Neural Engine package needs the MLX model '\(dir.lastPathComponent)' for long text and media; install it with: krill pull \(dir.lastPathComponent)")
        }
        let e = EmbeddingEngine()
        try e.loadEG2Sync(directory: dir)
        fallback = e
        return e
    }

    /// Text embedding on the ANE. Order-preserving; items longer than the largest bucket
    /// (after the request's `max_length` cap) are embedded by the fallback.
    func embedANE(_ ane: ANEEmbedder, tokenizer cp: CodePointBPETokenizer, texts: [String],
                  options: EmbeddingRequestOptions, cap: Int,
                  prompts: EmbeddingPromptTable?, mrl: [Int]?) throws -> EmbedResult {
        let prefix = try Self.resolvePrefix(options, prompts: prompts, mrl: mrl)
        var out = [[Float]](repeating: [], count: texts.count)
        var total = 0
        var overflow: [Int] = []
        for (i, text) in texts.enumerated() {
            var ids = cp.encode(prefix + text)
            // Truncate keeping the trailing <eos>, like HF `truncation=True` (same as the MLX path).
            if ids.count > cap, let eos = cp.eosId { ids = Array(ids.prefix(cap - 1)) + [eos] }
            else if ids.count > cap { ids = Array(ids.prefix(cap)) }
            if ids.count > ane.maxTokens { overflow.append(i); continue }
            total += ids.count
            let (raw, _) = try ane.embed(tokenIDs: ids.map { Int32($0) })
            guard raw.allSatisfy({ $0.isFinite }) else { throw EmbeddingError.nonFinite }
            let v = EmbeddingMath.l2Normalize(raw)
            out[i] = options.dimensions.map { EmbeddingMath.truncate(v, to: $0) } ?? v
        }
        guard !overflow.isEmpty else {
            return EmbedResult(vectors: out, promptTokens: total, servedBy: .ane)
        }
        let r = try ensureFallback().embed(overflow.map { texts[$0] }, options: options)
        for (k, i) in overflow.enumerated() { out[i] = r.vectors[k] }
        return EmbedResult(vectors: out, promptTokens: total + r.promptTokens,
                           servedBy: overflow.count == texts.count ? .gpu : .mixed)
    }

    /// Mixed or media input on an ANE package: text items on the ANE, media items on MLX.
    func embedANEInputs(_ inputs: [EmbeddingInput], options: EmbeddingRequestOptions) throws -> EmbedResult {
        let (ane, cp, prompts, mrl) = withLock { (self.ane, self.cpTokenizer, self.prompts, self.mrlDimensions) }
        guard let ane, let cp else { throw EmbeddingError.notLoaded }
        var out = [[Float]](repeating: [], count: inputs.count)
        var total = 0
        let textIdx = inputs.indices.filter { !inputs[$0].hasMedia }
        let mediaIdx = inputs.indices.filter { inputs[$0].hasMedia }
        let cap = EmbeddingRequestOptions.effectiveTokenCap(
            modelCap: EmbeddingGemma2Config.maxContext, requestCap: options.maxLength)
        var served: EmbedServedBy = .gpu
        if !textIdx.isEmpty {
            let r = try embedANE(ane, tokenizer: cp, texts: textIdx.map { inputs[$0].joinedText },
                                 options: options, cap: cap, prompts: prompts, mrl: mrl)
            for (k, i) in textIdx.enumerated() { out[i] = r.vectors[k] }
            total += r.promptTokens
            served = r.servedBy
        }
        let m = try ensureFallback().embed(inputs: mediaIdx.map { inputs[$0] }, options: options)
        for (k, i) in mediaIdx.enumerated() { out[i] = m.vectors[k] }
        total += m.promptTokens
        if !textIdx.isEmpty, served != .gpu { served = .mixed }
        return EmbedResult(vectors: out, promptTokens: total, servedBy: served)
    }
}
