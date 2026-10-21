import Foundation
import KrillCore
import KrillTokenizer

// Neural Engine (Core ML) backend of EmbeddingEngine. A directory holding `krill_ane.json`
// loads here instead of through the MLX path. Text up to the package's `max_tokens` is
// embedded by `ANEEmbedder`; text over the largest bucket and every image/audio/video input
// go to an ordinary MLX EmbeddingGemma 2 engine (`fallback`), loaded lazily on first need.

/// The tokenizer behind a loaded package: EmbeddingGemma 2's code-point BPE, or Krill's own for every other model.
enum ANETokenizer {
    case codePoint(CodePointBPETokenizer)
    case krill(KrillTokenizer)

    func encode(_ text: String) -> [Int] {
        switch self {
        case .codePoint(let t): return t.encode(text)
        case .krill(let t): return t.encode(text)
        }
    }
    var eosId: Int? {
        switch self {
        case .codePoint(let t): return t.eosId
        case .krill(let t): return t.eosTokenId
        }
    }
    var bosId: Int {
        switch self {
        case .codePoint: return 0
        case .krill(let t): return t.bosTokenId
        }
    }
}

extension EmbeddingEngine {
    /// Load an ANE package. `embedder` is injectable for tests; production opens Core ML. A package whose
    /// manifest names Krill's tokenizer needs `krillTokenizer` (the async `load` builds it).
    func loadANE(directory: URL, fallbackDirectory: URL?, embedder injected: ANEEmbedder? = nil,
                 krillTokenizer: KrillTokenizer? = nil) throws {
        let manifest = try ANEPackageManifest.load(directory: directory)
        guard manifest.kind == ANEPackageManifest.kindEmbedder else {
            throw EmbeddingError.unavailable("\(directory.lastPathComponent) is a reranker package; use /v1/rerank")
        }
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
        var cp: CodePointBPETokenizer?
        if manifest.tokenizer == ANEPackageManifest.tokenizerCodePointBPE {
            cp = try CodePointBPETokenizer(directory: directory)
        } else if krillTokenizer == nil {
            throw EmbeddingError.unavailable("package \(directory.lastPathComponent) needs Krill's tokenizer")
        }
        let siblings = directory.deletingLastPathComponent()
        let fb = fallbackDirectory ?? siblings.appendingPathComponent(manifest.fallbackModel)
        let promptTable = EmbeddingPromptTable.load(directory: directory)
        withLock {
            self.model = nil
            self.tokenizer = nil
            self.cpTokenizer = cp
            self.loadedDir = directory
            self.maxTokens = manifest.format == 1 ? EmbeddingGemma2Config.maxContext : manifest.contextTokens
            self.prompts = promptTable
            self.mrlDimensions = manifest.effectiveMRLDims
            self.ane = embedder
            self.aneFallbackDir = fb
            self.aneTokenizer = krillTokenizer
        }
        dropFallback()
    }

    /// The package's tokenizer, whichever kind.
    func aneTextTokenizer() -> ANETokenizer? {
        withLock {
            if let cp = cpTokenizer { return .codePoint(cp) }
            if let k = aneTokenizer { return .krill(k) }
            return nil
        }
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
        let cfg = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let mt = Self.modelType(from: cfg) ?? ""
        if mt == "embedding_gemma2" {
            try e.loadEG2Sync(directory: dir)
        } else {
            // The package carries the same tokenizer files as the model it was built from.
            guard let tok = withLock({ aneTokenizer }) else { throw EmbeddingError.notLoaded }
            try e.loadResolved(directory: dir, data: cfg, mt: mt, tok: tok)
        }
        fallback = e
        return e
    }

    /// Text embedding on the ANE. Order-preserving; items longer than the largest bucket
    /// (after the request's `max_length` cap) are embedded by the fallback.
    func embedANE(_ ane: ANEEmbedder, tokenizer tk: ANETokenizer, texts: [String],
                  options: EmbeddingRequestOptions, cap: Int,
                  prompts: EmbeddingPromptTable?, mrl: [Int]?) throws -> EmbedResult {
        let prefix = try Self.resolvePrefix(options, prompts: prompts, mrl: mrl)
        let eos = tk.eosId
        var out = [[Float]](repeating: [], count: texts.count)
        var total = 0
        var overflow: [Int] = []
        for (i, text) in texts.enumerated() {
            var ids = tk.encode(prefix + text)
            if case .codePoint = tk {
                // EmbeddingGemma 2: truncate keeping the trailing <eos>, like HF `truncation=True`.
                if ids.count > cap, let eos { ids = Array(ids.prefix(cap - 1)) + [eos] }
                else if ids.count > cap { ids = Array(ids.prefix(cap)) }
            } else {
                ids = Self.shapeIDs(ids, eos: eos, bos: tk.bosId, cap: cap, appendEOS: ane.manifest.appendEOS)
            }
            if ids.count > ane.maxTokens { overflow.append(i); continue }
            total += ids.count
            let (raw, _) = try ane.embed(tokenIDs: ids.map { Int32($0) })
            guard raw.allSatisfy({ $0.isFinite }) else { throw EmbeddingError.nonFinite }
            let v = EmbeddingMath.l2Normalize(raw)
            // `dimensions` only applies to matryoshka models; everything else ignores it, as on the GPU.
            out[i] = (mrl != nil) ? (options.dimensions.map { EmbeddingMath.truncate(v, to: $0) } ?? v) : v
        }
        guard !overflow.isEmpty else {
            return EmbedResult(vectors: out, promptTokens: total, servedBy: .ane)
        }
        let r = try ensureFallback().embed(overflow.map { texts[$0] }, options: options)
        for (k, i) in overflow.enumerated() { out[i] = r.vectors[k] }
        return EmbedResult(vectors: out, promptTokens: total + r.promptTokens,
                           servedBy: overflow.count == texts.count ? .gpu : .mixed)
    }

    /// Token shaping the MLX path applies to every non-Gemma embedder: an empty input becomes one BOS; decoder
    /// (last-token) embedders end in exactly one EOS within the cap; encoders are cut at the cap.
    static func shapeIDs(_ input: [Int], eos: Int?, bos: Int, cap: Int, appendEOS: Bool) -> [Int] {
        var ids = input
        if ids.isEmpty { ids = [bos] }
        if appendEOS, let eos {
            if ids.last == eos { ids.removeLast() }
            if ids.count > cap - 1 { ids = Array(ids.prefix(cap - 1)) }
            ids.append(eos)
        } else if ids.count > cap {
            ids = Array(ids.prefix(cap))
        }
        return ids
    }

    /// Mixed or media input on an ANE package: text items on the ANE, media items on MLX.
    func embedANEInputs(_ inputs: [EmbeddingInput], options: EmbeddingRequestOptions) throws -> EmbedResult {
        let (ane, prompts, mrl) = withLock { (self.ane, self.prompts, self.mrlDimensions) }
        guard let ane, let tk = aneTextTokenizer() else { throw EmbeddingError.notLoaded }
        // Only EmbeddingGemma 2 takes media; any other model's MLX build rejects it with the usual error.
        if case .krill = tk { return try ensureFallback().embed(inputs: inputs, options: options) }
        var out = [[Float]](repeating: [], count: inputs.count)
        var total = 0
        let textIdx = inputs.indices.filter { !inputs[$0].hasMedia }
        let mediaIdx = inputs.indices.filter { inputs[$0].hasMedia }
        let cap = EmbeddingRequestOptions.effectiveTokenCap(
            modelCap: EmbeddingGemma2Config.maxContext, requestCap: options.maxLength)
        var served: EmbedServedBy = .gpu
        if !textIdx.isEmpty {
            let r = try embedANE(ane, tokenizer: tk, texts: textIdx.map { inputs[$0].joinedText },
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
