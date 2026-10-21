import Foundation
import KrillCore

// Platform-neutral half of the Neural Engine (Core ML) embedding backend:
// package manifest, bucket selection and the fp16 embedding table. Nothing in
// this file imports Core ML, so it builds and is unit-tested everywhere.

/// `krill_ane.json`: describes an ANE embedder package (a Krill model directory
/// such as `embeddinggemma-2-ane-lin8`). Built offline by
/// `tools/ane/build_ane_embedder.py`; see docs/ANE_EMBEDDER.md.
public struct ANEPackageManifest: Codable, Equatable, Sendable {
    public static let fileName = "krill_ane.json"
    public static let backendCoreML = "coreml"
    public static let supportedFormat = 2

    public struct Bucket: Codable, Equatable, Sendable {
        /// Padded sequence length of the compiled model.
        public let seq: Int
        /// Batch size of the compiled model (1 for the shipped buckets).
        public let batch: Int
        /// Compiled `.mlmodelc` directory, relative to the package.
        public let path: String
        public init(seq: Int, batch: Int = 1, path: String) {
            self.seq = seq; self.batch = batch; self.path = path
        }
    }

    public struct Table: Codable, Equatable, Sendable {
        /// Raw little-endian fp16 matrix `[vocab, dim]`, row-major, relative to the package.
        public let file: String
        public let vocab: Int
        public let dim: Int
        public let dtype: String
        public init(file: String, vocab: Int, dim: Int, dtype: String = "float16") {
            self.file = file; self.vocab = vocab; self.dim = dim; self.dtype = dtype
        }
    }

    public let backend: String
    /// 1 = EmbeddingGemma 2 only (PR 343); 2 adds the per-model fields below. Format 1 is still read.
    public let format: Int
    /// Weight variant label: `lin8` (int8 per-channel, default) or `fp16`.
    public let variant: String
    /// Output (pre-truncation) embedding width; 1 for a reranker (the score).
    public let outputDim: Int
    /// Host-side multiplier applied to gathered embedding rows (`sqrt(hidden)` for Gemma, 1 elsewhere).
    public let embedScale: Float
    /// Longest sequence the ANE buckets serve; longer inputs go to the fallback model.
    public let maxTokens: Int
    public let buckets: [Bucket]
    public let table: Table
    /// Registry name of the MLX model that serves what the buckets cannot (the GPU build of the same model).
    public let fallbackModel: String
    // ---- format 2 ----
    /// `embedder` or `reranker`.
    public let kind: String
    /// Registry name of the model this package was built from (nil for format 1: EmbeddingGemma 2).
    public let model: String?
    /// `mean`, `cls`, `last_token` (embedders) or `classifier` (cross-encoder score). Baked into the graph.
    public let pooling: String
    /// Where the token-embedding rows are gathered: `host` (the mapped fp16 table). `model` is reserved.
    public let gather: String
    /// `codepoint_bpe` (EmbeddingGemma 2) or `krill` (the tokenizer Krill's GPU path uses for that model).
    public let tokenizer: String
    /// Name of the Core ML output.
    public let outputName: String
    /// The model's own context in tokens (what the GPU path truncates to); defaults to `maxTokens`.
    public let contextTokens: Int
    /// Decoder embedders pool the last token, so exactly one trailing EOS is appended.
    public let appendEOS: Bool
    /// Matryoshka widths the model supports (nil = not an MRL model).
    public let mrlDims: [Int]?

    public static let kindEmbedder = "embedder"
    public static let kindReranker = "reranker"
    public static let tokenizerCodePointBPE = "codepoint_bpe"
    public static let tokenizerKrill = "krill"

    public init(backend: String = ANEPackageManifest.backendCoreML, format: Int = 1,
                variant: String, outputDim: Int, embedScale: Float, maxTokens: Int,
                buckets: [Bucket], table: Table, fallbackModel: String,
                kind: String = ANEPackageManifest.kindEmbedder, model: String? = nil, pooling: String = "mean",
                gather: String = "host", tokenizer: String = ANEPackageManifest.tokenizerCodePointBPE,
                outputName: String = "emb", contextTokens: Int? = nil, appendEOS: Bool = false,
                mrlDims: [Int]? = nil) {
        self.backend = backend; self.format = format; self.variant = variant
        self.outputDim = outputDim; self.embedScale = embedScale; self.maxTokens = maxTokens
        self.buckets = buckets; self.table = table; self.fallbackModel = fallbackModel
        self.kind = kind; self.model = model; self.pooling = pooling; self.gather = gather
        self.tokenizer = tokenizer; self.outputName = outputName
        self.contextTokens = contextTokens ?? maxTokens; self.appendEOS = appendEOS; self.mrlDims = mrlDims
    }

    enum CodingKeys: String, CodingKey {
        case backend, format, variant, buckets, table, kind, model, pooling, gather, tokenizer
        case outputDim = "output_dim", embedScale = "embed_scale", maxTokens = "max_tokens"
        case fallbackModel = "fallback_model", outputName = "output_name", contextTokens = "context_tokens"
        case appendEOS = "append_eos", mrlDims = "mrl_dims"
    }

    /// Format 1 has none of the format-2 keys; they default to the EmbeddingGemma 2 values it implied.
    public init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        let fmt = try c.decode(Int.self, forKey: .format)
        let max = try c.decode(Int.self, forKey: .maxTokens)
        self.init(backend: try c.decode(String.self, forKey: .backend), format: fmt,
                  variant: try c.decode(String.self, forKey: .variant),
                  outputDim: try c.decode(Int.self, forKey: .outputDim),
                  embedScale: try c.decode(Float.self, forKey: .embedScale), maxTokens: max,
                  buckets: try c.decode([Bucket].self, forKey: .buckets),
                  table: try c.decode(Table.self, forKey: .table),
                  fallbackModel: try c.decode(String.self, forKey: .fallbackModel),
                  kind: try c.decodeIfPresent(String.self, forKey: .kind) ?? Self.kindEmbedder,
                  model: try c.decodeIfPresent(String.self, forKey: .model),
                  pooling: try c.decodeIfPresent(String.self, forKey: .pooling) ?? "mean",
                  gather: try c.decodeIfPresent(String.self, forKey: .gather) ?? "host",
                  tokenizer: try c.decodeIfPresent(String.self, forKey: .tokenizer) ?? Self.tokenizerCodePointBPE,
                  outputName: try c.decodeIfPresent(String.self, forKey: .outputName) ?? "emb",
                  contextTokens: try c.decodeIfPresent(Int.self, forKey: .contextTokens),
                  appendEOS: try c.decodeIfPresent(Bool.self, forKey: .appendEOS) ?? false,
                  mrlDims: try c.decodeIfPresent([Int].self, forKey: .mrlDims))
    }

    public func encode(to e: Encoder) throws {
        var c = e.container(keyedBy: CodingKeys.self)
        try c.encode(backend, forKey: .backend); try c.encode(format, forKey: .format)
        try c.encode(variant, forKey: .variant); try c.encode(outputDim, forKey: .outputDim)
        try c.encode(embedScale, forKey: .embedScale); try c.encode(maxTokens, forKey: .maxTokens)
        try c.encode(buckets, forKey: .buckets); try c.encode(table, forKey: .table)
        try c.encode(fallbackModel, forKey: .fallbackModel)
        if format >= 2 {
            try c.encode(kind, forKey: .kind); try c.encodeIfPresent(model, forKey: .model)
            try c.encode(pooling, forKey: .pooling); try c.encode(gather, forKey: .gather)
            try c.encode(tokenizer, forKey: .tokenizer); try c.encode(outputName, forKey: .outputName)
            try c.encode(contextTokens, forKey: .contextTokens); try c.encode(appendEOS, forKey: .appendEOS)
            try c.encodeIfPresent(mrlDims, forKey: .mrlDims)
        }
    }

    public enum ManifestError: Error, CustomStringConvertible, Equatable {
        case invalid(String)
        public var description: String {
            if case .invalid(let m) = self { return "invalid \(ANEPackageManifest.fileName): \(m)" }
            return ""
        }
    }

    /// True when `directory` holds an ANE package.
    public static func exists(in directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(fileName).path)
    }

    public static func load(directory: URL) throws -> ANEPackageManifest {
        let data = try Data(contentsOf: directory.appendingPathComponent(fileName))
        let m = try JSONDecoder().decode(ANEPackageManifest.self, from: data)
        try m.validate()
        return m
    }

    /// Format 1 is EmbeddingGemma 2 only; the matryoshka widths it implied.
    public var effectiveMRLDims: [Int]? { format == 1 ? EmbeddingGemma2Config.mrlDimensions : mrlDims }

    public func validate() throws {
        guard backend == Self.backendCoreML else { throw ManifestError.invalid("unknown backend '\(backend)'") }
        guard (1 ... Self.supportedFormat).contains(format) else { throw ManifestError.invalid("unsupported format \(format)") }
        guard kind == Self.kindEmbedder || kind == Self.kindReranker else { throw ManifestError.invalid("unknown kind '\(kind)'") }
        guard gather == "host" else { throw ManifestError.invalid("unsupported gather '\(gather)'") }
        guard tokenizer == Self.tokenizerCodePointBPE || tokenizer == Self.tokenizerKrill else {
            throw ManifestError.invalid("unknown tokenizer '\(tokenizer)'")
        }
        guard ["mean", "cls", "last_token", "classifier"].contains(pooling) else { throw ManifestError.invalid("unknown pooling '\(pooling)'") }
        guard !outputName.isEmpty, contextTokens > 0 else { throw ManifestError.invalid("bad output_name or context_tokens") }
        let single = buckets.filter { $0.batch == 1 }
        guard !single.isEmpty else { throw ManifestError.invalid("no batch-1 bucket") }
        guard Set(single.map(\.seq)).count == single.count else { throw ManifestError.invalid("duplicate bucket length") }
        guard single.allSatisfy({ $0.seq > 0 }) else { throw ManifestError.invalid("bucket length must be positive") }
        guard let top = single.map(\.seq).max(), top >= maxTokens else {
            throw ManifestError.invalid("largest bucket \(single.map(\.seq).max() ?? 0) is below max_tokens \(maxTokens)")
        }
        guard table.dtype == "float16", table.vocab > 0, table.dim > 0 else { throw ManifestError.invalid("bad table") }
        guard outputDim > 0, embedScale > 0, !fallbackModel.isEmpty else { throw ManifestError.invalid("bad scalar field") }
    }

    /// Batch-1 bucket lengths, ascending.
    public var singleBucketLengths: [Int] { buckets.filter { $0.batch == 1 }.map(\.seq).sorted() }
}

/// Smallest-bucket-that-fits selection.
public enum ANEBucketPlanner {
    /// The smallest bucket length >= `tokens`, or nil when the input exceeds every bucket
    /// (it is then served by the fallback model).
    public static func bucket(forTokens tokens: Int, lengths: [Int]) -> Int? {
        lengths.sorted().first { $0 >= max(1, tokens) }
    }
}

/// fp16 <-> fp32 by bit manipulation. Used instead of `Float16` so the host-side
/// code compiles on every platform (`Float16` is unavailable on x86 macOS).
public enum Half {
    @inline(__always)
    public static func toFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let e = UInt32(h >> 10) & 0x1f
        var m = UInt32(h & 0x3ff)
        if e == 0 {
            if m == 0 { return Float(bitPattern: sign) }
            var ee: UInt32 = 113
            while m & 0x400 == 0 { m <<= 1; ee -= 1 }
            m &= 0x3ff
            return Float(bitPattern: sign | (ee << 23) | (m << 13))
        }
        if e == 31 { return Float(bitPattern: sign | 0x7f80_0000 | (m << 13)) }
        return Float(bitPattern: sign | ((e + 112) << 23) | (m << 13))
    }

    /// Round-to-nearest-even.
    @inline(__always)
    public static func fromFloat(_ f: Float) -> UInt16 {
        let x = f.bitPattern
        let sign = UInt16((x >> 16) & 0x8000)
        let ef = Int((x >> 23) & 0xff)
        var man = x & 0x7f_ffff
        if ef == 0xff { return sign | 0x7c00 | (man != 0 ? 0x200 : 0) }
        let exp = ef - 127 + 15
        if exp >= 31 { return sign | 0x7c00 }
        if exp <= 0 {
            if exp < -10 { return sign }
            man |= 0x80_0000
            let shift = UInt32(14 - exp)
            var h = man >> shift
            let rem = man & ((1 << shift) - 1)
            let half: UInt32 = 1 << (shift - 1)
            if rem > half || (rem == half && (h & 1) == 1) { h += 1 }
            return sign | UInt16(h)
        }
        var h = (UInt32(exp) << 10) | (man >> 13)
        let rem = man & 0x1fff
        if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) { h += 1 }
        return sign | UInt16(truncatingIfNeeded: h)
    }
}

/// The token-embedding table, memory-mapped. Only the pages of rows that are
/// actually gathered ever become resident; the 268 MB file never counts as a
/// private allocation.
public final class ANEEmbeddingTable: @unchecked Sendable {
    public let vocab: Int
    public let dim: Int
    private let data: Data   // .alwaysMapped

    public init(url: URL, vocab: Int, dim: Int) throws {
        let d = try Data(contentsOf: url, options: .alwaysMapped)
        guard d.count == vocab * dim * 2 else {
            throw ANEPackageManifest.ManifestError.invalid(
                "embedding table is \(d.count) bytes, expected \(vocab * dim * 2) for \(vocab)x\(dim) fp16")
        }
        self.data = d; self.vocab = vocab; self.dim = dim
    }

    /// Write `ids.count` scaled rows into `dst` (fp16 bit patterns, row-major `[n, dim]`).
    /// Out-of-range ids map to row 0 (`<pad>`); the tokenizer never produces them.
    public func gather(ids: [Int32], scale: Float, into dst: UnsafeMutablePointer<UInt16>) {
        data.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: UInt16.self)
            for (r, id) in ids.enumerated() {
                let row = (id >= 0 && Int(id) < vocab) ? Int(id) : 0
                let src = base.baseAddress! + row * dim
                let out = dst + r * dim
                for c in 0 ..< dim { out[c] = Half.fromFloat(Half.toFloat(src[c]) * scale) }
            }
        }
    }
}

/// Which backend produced an embedding response (header `x-krill-embed-device`, debug log, tests).
public enum EmbedServedBy: String, Sendable, Equatable {
    case gpu      // MLX
    case ane      // Core ML on the Neural Engine
    case mixed    // an ANE request with some items (long text, media) served by MLX
}

/// Platform support for the Neural Engine backend.
public enum ANEPlatform {
    /// Core ML on Apple Silicon. Everything Core ML-specific is behind `canImport(CoreML)`.
    public static var coreMLAvailable: Bool {
        #if canImport(CoreML) && arch(arm64)
        return true
        #else
        return false
        #endif
    }
}
