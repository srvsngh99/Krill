import Foundation

/// One loaded, fixed-shape bucket. `run` fills the (reused) input buffers, predicts
/// and returns the model's output row. Implementations serialise their own buffers.
public protocol ANEBucketRunner: AnyObject, Sendable {
    var seq: Int { get }
    /// - Parameter fill: writes `ids.count` gathered rows into the embeds buffer
    ///   (`[seq, hidden]` fp16, pre-zeroed beyond the real tokens) and the fp16 mask.
    func run(tokenCount: Int, fill: (_ embeds: UnsafeMutablePointer<UInt16>, _ mask: UnsafeMutablePointer<UInt16>) -> Void) throws -> [Float]
}

/// Neural Engine embedder: tokens in, L2-normalised vector out.
///
/// - Picks the smallest compiled bucket that fits, pads to it and masks the padding.
/// - Gathers rows from the mmapped fp16 table on the host, scaled by `embed_scale`.
/// - Loads the smallest (primary) bucket first; larger ones on demand. Idle larger
///   buckets stay resident until the embedder is released (``unloadAll()`` drops them at once).
///
/// Thread safety: ``embed(tokenIDs:)`` may be called concurrently. It takes a strong
/// reference to the bucket it uses, so an unload racing a running prediction only
/// drops the embedder's own reference; the prediction finishes on the reference it
/// holds and the model is freed after it.
public final class ANEEmbedder: @unchecked Sendable {
    public typealias RunnerFactory = @Sendable (_ bucket: ANEPackageManifest.Bucket, _ directory: URL) throws -> ANEBucketRunner

    public let manifest: ANEPackageManifest
    public let directory: URL
    private let table: ANEEmbeddingTable
    private let factory: RunnerFactory
    private let lock = NSLock()
    private let loadLock = NSLock()      // one compile at a time, without blocking predictions on resident buckets
    private var runners: [Int: ANEBucketRunner] = [:]
    private var generation = 0

    public init(directory: URL, manifest: ANEPackageManifest, factory: @escaping RunnerFactory) throws {
        self.directory = directory
        self.manifest = manifest
        self.factory = factory
        self.table = try ANEEmbeddingTable(
            url: directory.appendingPathComponent(manifest.table.file),
            vocab: manifest.table.vocab, dim: manifest.table.dim)
    }

    /// Smallest bucket length; always resident while the embedder is.
    public var primaryBucket: Int { manifest.singleBucketLengths[0] }
    public var maxTokens: Int { manifest.maxTokens }

    public var loadedBuckets: [Int] {
        lock.lock(); defer { lock.unlock() }
        return runners.keys.sorted()
    }

    /// Load the primary bucket now (the first ANE compile can take seconds).
    public func warmPrimary() throws { _ = try runner(forSeq: primaryBucket) }

    private func runner(forSeq seq: Int) throws -> ANEBucketRunner {
        lock.lock()
        if let r = runners[seq] { lock.unlock(); return r }
        lock.unlock()
        loadLock.lock(); defer { loadLock.unlock() }
        lock.lock()
        if let r = runners[seq] { lock.unlock(); return r }
        let gen = generation
        lock.unlock()
        guard let spec = manifest.buckets.first(where: { $0.seq == seq && $0.batch == 1 }) else {
            throw EmbeddingError.invalidOption("no ANE bucket of length \(seq)")
        }
        let r = try factory(spec, directory)
        lock.lock(); defer { lock.unlock() }
        // An unload that raced the (slow) compile wins: do not resurrect the bucket.
        if gen == generation { runners[seq] = r }
        return r
    }

    /// Embed one token sequence (BOS/EOS already included, `1...maxTokens` long).
    /// Returns the model's L2-normalised output as Float and the bucket that served it.
    public func embed(tokenIDs ids: [Int32]) throws -> (vector: [Float], bucket: Int) {
        let ids = ids.isEmpty ? [0] : ids
        guard let seq = ANEBucketPlanner.bucket(forTokens: ids.count, lengths: manifest.singleBucketLengths),
              ids.count <= manifest.maxTokens else {
            throw EmbeddingError.invalidOption("sequence of \(ids.count) tokens exceeds the ANE buckets")
        }
        let r = try runner(forSeq: seq)
        let scale = manifest.embedScale
        let table = self.table
        let v = try r.run(tokenCount: ids.count) { embeds, mask in
            table.gather(ids: ids, scale: scale, into: embeds)
            for i in 0 ..< ids.count { mask[i] = 0x3C00 }   // fp16 1.0
        }
        return (v, seq)
    }

    /// Drop every bucket (the table mapping goes when the embedder is released).
    @discardableResult
    public func unloadAll() -> Bool {
        lock.lock(); defer { lock.unlock() }
        generation += 1
        let had = !runners.isEmpty
        runners.removeAll()
        return had
    }
}

#if canImport(CoreML)
import CoreML

/// Core ML bucket: one compiled model plus input/output buffers allocated once and
/// reused for every prediction. Predictions run inside `autoreleasepool` - a loop
/// that allocated an `MLMultiArray` per call without one grew ~70 KB per call at
/// S=64 and ~500 KB at S=512 (3 GB over 6,000 calls) in the spike.
final class CoreMLBucketRunner: ANEBucketRunner, @unchecked Sendable {
    let seq: Int
    private let model: MLModel
    private let hidden: Int
    private let embeds: MLMultiArray
    private let mask: MLMultiArray
    private let output: MLMultiArray
    private let provider: MLDictionaryFeatureProvider
    private let options = MLPredictionOptions()
    private let lock = NSLock()
    private let outDim: Int

    init(url: URL, seq: Int, hidden: Int, outDim: Int, units: MLComputeUnits) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        self.model = try MLModel(contentsOf: url, configuration: cfg)
        self.seq = seq; self.hidden = hidden; self.outDim = outDim
        embeds = try MLMultiArray(shape: [1, NSNumber(value: seq), NSNumber(value: hidden)], dataType: .float16)
        mask = try MLMultiArray(shape: [1, NSNumber(value: seq)], dataType: .float16)
        output = try MLMultiArray(shape: [1, NSNumber(value: outDim)], dataType: .float16)
        provider = try MLDictionaryFeatureProvider(dictionary: ["embeds": embeds, "mask": mask])
        options.outputBackings = ["emb": output]
    }

    func run(tokenCount: Int, fill: (UnsafeMutablePointer<UInt16>, UnsafeMutablePointer<UInt16>) -> Void) throws -> [Float] {
        lock.lock(); defer { lock.unlock() }
        let e = embeds.dataPointer.bindMemory(to: UInt16.self, capacity: seq * hidden)
        let m = mask.dataPointer.bindMemory(to: UInt16.self, capacity: seq)
        // Zero the padding region (and the previous request's rows) before filling.
        memset(e, 0, seq * hidden * 2)
        memset(m, 0, seq * 2)
        fill(e, m)
        var out = [Float](repeating: 0, count: outDim)
        var failure: Error?
        autoreleasepool {
            do {
                let result = try model.prediction(from: provider, options: options)
                // With outputBackings the result wraps `output`; read through the result to be safe.
                guard let arr = result.featureValue(for: "emb")?.multiArrayValue else {
                    throw EmbeddingError.unsupported("Core ML model has no 'emb' output")
                }
                let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: outDim)
                for i in 0 ..< outDim { out[i] = Half.toFloat(p[i]) }
            } catch { failure = error }
        }
        if let failure { throw failure }
        return out
    }
}

public extension ANEEmbedder {
    /// Open a package with Core ML buckets on CPU + Neural Engine.
    static func coreML(directory: URL, manifest: ANEPackageManifest,
                       units: MLComputeUnits = .cpuAndNeuralEngine) throws -> ANEEmbedder {
        let hidden = manifest.table.dim
        let outDim = manifest.outputDim
        return try ANEEmbedder(directory: directory, manifest: manifest) { bucket, dir in
            try CoreMLBucketRunner(url: dir.appendingPathComponent(bucket.path), seq: bucket.seq,
                                   hidden: hidden, outDim: outDim, units: units)
        }
    }
}
#endif
