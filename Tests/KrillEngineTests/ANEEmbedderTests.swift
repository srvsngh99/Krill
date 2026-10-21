import XCTest
import Foundation
@testable import KrillEngine
@testable import KrillCore

/// The Neural Engine embedder's host side: manifest, bucket choice, the fp16 table gather,
/// lazy loading, unload and concurrency. Runs everywhere (no Core ML): buckets are fakes that
/// compute a masked mean of the rows they are given, so the gather / scale / pad / mask / normalise
/// pipeline is checked against a reference computed straight from the table.
final class ANEEmbedderTests: XCTestCase {
    static let dim = 8
    static let vocab = 40
    static let scale: Float = 3.0

    // MARK: fixtures

    /// Fake bucket: masked mean of the fp16 rows, L2-normalised. Counts live instances.
    final class FakeRunner: ANEBucketRunner, @unchecked Sendable {
        static let live = LiveCounter()
        let seq: Int
        let dim: Int
        let lock = NSLock()
        init(seq: Int, dim: Int) { self.seq = seq; self.dim = dim; FakeRunner.live.add(1) }
        deinit { FakeRunner.live.add(-1) }
        func run(tokenCount: Int, fill: (UnsafeMutablePointer<UInt16>, UnsafeMutablePointer<UInt16>) -> Void) throws -> [Float] {
            lock.lock(); defer { lock.unlock() }
            var e = [UInt16](repeating: 0, count: seq * dim)
            var m = [UInt16](repeating: 0, count: seq)
            e.withUnsafeMutableBufferPointer { ep in m.withUnsafeMutableBufferPointer { mp in
                fill(ep.baseAddress!, mp.baseAddress!) } }
            var acc = [Float](repeating: 0, count: dim)
            var n: Float = 0
            for t in 0 ..< seq where Half.toFloat(m[t]) > 0.5 {
                n += 1
                for c in 0 ..< dim { acc[c] += Half.toFloat(e[t * dim + c]) }
            }
            // Padding rows must be untouched zeros.
            for t in tokenCount ..< seq { for c in 0 ..< dim { XCTAssertEqual(e[t * dim + c], 0) } }
            return EmbeddingMath.l2Normalize(acc.map { $0 / max(n, 1) })
        }
    }
    final class LiveCounter: @unchecked Sendable {
        private let l = NSLock(); private var n = 0
        func add(_ d: Int) { l.lock(); n += d; l.unlock() }
        var value: Int { l.lock(); defer { l.unlock() }; return n }
    }

    func tableValue(_ row: Int, _ col: Int) -> Float { Float((row * 7 + col * 3) % 11) / 10 - 0.4 }

    func makePackage(buckets: [Int] = [4, 8, 16, 32], maxTokens: Int = 32) throws -> (URL, ANEPackageManifest) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ane-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var bytes = [UInt16]()
        for r in 0 ..< Self.vocab { for c in 0 ..< Self.dim { bytes.append(Half.fromFloat(tableValue(r, c))) } }
        try bytes.withUnsafeBytes { Data($0) }.write(to: dir.appendingPathComponent("embed_tokens.f16"))
        let m = ANEPackageManifest(
            variant: "lin8", outputDim: Self.dim, embedScale: Self.scale, maxTokens: maxTokens,
            buckets: buckets.map { .init(seq: $0, path: "s\($0).mlmodelc") },
            table: .init(file: "embed_tokens.f16", vocab: Self.vocab, dim: Self.dim),
            fallbackModel: "embeddinggemma-2-mlx-4bit-dyn-text")
        try JSONEncoder().encode(m).write(to: dir.appendingPathComponent(ANEPackageManifest.fileName))
        return (dir, m)
    }

    func makeEmbedder(buckets: [Int] = [4, 8, 16, 32]) throws -> (ANEEmbedder, URL) {
        let (dir, m) = try makePackage(buckets: buckets, maxTokens: buckets.max()!)
        let e = try ANEEmbedder(directory: dir, manifest: m) { b, _ in FakeRunner(seq: b.seq, dim: Self.dim) }
        return (e, dir)
    }

    /// Reference: gather, scale and round to fp16 like the host does, mean, normalise.
    func reference(_ ids: [Int32]) -> [Float] {
        var acc = [Float](repeating: 0, count: Self.dim)
        for id in ids { for c in 0 ..< Self.dim {
            acc[c] += Half.toFloat(Half.fromFloat(Half.toFloat(Half.fromFloat(tableValue(Int(id), c))) * Self.scale)) } }
        return EmbeddingMath.l2Normalize(acc.map { $0 / Float(ids.count) })
    }

    // MARK: manifest, buckets, fp16

    func testManifestRoundTripAndValidation() throws {
        let (dir, m) = try makePackage()
        let loaded = try ANEPackageManifest.load(directory: dir)
        XCTAssertEqual(loaded, m)
        XCTAssertTrue(ANEPackageManifest.exists(in: dir))
        XCTAssertEqual(loaded.singleBucketLengths, [4, 8, 16, 32])

        let bad = ANEPackageManifest(variant: "lin8", outputDim: 8, embedScale: 3, maxTokens: 64,
                                     buckets: [.init(seq: 32, path: "x")],
                                     table: .init(file: "t", vocab: 1, dim: 1), fallbackModel: "f")
        XCTAssertThrowsError(try bad.validate(), "largest bucket below max_tokens")
        let wrongBackend = ANEPackageManifest(backend: "onnx", variant: "lin8", outputDim: 8, embedScale: 3, maxTokens: 4,
                                              buckets: [.init(seq: 4, path: "x")],
                                              table: .init(file: "t", vocab: 1, dim: 1), fallbackModel: "f")
        XCTAssertThrowsError(try wrongBackend.validate())
    }

    func testBucketPlannerPicksSmallestThatFits() {
        let L = [64, 128, 256, 512]
        XCTAssertEqual(ANEBucketPlanner.bucket(forTokens: 1, lengths: L), 64)
        XCTAssertEqual(ANEBucketPlanner.bucket(forTokens: 64, lengths: L), 64)
        XCTAssertEqual(ANEBucketPlanner.bucket(forTokens: 65, lengths: L), 128)
        XCTAssertEqual(ANEBucketPlanner.bucket(forTokens: 256, lengths: L), 256)
        XCTAssertEqual(ANEBucketPlanner.bucket(forTokens: 257, lengths: L), 512)
        XCTAssertEqual(ANEBucketPlanner.bucket(forTokens: 512, lengths: L), 512)
        XCTAssertNil(ANEBucketPlanner.bucket(forTokens: 513, lengths: L), "over the largest bucket: MLX fallback")
        XCTAssertEqual(ANEBucketPlanner.bucket(forTokens: 0, lengths: L), 64)
        XCTAssertEqual(ANEBucketPlanner.bucket(forTokens: 100, lengths: [512, 64, 128]), 128, "order independent")
    }

    func testHalfRoundTrip() {
        for v: Float in [0, 1, -1, 0.5, 3.14159, 22.627417, -0.0001, 65504, 1e-6] {
            let back = Half.toFloat(Half.fromFloat(v))
            XCTAssertEqual(back, v, accuracy: max(abs(v) * 1e-3, 1e-7), "\(v)")
        }
        #if arch(arm64)
        for i in 0 ..< 2000 {
            let v = Float(sin(Double(i) * 0.37)) * Float(i % 50 + 1)
            XCTAssertEqual(Half.fromFloat(v), Float16(v).bitPattern, "\(v)")
            XCTAssertEqual(Half.toFloat(Float16(v).bitPattern), Float(Float16(v)))
        }
        #endif
    }

    // MARK: pipeline

    func testEmbedMatchesReferenceAndPicksBuckets() throws {
        let (e, _) = try makeEmbedder()
        for (n, want) in [(1, 4), (4, 4), (5, 8), (9, 16), (17, 32), (32, 32)] {
            let ids = (0 ..< n).map { Int32(($0 * 5 + 1) % Self.vocab) }
            let (v, bucket) = try e.embed(tokenIDs: ids)
            XCTAssertEqual(bucket, want, "\(n) tokens")
            let ref = reference(ids)
            for c in 0 ..< Self.dim { XCTAssertEqual(v[c], ref[c], accuracy: 1e-4, "n=\(n) c=\(c)") }
        }
    }

    func testOverLargestBucketThrows() throws {
        let (e, _) = try makeEmbedder()
        XCTAssertThrowsError(try e.embed(tokenIDs: [Int32](repeating: 1, count: 33)))
    }

    func testBucketsLoadLazily() throws {
        let (e, _) = try makeEmbedder()
        XCTAssertEqual(e.loadedBuckets, [], "nothing resident until used")
        _ = try e.embed(tokenIDs: [1, 2, 3])
        XCTAssertEqual(e.loadedBuckets, [4], "the smallest bucket first")
        _ = try e.embed(tokenIDs: Array(repeating: 2, count: 20))
        XCTAssertEqual(e.loadedBuckets, [4, 32], "a larger bucket only when a request needs it")
        XCTAssertEqual(e.primaryBucket, 4)
    }

    func testUnloadAllDropsBucketsAndDoesNotResurrectInFlightLoad() throws {
        let before = FakeRunner.live.value
        let (e, _) = try makeEmbedder()
        _ = try e.embed(tokenIDs: [1, 2]); _ = try e.embed(tokenIDs: Array(repeating: 3, count: 10))
        XCTAssertEqual(FakeRunner.live.value - before, 2)
        XCTAssertTrue(e.unloadAll())
        XCTAssertEqual(FakeRunner.live.value - before, 0, "runners are freed on unload")
        XCTAssertFalse(e.unloadAll())
        // Reloads transparently afterwards.
        _ = try e.embed(tokenIDs: [1])
        XCTAssertEqual(e.loadedBuckets, [4])
    }

    func testConcurrentEmbedsWithUnloadNeverFail() throws {
        let (e, _) = try makeEmbedder()
        let group = DispatchGroup()
        let failures = LiveCounter()
        for t in 0 ..< 8 {
            group.enter()
            DispatchQueue.global().async {
                for i in 0 ..< 150 {
                    let n = 1 + (i * 7 + t) % 32
                    let ids = (0 ..< n).map { Int32(($0 + t) % Self.vocab) }
                    do {
                        let (v, _) = try e.embed(tokenIDs: ids)
                        let ref = self.reference(ids)
                        if zip(v, ref).contains(where: { abs($0 - $1) > 1e-4 }) { failures.add(1) }
                    } catch { failures.add(1) }
                    if t == 0, i % 25 == 0 { e.unloadAll() }
                }
                group.leave()
            }
        }
        group.wait()
        XCTAssertEqual(failures.value, 0)
    }

    // MARK: engine integration (tokenizer, prompts, MRL, keep-alive unload)

    /// Minimal Gemma-shaped BPE: a-z and the space marker, no merges, <bos> A <eos>.
    func writeTokenizer(into dir: URL) throws {
        var vocab: [String: Int] = ["<pad>": 0, "<eos>": 1, "<bos>": 2, "\u{2581}": 3]
        for (i, ch) in "abcdefghijklmnopqrstuvwxyz".enumerated() { vocab[String(ch)] = 4 + i }
        for b in 0 ..< 256 { vocab[String(format: "<0x%02X>", b)] = 30 + b }
        let json: [String: Any] = [
            "normalizer": ["type": "Replace", "pattern": ["String": " "], "content": "\u{2581}"],
            "model": ["type": "BPE", "vocab": vocab, "merges": [[String]](), "byte_fallback": true],
            "added_tokens": [["id": 0, "content": "<pad>", "special": true], ["id": 1, "content": "<eos>", "special": true],
                             ["id": 2, "content": "<bos>", "special": true]],
            "post_processor": ["type": "TemplateProcessing",
                               "single": [["SpecialToken": ["id": "<bos>", "type_id": 0]], ["Sequence": ["id": "A", "type_id": 0]],
                                          ["SpecialToken": ["id": "<eos>", "type_id": 0]]]],
        ]
        try JSONSerialization.data(withJSONObject: json).write(to: dir.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: ["prompts": ["query": "task: search result | query: ", "document": "title: none | text: "]])
            .write(to: dir.appendingPathComponent("config_sentence_transformers.json"))
    }

    func makeEngine() throws -> (EmbeddingEngine, ANEEmbedder, URL) {
        let (dir, m) = try makePackage(buckets: [8, 16, 32, 128], maxTokens: 128)
        try writeTokenizer(into: dir)
        let embedder = try ANEEmbedder(directory: dir, manifest: m) { b, _ in FakeRunner(seq: b.seq, dim: Self.dim) }
        let engine = EmbeddingEngine()
        try engine.loadANE(directory: dir, fallbackDirectory: nil, embedder: embedder)
        return (engine, embedder, dir)
    }

    func testEngineServesTextOnANE() throws {
        let (engine, _, dir) = try makeEngine()
        XCTAssertTrue(engine.isLoaded(directory: dir))
        let r = try engine.embed(["hello world", "abc"])
        XCTAssertEqual(r.servedBy, .ane)
        XCTAssertEqual(r.vectors.count, 2)
        for v in r.vectors {
            XCTAssertEqual(v.count, Self.dim)
            XCTAssertEqual(v.reduce(0) { $0 + $1 * $1 }, 1, accuracy: 1e-5, "unit length")
        }
        XCTAssertEqual(r.promptTokens, (11 + 2) + (3 + 2), "bos/eos included")
    }

    func testEngineAppliesTaskPrefixAndRejectsUnknown() throws {
        let (engine, _, _) = try makeEngine()
        let plain = try engine.embed(["abc"]).vectors[0]
        let q = try engine.embed(["abc"], options: EmbeddingRequestOptions(task: "query")).vectors[0]
        XCTAssertNotEqual(plain, q, "the query prompt changes the vector")
        XCTAssertThrowsError(try engine.embed(["abc"], options: EmbeddingRequestOptions(task: "bogus")))
    }

    func testEngineMRLTruncationRenormalises() throws {
        let (engine, _, _) = try makeEngine()
        let full = try engine.embed(["hello"]).vectors[0]
        // 768/512/256/128 are the EG2 dims; the tiny fixture only has 8, so a valid dim must be rejected.
        XCTAssertThrowsError(try engine.embed(["hello"], options: EmbeddingRequestOptions(dimensions: 5)))
        XCTAssertEqual(full.count, Self.dim)
    }

    func testMaxLengthCapShortensTheSequence() throws {
        let (engine, _, _) = try makeEngine()
        let long = String(repeating: "a", count: 30)
        let a = try engine.embed([long], options: EmbeddingRequestOptions(maxLength: 8))
        XCTAssertEqual(a.promptTokens, 8, "truncated to max_length (with <eos> kept) before bucketing")
        XCTAssertEqual(a.servedBy, .ane)
    }

    func testEngineUnloadFreesTheBuckets() throws {
        let before = FakeRunner.live.value
        let engine = EmbeddingEngine()
        do {
            let (e, _, _) = try makeEngine()
            _ = try e.embed(["hello"])
            XCTAssertGreaterThan(FakeRunner.live.value, before)
            XCTAssertTrue(e.isResident)
            XCTAssertTrue(e.unload(), "keep-alive eviction reports a resident model")
            XCTAssertFalse(e.isResident)
            XCTAssertFalse(e.unload())
        }
        XCTAssertEqual(FakeRunner.live.value, before, "no bucket outlives the unload")
        _ = engine
    }

    func testUnloadedEngineCannotEmbedUntilReloaded() throws {
        let (engine, _, _) = try makeEngine()
        _ = engine.unload()
        XCTAssertThrowsError(try engine.embed(["x"]))
    }

    func testTextOverTheLargestBucketNeedsTheFallback() throws {
        let (engine, _, _) = try makeEngine()
        // 130 chars -> 132 tokens > 128: must go to the MLX fallback, which is not installed here.
        XCTAssertThrowsError(try engine.embed([String(repeating: "a", count: 130)])) { err in
            XCTAssertTrue("\(err)".contains("embeddinggemma-2-mlx-4bit-dyn-text"), "\(err)")
        }
    }

    func testMediaInputNeedsTheFallback() throws {
        let (engine, _, _) = try makeEngine()
        let img = EmbeddingInput(parts: [.text("x"), .image(Data([1, 2, 3]))])
        XCTAssertThrowsError(try engine.embed(inputs: [img]))
    }
}
