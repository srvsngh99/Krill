import XCTest
import MLX
@testable import KLMCore

final class VisionEncoderCacheTests: XCTestCase {

    func testKeyIsStableForSameBytes() {
        let bytes = Data((0..<128).map { UInt8($0) })
        let k1 = VisionEncoderCache.key(forImageBytes: bytes)
        let k2 = VisionEncoderCache.key(forImageBytes: bytes)
        XCTAssertEqual(k1, k2)
        XCTAssertEqual(k1.count, 64)  // SHA-256 hex
    }

    func testKeyDiffersForDifferentBytes() {
        let a = Data((0..<128).map { UInt8($0) })
        let b = Data((0..<128).map { UInt8($0 ^ 0x55) })
        XCTAssertNotEqual(
            VisionEncoderCache.key(forImageBytes: a),
            VisionEncoderCache.key(forImageBytes: b))
    }

    func testStoreThenLookupReturnsValueAndCountsHit() {
        let cache = VisionEncoderCache(capacity: 4)
        let key = "abc"
        let value = MLXArray([Float(1), 2, 3, 4])
        cache.store(key, value: value)

        XCTAssertEqual(cache.count, 1)
        XCTAssertEqual(cache.hits, 0)
        XCTAssertEqual(cache.misses, 0)

        guard let hit = cache.lookup(key) else {
            XCTFail("expected cache hit")
            return
        }
        XCTAssertEqual(cache.hits, 1)
        XCTAssertEqual(cache.misses, 0)

        // Bit-identical check via value comparison.
        let same = MLX.allClose(hit, value, rtol: 0, atol: 0).item(Bool.self)
        XCTAssertTrue(same)
    }

    func testMissCountsAsMiss() {
        let cache = VisionEncoderCache(capacity: 4)
        XCTAssertNil(cache.lookup("does-not-exist"))
        XCTAssertEqual(cache.misses, 1)
        XCTAssertEqual(cache.hits, 0)
    }

    func testLRUEvictionAtCapacity() {
        let cache = VisionEncoderCache(capacity: 2)
        cache.store("a", value: MLXArray([Float(1)]))
        cache.store("b", value: MLXArray([Float(2)]))
        cache.store("c", value: MLXArray([Float(3)]))  // evicts "a"

        XCTAssertEqual(cache.count, 2)
        XCTAssertNil(cache.lookup("a"))
        XCTAssertNotNil(cache.lookup("b"))
        XCTAssertNotNil(cache.lookup("c"))
    }

    func testLRUTouchOnLookup() {
        let cache = VisionEncoderCache(capacity: 2)
        cache.store("a", value: MLXArray([Float(1)]))
        cache.store("b", value: MLXArray([Float(2)]))
        // Touch "a" so "b" becomes the LRU entry.
        _ = cache.lookup("a")
        cache.store("c", value: MLXArray([Float(3)]))  // should evict "b", not "a"

        XCTAssertNotNil(cache.lookup("a"))
        XCTAssertNil(cache.lookup("b"))
        XCTAssertNotNil(cache.lookup("c"))
    }

    func testTwoDifferentImagesPopulateCache() {
        let cache = VisionEncoderCache(capacity: 4)
        let bytesA = Data((0..<64).map { UInt8($0) })
        let bytesB = Data((0..<64).map { UInt8(0xFF - $0) })
        let keyA = VisionEncoderCache.key(forImageBytes: bytesA)
        let keyB = VisionEncoderCache.key(forImageBytes: bytesB)
        XCTAssertNotEqual(keyA, keyB)

        let valA = MLXArray([Float(1), 2, 3])
        let valB = MLXArray([Float(4), 5, 6])
        cache.store(keyA, value: valA)
        cache.store(keyB, value: valB)
        XCTAssertEqual(cache.count, 2)

        guard let hitA = cache.lookup(keyA), let hitB = cache.lookup(keyB) else {
            XCTFail("both keys should hit")
            return
        }
        XCTAssertTrue(MLX.allClose(hitA, valA, rtol: 0, atol: 0).item(Bool.self))
        XCTAssertTrue(MLX.allClose(hitB, valB, rtol: 0, atol: 0).item(Bool.self))
        XCTAssertFalse(MLX.allClose(hitA, hitB, rtol: 0, atol: 0).item(Bool.self))
    }

    // Integration test: run the actual SigLIP2 encoder through Gemma4MultimodalModel
    // twice with the same hash, verify the cache reports a hit and the output
    // matches bit-for-bit. Skipped when no Gemma 4 model is available (same gate
    // as Gemma4SmokeTests).
    func testEncoderCacheHitProducesIdenticalOutput() throws {
        guard let path = ProcessInfo.processInfo.environment["KLM_GEMMA4_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KLM_GEMMA4_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("KLM_GEMMA4_MODEL_PATH is not a directory: \(path)")
        }

        let dir = URL(fileURLWithPath: path, isDirectory: true)
        let loaded = try loadModel(from: dir)
        guard let mmForward = loaded.multimodalForward,
              let mmModel = loaded.module as? Gemma4MultimodalModel else {
            throw XCTSkip("Loaded model is not multimodal Gemma 4")
        }

        // Solid-red 48x48 PNG-ish raw bytes — preprocessImage decodes via CGImageSource.
        let assets = ProcessInfo.processInfo.environment["KLM_BENCH_ASSETS_DIR"]
            .map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".build")
                .appendingPathComponent("benchmarks")
                .appendingPathComponent("assets")
        let imageURL = assets.appendingPathComponent("gemma4-red-box.png")
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            throw XCTSkip("red-box asset missing at \(imageURL.path)")
        }
        let imageData = try Data(contentsOf: imageURL)

        let pixels = try preprocessImage(imageData)
        let hash = VisionEncoderCache.key(forImageBytes: imageData)
        let dummyTokens = MLXArray([Int32(0)]).reshaped(1, 1)

        mmModel.visionCache.resetCounters()
        let logitsA = mmForward(dummyTokens, nil, pixels, nil, hash)
        MLX.eval(logitsA)
        XCTAssertEqual(mmModel.visionCache.misses, 1)
        XCTAssertEqual(mmModel.visionCache.hits, 0)

        let logitsB = mmForward(dummyTokens, nil, pixels, nil, hash)
        MLX.eval(logitsB)
        XCTAssertEqual(mmModel.visionCache.hits, 1, "second call should be a cache hit")

        // The cache stores image embeddings; the LM forward is re-run, but its
        // output for identical inputs (no caches passed) must be identical.
        XCTAssertTrue(
            MLX.allClose(logitsA, logitsB, rtol: 0, atol: 0).item(Bool.self),
            "cached encoder output should yield bit-identical logits")

        // Two different hashes should both populate the cache.
        let altHash = "deadbeef" + String(repeating: "0", count: 56)
        _ = mmForward(dummyTokens, nil, pixels, nil, altHash)
        XCTAssertEqual(mmModel.visionCache.count, 2)
    }
}
