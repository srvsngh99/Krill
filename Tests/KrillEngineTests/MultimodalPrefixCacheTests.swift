import XCTest
@testable import KrillCache
import KrillCore
import KrillRuntime

#if canImport(MLX)
import MLX
#endif

/// Verifies that the prefix cache key incorporates the media (image/audio)
/// content hash so that two requests with identical text tokens but
/// different image bytes do NOT mis-hit each other's cached KV state.
///
/// This guards against a multimodal correctness bug where the previous key
/// schema (tokens + modelId only) would serve a cached prefix conditioned on
/// image A in response to a request that supplies image B.
final class MultimodalPrefixCacheTests: XCTestCase {

    private func makeTempCache() -> PrefixCache {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-mm-prefix-test-\(UUID().uuidString)")
        addTeardownBlock {
            try? FileManager.default.removeItem(at: dir)
        }
        return PrefixCache(cacheDir: dir, maxMemoryEntries: 4, minPrefixLength: 4)
    }

    func testTextOnlyHitWithNilMediaHash() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()
            let tokens = Array(0 ..< 8)
            let kv = makeKV(seqLen: tokens.count, start: 1)

            cache.store(
                tokens: tokens, modelId: "m",
                keys: [[kv]], values: [[kv]],
                mediaHash: nil
            )

            let hit = cache.lookup(tokens: tokens, modelId: "m", mediaHash: nil)
            XCTAssertNotNil(hit, "Same key (no media) should hit")
            XCTAssertEqual(hit?.prefixLength, tokens.count)
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testDifferentImageHashMisses() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()
            let tokens = Array(0 ..< 8)
            let kv = makeKV(seqLen: tokens.count, start: 1)

            let hashX = "imghash_X"
            let hashY = "imghash_Y"

            cache.store(
                tokens: tokens, modelId: "m",
                keys: [[kv]], values: [[kv]],
                mediaHash: hashX
            )

            // Same text, same model, different image -> MUST miss.
            XCTAssertNil(
                cache.lookup(tokens: tokens, modelId: "m", mediaHash: hashY),
                "Different image hash must not hit a prior entry"
            )

            // Same text, same model, no image at all -> MUST miss
            // (text-only request must not consume an entry conditioned on an image).
            XCTAssertNil(
                cache.lookup(tokens: tokens, modelId: "m", mediaHash: nil),
                "Text-only lookup must not hit an image-conditioned entry"
            )

            // Same text+image -> MUST hit.
            XCTAssertNotNil(
                cache.lookup(tokens: tokens, modelId: "m", mediaHash: hashX),
                "Same image hash should hit"
            )
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testRealImageBytesHashKeyDistinguishesImages() {
        let imgA = Data((0 ..< 256).map { UInt8($0) })
        let imgB = Data((0 ..< 256).map { UInt8($0 ^ 0xAA) })
        let kA = VisionEncoderCache.key(forImageBytes: imgA)
        let kB = VisionEncoderCache.key(forImageBytes: imgB)
        XCTAssertNotEqual(kA, kB, "SHA-256 keys must differ for different image bytes")
        XCTAssertEqual(kA, VisionEncoderCache.key(forImageBytes: imgA), "Stable for same bytes")
    }

    // MARK: - Helpers

    #if canImport(MLX) && os(macOS) && arch(arm64)
    private func withMLXCPU(_ body: () throws -> Void) throws {
        guard MLXMetalRuntime.canInitializeMLXForTests else {
            throw XCTSkip("MLX Metal runtime is unavailable to this test process.")
        }
        try Device.withDefaultDevice(.cpu) {
            try body()
        }
    }

    private func makeKV(
        batch: Int = 1,
        heads: Int = 2,
        seqLen: Int,
        headDim: Int = 2,
        start: Int32
    ) -> MLXArray {
        let count = batch * heads * seqLen * headDim
        let values = (0 ..< count).map { start + Int32($0) }
        return MLXArray(values, [batch, heads, seqLen, headDim])
    }
    #endif
}
