import XCTest
import KrillCache
@testable import KrillEngine

/// End-to-end check that int8 KV cache and the prefix cache compose.
///
/// Runs the same prompt twice through one `InferenceEngine` configured with
/// `kvCacheDtype: "int8"` and a shared in-process `PrefixCache`. The first
/// run is a cold miss; the second must restore the int8 snapshot, truncate,
/// and re-forward only the last token while producing the same greedy tokens
/// as the first run.
///
/// Skipped unless `KRILL_GEMMA4_MODEL_PATH` points at a real Gemma 4 checkpoint,
/// matching the existing live-test gating convention.
final class QuantizedPrefixCacheLiveTests: XCTestCase {

    private func requireModelDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KRILL_GEMMA4_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KRILL_GEMMA4_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("KRILL_GEMMA4_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private func makeTempCache() -> PrefixCache {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-q8-prefix-live-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return PrefixCache(cacheDir: dir, maxMemoryEntries: 4, minPrefixLength: 4)
    }

    private func generateTokens(
        engine: InferenceEngine,
        prompt: String,
        maxTokens: Int
    ) async -> [Int] {
        let (stream, _) = engine.generate(
            prompt: prompt,
            params: .greedy,
            maxTokens: maxTokens,
            usePrefixCache: true
        )
        var tokens: [Int] = []
        for await event in stream {
            if event.isEnd { break }
            tokens.append(event.tokenId)
            if tokens.count >= maxTokens { break }
        }
        return tokens
    }

    func testInt8PrefixCacheReplayMatchesColdRun() async throws {
        let modelDir = try requireModelDirectory()
        let cache = makeTempCache()
        let prompt = "List three primary colors."
        let maxTokens = 12

        let engine = InferenceEngine(
            modelDirectory: modelDir,
            prefixCache: cache,
            kvCacheDtype: "int8"
        )
        try await engine.load()

        // Cold run: miss → prefill → store quantized snapshot.
        XCTAssertEqual(cache.memoryCount, 0, "cache should start empty")
        let cold = await generateTokens(engine: engine, prompt: prompt, maxTokens: maxTokens)
        XCTAssertFalse(cold.isEmpty, "cold int8 run produced no tokens")

        // Give the write-behind Task a beat to land the entry in memory.
        // The store path is fire-and-forget, so we busy-wait briefly.
        for _ in 0 ..< 50 {
            if cache.memoryCount > 0 { break }
            try await Task.sleep(nanoseconds: 20_000_000) // 20 ms
        }
        XCTAssertGreaterThan(cache.memoryCount, 0,
            "quantized prefill should have populated the prefix cache")

        // Warm run: must hit the quantized entry. Tokens must match the cold
        // run — restore + truncate + re-forward the trailing prompt token is
        // expected to be deterministic under greedy sampling.
        let warm = await generateTokens(engine: engine, prompt: prompt, maxTokens: maxTokens)
        let compareLen = min(maxTokens, cold.count, warm.count)
        XCTAssertEqual(Array(warm.prefix(compareLen)), Array(cold.prefix(compareLen)),
            "int8 prefix-cache replay diverged from the cold int8 run")
    }
}
