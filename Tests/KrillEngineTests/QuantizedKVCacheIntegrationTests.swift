import XCTest
@testable import KrillEngine

/// End-to-end check that the int8 KV cache path produces output close to the fp16 path.
///
/// To enable int8 in the multimodal benchmark scenario set:
///   `KRILL_KV_CACHE_DTYPE=int8`
/// (no engine recompile required; the engine reads it at construction time).
final class QuantizedKVCacheIntegrationTests: XCTestCase {

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

    private func generateGreedy(
        modelDir: URL,
        kvCacheDtype: String,
        prompt: String,
        maxTokens: Int
    ) async throws -> [Int] {
        let engine = InferenceEngine(modelDirectory: modelDir, kvCacheDtype: kvCacheDtype)
        try await engine.load()

        let (stream, _) = engine.generate(
            prompt: prompt,
            params: .greedy,
            maxTokens: maxTokens,
            usePrefixCache: false
        )

        var tokens: [Int] = []
        for await event in stream {
            if event.isEnd { break }
            tokens.append(event.tokenId)
            if tokens.count >= maxTokens { break }
        }
        return tokens
    }

    func testInt8AndFp16ProduceSimilarGreedyPrefix() async throws {
        let modelDir = try requireModelDirectory()
        let prompt = "List three primary colors."
        let maxTokens = 16

        let fp16Tokens = try await generateGreedy(
            modelDir: modelDir, kvCacheDtype: "fp16",
            prompt: prompt, maxTokens: maxTokens)
        let int8Tokens = try await generateGreedy(
            modelDir: modelDir, kvCacheDtype: "int8",
            prompt: prompt, maxTokens: maxTokens)

        XCTAssertFalse(fp16Tokens.isEmpty, "fp16 path produced no tokens")
        XCTAssertFalse(int8Tokens.isEmpty, "int8 path produced no tokens")

        // First 8 greedy tokens must match. If int8 quantization corrupts the
        // logits enough that the argmax diverges this early, the int8 path is
        // broken and the test fails.
        let compareLen = min(8, fp16Tokens.count, int8Tokens.count)
        let fp16Prefix = Array(fp16Tokens.prefix(compareLen))
        let int8Prefix = Array(int8Tokens.prefix(compareLen))
        XCTAssertEqual(int8Prefix, fp16Prefix,
            "int8 KV greedy decode diverged from fp16 within first \(compareLen) tokens")
    }
}
