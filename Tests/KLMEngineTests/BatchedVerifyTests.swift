import XCTest
@testable import KLMEngine

/// Foundational correctness gate for batched speculative decode: a single
/// `[R, L]` batched verify forward (per-row block-causal mask + per-row RoPE)
/// must reproduce, for every row and every position, the logits of L sequential
/// single-token steps. If this holds, a batched speculative verify can commit
/// multiple tokens per row in one forward — the prerequisite for batched n-gram
/// spec (filling the R=4-8 GPU-occupancy gap, see docs/CONCURRENT_THROUGHPUT.md).
///
/// Gated on `KLM_BATCH_MODEL_PATH` (a plain-causal checkpoint), matching the
/// live-test convention in `BatchedDecodeLiveTests`.
final class BatchedVerifyTests: XCTestCase {

    private func requireModelDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KLM_BATCH_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KLM_BATCH_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("KLM_BATCH_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    func testBatchedVerifyMatchesSequentialPerRow() async throws {
        let dir = try requireModelDirectory()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        guard engine.supportsBatchedDecode else {
            throw XCTSkip("loaded model is not batched-eligible (not plain-causal)")
        }

        // Ragged-length prompts so per-row left-pad + per-row RoPE offsets are all
        // exercised (the same cases the single-token batched gate covers).
        let texts = [
            "The capital of France is",
            "Write one short sentence about the ocean and why it matters to the planet.",
            "List three colors:",
        ]
        let prompts = try texts.map { text -> [Int] in
            guard let toks = engine.encodeForBatchTest(text), !toks.isEmpty else {
                throw XCTSkip("tokenizer produced no tokens")
            }
            return toks
        }

        // L = 8 new tokens verified in ONE [R, 8] forward vs 8 sequential steps.
        let diffs = try XCTUnwrap(
            engine.batchedVerifyVsSerialMaxDiff(promptTokens: prompts, steps: 8))
        XCTAssertEqual(diffs.count, prompts.count)
        for (r, d) in diffs.enumerated() {
            XCTAssertLessThan(d, 0.05,
                "row \(r): [R,L] batched verify logits must match L sequential steps "
                + "(max abs diff \(d)) — block-causal mask + per-row RoPE correct")
        }
    }
}
