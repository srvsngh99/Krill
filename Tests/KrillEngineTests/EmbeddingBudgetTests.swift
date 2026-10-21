import XCTest
import Foundation
@testable import KrillEngine

/// Memory-bounding knobs of the embedding/rerank path: padded-token batch
/// budget, per-request / server-wide max input length, idle unload. No model
/// weights needed.
final class EmbeddingBudgetTests: XCTestCase {

    // MARK: token budget

    func testPlanChunksRespectsBudgetAndCoversEverythingOnce() {
        let lengths = [5, 8, 8, 20, 64, 64, 64, 512, 512, 900]
        for budget in [64, 256, 2048, 8192] {
            let chunks = EmbeddingEngine.planChunks(sortedLengths: lengths, budget: budget, maxBatch: 32)
            XCTAssertEqual(chunks.flatMap { Array($0) }, Array(lengths.indices))
            for c in chunks where c.count > 1 {
                XCTAssertLessThanOrEqual(c.count * lengths[c.upperBound - 1], budget,
                                         "chunk \(c) exceeds budget \(budget)")
            }
        }
    }

    func testPlanChunksOversizedItemRunsAloneAndMaxBatchHolds() {
        let chunks = EmbeddingEngine.planChunks(sortedLengths: [10, 10, 10, 10, 5000],
                                                budget: 100, maxBatch: 3)
        XCTAssertEqual(chunks, [0 ..< 3, 3 ..< 4, 4 ..< 5])
        XCTAssertTrue(EmbeddingEngine.planChunks(sortedLengths: [], budget: 10, maxBatch: 4).isEmpty)
    }

    func testBudgetEnvParsing() {
        XCTAssertEqual(EmbeddingEngine.paddedTokenBudget(environment: [:]),
                       EmbeddingEngine.defaultPaddedTokenBudget)
        XCTAssertEqual(EmbeddingEngine.paddedTokenBudget(
            environment: ["KRILL_EMBED_MAX_BATCH_TOKENS": " 4096 "]), 4096)
        for bad in ["0", "-5", "abc", ""] {
            XCTAssertEqual(EmbeddingEngine.paddedTokenBudget(
                environment: ["KRILL_EMBED_MAX_BATCH_TOKENS": bad]),
                EmbeddingEngine.defaultPaddedTokenBudget, bad)
        }
    }

    func testModelDefaultBudgetFollowsPrecisionAndEnvStillWins() {
        XCTAssertEqual(EmbeddingEngine.modelDefaultBudget(quantized: true), 512)
        XCTAssertEqual(EmbeddingEngine.modelDefaultBudget(quantized: false), 2048)
        let full = EmbeddingEngine.modelDefaultBudget(quantized: false)
        XCTAssertEqual(EmbeddingEngine.paddedTokenBudget(modelDefault: full, environment: [:]), 2048)
        XCTAssertEqual(EmbeddingEngine.paddedTokenBudget(
            modelDefault: full, environment: ["KRILL_EMBED_MAX_BATCH_TOKENS": "512"]), 512)
        XCTAssertEqual(EmbeddingEngine.paddedTokenBudget(
            modelDefault: full, environment: ["KRILL_EMBED_MAX_BATCH_TOKENS": "junk"]), 2048)
    }

    // MARK: max input length

    func testDefaultCapIsModelCapUnchanged() {
        XCTAssertEqual(EmbeddingRequestOptions.effectiveTokenCap(
            modelCap: 8192, requestCap: nil, environment: [:]), 8192)
    }

    func testCapTakesTheSmallestOfModelServerRequest() {
        let env = ["KRILL_EMBED_MAX_TOKENS": "1024"]
        XCTAssertEqual(EmbeddingRequestOptions.effectiveTokenCap(
            modelCap: 8192, requestCap: nil, environment: env), 1024)
        XCTAssertEqual(EmbeddingRequestOptions.effectiveTokenCap(
            modelCap: 8192, requestCap: 512, environment: env), 512)
        // a request can only lower the cap, never raise it
        XCTAssertEqual(EmbeddingRequestOptions.effectiveTokenCap(
            modelCap: 8192, requestCap: 4000, environment: env), 1024)
        XCTAssertEqual(EmbeddingRequestOptions.effectiveTokenCap(
            modelCap: 512, requestCap: 9000, environment: [:]), 512)
        XCTAssertEqual(EmbeddingRequestOptions.effectiveTokenCap(
            modelCap: 512, requestCap: 1, environment: [:]), 2)
        XCTAssertEqual(EmbeddingRequestOptions.effectiveTokenCap(
            modelCap: 512, requestCap: nil, environment: ["KRILL_EMBED_MAX_TOKENS": "junk"]), 512)
    }

    func testParsesMaxLengthAndOllamaNumCtx() throws {
        XCTAssertEqual(try EmbeddingRequestOptions.parse(["max_length": 256]).maxLength, 256)
        XCTAssertEqual(try EmbeddingRequestOptions.parse(["options": ["num_ctx": 300]]).maxLength, 300)
        // explicit max_length wins over options.num_ctx
        XCTAssertEqual(try EmbeddingRequestOptions.parse(
            ["max_length": 128, "options": ["num_ctx": 300]]).maxLength, 128)
        XCTAssertNil(try EmbeddingRequestOptions.parse(["max_length": NSNull()]).maxLength)
        for bad: Any in [0, -1, 2.5, "64", true] {
            XCTAssertThrowsError(try EmbeddingRequestOptions.parse(["max_length": bad]), "\(bad)")
        }
        XCTAssertThrowsError(try EmbeddingRequestOptions.parse(["options": ["num_ctx": 0]]))
    }

    // MARK: unload

    func testUnloadOnEmptyEnginesIsNoOp() {
        let e = EmbeddingEngine()
        XCTAssertFalse(e.isResident)
        XCTAssertFalse(e.unload())
        let r = RerankEngine()
        XCTAssertFalse(r.unload())
        XCTAssertNil(e.loadedModelName)
    }
}
