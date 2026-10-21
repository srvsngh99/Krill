import XCTest
import Foundation
@testable import KrillEngine

/// Manifest format 2 (per-model packages) and its backward-compatible reading of format 1.
final class ANEGenericPackageTests: XCTestCase {
    private func fixture(_ name: String, file: StaticString = #filePath) -> URL {
        URL(fileURLWithPath: "\(file)").deletingLastPathComponent().appendingPathComponent("Fixtures/\(name)")
    }

    func testFormat1PackageStillReads() throws {
        let m = try ANEPackageManifest.load(directory: fixture("ane_tiny"))
        XCTAssertEqual(m.format, 1)
        XCTAssertEqual(m.kind, "embedder")
        XCTAssertEqual(m.tokenizer, "codepoint_bpe", "format 1 is EmbeddingGemma 2")
        XCTAssertEqual(m.outputName, "emb")
        XCTAssertEqual(m.gather, "host")
        XCTAssertEqual(m.contextTokens, m.maxTokens)
        XCTAssertEqual(m.effectiveMRLDims, [768, 512, 256, 128])
        XCTAssertFalse(m.appendEOS)
        XCTAssertNil(m.model)
    }

    func testFormat2EmbedderParses() throws {
        let m = try ANEPackageManifest.load(directory: fixture("ane_tiny_bert"))
        XCTAssertEqual(m.format, 2)
        XCTAssertEqual(m.kind, "embedder")
        XCTAssertEqual(m.pooling, "mean")
        XCTAssertEqual(m.tokenizer, "krill")
        XCTAssertEqual(m.outputName, "emb")
        XCTAssertEqual(m.contextTokens, 64)
        XCTAssertEqual(m.maxTokens, 16)
        XCTAssertEqual(m.embedScale, 1)
        XCTAssertNil(m.effectiveMRLDims, "only EmbeddingGemma 2 is a matryoshka model here")
        XCTAssertEqual(m.table.dim, 32)
    }

    func testFormat2RerankerParses() throws {
        let m = try ANEPackageManifest.load(directory: fixture("ane_tiny_rerank"))
        XCTAssertEqual(m.kind, "reranker")
        XCTAssertEqual(m.pooling, "classifier")
        XCTAssertEqual(m.outputName, "score")
        XCTAssertEqual(m.outputDim, 1)
    }

    func testFormat2RoundTrips() throws {
        let m = ANEPackageManifest(format: 2, variant: "lin8", outputDim: 1536, embedScale: 1, maxTokens: 512,
                                   buckets: [.init(seq: 512, path: "s512.mlmodelc")],
                                   table: .init(file: "t", vocab: 10, dim: 1536), fallbackModel: "gte-qwen2-1.5b",
                                   kind: "embedder", model: "gte-qwen2-1.5b", pooling: "last_token",
                                   tokenizer: "krill", contextTokens: 8192, appendEOS: true, mrlDims: [1536, 768])
        let back = try JSONDecoder().decode(ANEPackageManifest.self, from: try JSONEncoder().encode(m))
        XCTAssertEqual(back, m)
        try back.validate()
        XCTAssertTrue(back.appendEOS)
        XCTAssertEqual(back.effectiveMRLDims, [1536, 768])
    }

    func testEveryPoolingVariantValidatesAndUnknownOnesDoNot() throws {
        func make(_ pooling: String, kind: String = "embedder", gather: String = "host", tokenizer: String = "krill",
                  format: Int = 2) -> ANEPackageManifest {
            ANEPackageManifest(format: format, variant: "fp16", outputDim: 8, embedScale: 1, maxTokens: 16,
                               buckets: [.init(seq: 16, path: "x")], table: .init(file: "t", vocab: 4, dim: 8),
                               fallbackModel: "m", kind: kind, pooling: pooling, gather: gather, tokenizer: tokenizer)
        }
        for p in ["mean", "cls", "last_token", "classifier"] { XCTAssertNoThrow(try make(p).validate(), p) }
        XCTAssertThrowsError(try make("max").validate())
        XCTAssertThrowsError(try make("mean", kind: "classifier").validate())
        XCTAssertThrowsError(try make("mean", gather: "model").validate(), "in-model gather is reserved, not implemented")
        XCTAssertThrowsError(try make("mean", tokenizer: "sentencepiece").validate())
        XCTAssertThrowsError(try make("mean", format: 3).validate(), "a newer format is refused, not guessed at")
    }

    func testTokenShapingForEncodersAndDecoders() {
        // Encoders (mean / CLS pooling): cut at the cap, no EOS games; empty input becomes one BOS.
        XCTAssertEqual(EmbeddingEngine.shapeIDs([1, 5, 6, 7, 2], eos: 2, bos: 1, cap: 4, appendEOS: false), [1, 5, 6, 7])
        XCTAssertEqual(EmbeddingEngine.shapeIDs([1, 5, 2], eos: 2, bos: 1, cap: 8, appendEOS: false), [1, 5, 2])
        XCTAssertEqual(EmbeddingEngine.shapeIDs([], eos: 2, bos: 1, cap: 8, appendEOS: false), [1])
        // Decoders (last-token pooling): exactly one trailing EOS, inside the cap.
        XCTAssertEqual(EmbeddingEngine.shapeIDs([5, 6], eos: 9, bos: 1, cap: 8, appendEOS: true), [5, 6, 9])
        XCTAssertEqual(EmbeddingEngine.shapeIDs([5, 6, 9], eos: 9, bos: 1, cap: 8, appendEOS: true), [5, 6, 9], "already ends in EOS")
        XCTAssertEqual(EmbeddingEngine.shapeIDs([5, 6, 7, 8, 9], eos: 9, bos: 1, cap: 4, appendEOS: true), [5, 6, 7, 9])
        XCTAssertEqual(EmbeddingEngine.shapeIDs([], eos: 9, bos: 1, cap: 4, appendEOS: true), [1, 9])
    }
}
