import XCTest
@testable import KrillRegistry

/// The published Neural Engine packages (docs/ANE_EMBEDDER.md, "Supported models").
final class ANEPackageAliasTests: XCTestCase {
    private let published: [(model: String, variant: String, family: ModelFamily)] = [
        ("all-minilm-l6-v2", "fp16", .bert),
        ("bge-small-en", "fp16", .bert),
        ("bge-base-en", "fp16", .bert),
        ("all-mpnet-base-v2", "fp16", .bert),
        ("gte-base-en-v1.5", "fp16", .bert),
        ("jina-embeddings-v2-base-en", "fp16", .bert),
        ("nomic-embed-text", "fp16", .bert),
        ("nomic-embed-text-v1", "fp16", .bert),
        ("nomic-embed-text-v2-moe", "fp16", .bert),
        ("bge-reranker-base", "lin8", .reranker),
        ("embeddinggemma-2", "lin8", .bert),
    ]

    func testEveryPublishedPackageResolvesToItsOwnSrvSnghRepo() throws {
        for p in published {
            let name = "\(p.model)-ane-\(p.variant)"
            let r = try XCTUnwrap(AliasMap.resolve(name), name)
            XCTAssertEqual(r.repo, "srv-sngh/\(name)")
            XCTAssertEqual(r.name, name)
            XCTAssertEqual(r.family, p.family)
            XCTAssertEqual(r.quant, "ane-\(p.variant)")
            // The source model is a separate alias, pulled alongside.
            XCTAssertNotNil(AliasMap.resolve(p.model), p.model)
        }
    }

    func testGPUOnlyModelsHaveNoPackageAlias() {
        for name in ["gte-large-en-v1.5-ane-fp16", "gte-modernbert-base-ane-fp16",
                     "gte-qwen2-7b-ane-fp16", "bge-reranker-large-ane-lin8"] {
            XCTAssertNil(AliasMap.resolve(name), name)
        }
    }
}
