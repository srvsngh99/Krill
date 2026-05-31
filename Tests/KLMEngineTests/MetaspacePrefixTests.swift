import XCTest
@testable import KLMTokenizer

/// Guards the Metaspace `add_prefix_space` injection that restores correct XLM-R
/// (SentencePiece Unigram) tokenization for new-style Metaspace configs. Without
/// it swift-transformers never prepends `▁`, so every word matches the wrong,
/// non-word-initial vocab entries and embeddings come back garbage.
final class MetaspacePrefixTests: XCTestCase {

    /// New-style Metaspace (prepend_scheme present, add_prefix_space absent), as
    /// nomic-embed-text-v2-moe ships it, nested in a Sequence after
    /// WhitespaceSplit: the Metaspace node must gain add_prefix_space=true.
    func testInjectsOnNewStyleSequenceMetaspace() {
        var node: [String: Any] = [
            "type": "Sequence",
            "pretokenizers": [
                ["type": "WhitespaceSplit"],
                ["type": "Metaspace", "replacement": "\u{2581}",
                 "prepend_scheme": "always", "split": true],
            ],
        ]
        XCTAssertTrue(KLMTokenizer.injectMetaspaceAddPrefixSpace(&node))
        let subs = node["pretokenizers"] as! [[String: Any]]
        let meta = subs[1]
        XCTAssertEqual(meta["add_prefix_space"] as? Bool, true)
        // WhitespaceSplit sibling is untouched.
        XCTAssertNil(subs[0]["add_prefix_space"])
    }

    /// A Metaspace that already declares add_prefix_space (e.g. bge-reranker-v2-m3)
    /// must be left untouched, so the working reranker is unaffected.
    func testLeavesExistingAddPrefixSpaceUntouched() {
        var node: [String: Any] = [
            "type": "Metaspace", "replacement": "\u{2581}",
            "add_prefix_space": true, "prepend_scheme": "always",
        ]
        XCTAssertFalse(KLMTokenizer.injectMetaspaceAddPrefixSpace(&node))
        XCTAssertEqual(node["add_prefix_space"] as? Bool, true)
    }

    /// A Metaspace without prepend_scheme (legacy, add_prefix_space governs alone)
    /// is not a new-style config and must not be rewritten.
    func testIgnoresMetaspaceWithoutPrependScheme() {
        var node: [String: Any] = ["type": "Metaspace", "replacement": "\u{2581}"]
        XCTAssertFalse(KLMTokenizer.injectMetaspaceAddPrefixSpace(&node))
        XCTAssertNil(node["add_prefix_space"])
    }

    /// Non-Metaspace pretokenizers are never touched.
    func testIgnoresNonMetaspace() {
        var node: [String: Any] = ["type": "ByteLevel", "add_prefix_space": false]
        XCTAssertFalse(KLMTokenizer.injectMetaspaceAddPrefixSpace(&node))
    }
}
