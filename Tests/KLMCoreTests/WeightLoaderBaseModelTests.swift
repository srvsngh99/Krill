import XCTest
@testable import KLMCore

/// Guards the base-model key normalization that lets sentence-embedding
/// decoders shipped as `MistralModel`/`Qwen2Model` (no `model.` prefix, no
/// lm_head) load into the causal `*ForCausalLM` backbones used as embedders.
final class WeightLoaderBaseModelTests: XCTestCase {

    /// A bare base-model checkpoint (e5-mistral / SFR layout) must be detected
    /// so its backbone keys get the `model.` prefix the modules expect.
    func testBareBackboneNeedsPrefix() {
        let keys = [
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "norm.weight",
        ]
        XCTAssertTrue(baseModelNeedsModelPrefix(keys))
    }

    /// A normal `*ForCausalLM` checkpoint already nests under `model.` and adds
    /// `lm_head`; it must be left untouched (no double `model.model.` prefix).
    func testNestedCausalCheckpointUntouched() {
        let keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        XCTAssertFalse(baseModelNeedsModelPrefix(keys))
    }

    /// An empty or non-decoder key set must not trigger the rewrite.
    func testNoBackboneKeysIsFalse() {
        XCTAssertFalse(baseModelNeedsModelPrefix([String]()))
        XCTAssertFalse(baseModelNeedsModelPrefix(["word_embeddings.weight", "encoder.layer.0.attention.qkv_proj.weight"]))
    }
}
