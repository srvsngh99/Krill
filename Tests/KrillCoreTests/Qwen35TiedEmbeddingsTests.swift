import XCTest
import MLX
import MLXNN
@testable import KrillCore

/// Regression guard for `mlx-community/Qwen3.5-4B-MLX-4bit`: unlike
/// Ornith-9B/Qwythos-9B/Qwen3.8-27B (all untied), this checkpoint uses TIED
/// embeddings (`text_config.tie_word_embeddings: true`) and ships no
/// `lm_head.*` weight at all. `Qwen35Config` already decoded that flag
/// correctly - the bug was that `Qwen35ForCausalLM` never consulted it and
/// always built an independent `lm_head` regardless, so a tied checkpoint
/// silently left it randomly-initialized (the VL loader's `verify: []` never
/// caught it) and generated fluent-looking garbage with no load error -
/// confirmed against the real checkpoint via `krill run` before this fix
/// landed.
final class Qwen35TiedEmbeddingsTests: XCTestCase {
    // `head_dim: 8` with the default `partial_rotary_factor` (0.25) gives a
    // rotary dim of 2 - MLX's fused RoPE kernel requires an even dim, and the
    // full-attention layer (idx 1 under `full_attention_interval: 2`) hits it.
    private func makeTextConfigJSON(tieWordEmbeddings: Bool) -> Data {
        let json = """
        {
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "tie_word_embeddings": \(tieWordEmbeddings),
            "full_attention_interval": 2,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4
        }
        """
        return Data(json.utf8)
    }

    /// `Qwen35ForCausalLM.lmHead` must be nil when `tie_word_embeddings` is
    /// true, and non-nil (independent `Linear`) when false — the two
    /// branches `project(_:)` dispatches on.
    func testLmHeadNilOnlyWhenTied() throws {
        let tied = try JSONDecoder().decode(Qwen35Config.self, from: makeTextConfigJSON(tieWordEmbeddings: true))
        XCTAssertNil(Qwen35ForCausalLM(tied).lmHead)

        let untied = try JSONDecoder().decode(Qwen35Config.self, from: makeTextConfigJSON(tieWordEmbeddings: false))
        XCTAssertNotNil(Qwen35ForCausalLM(untied).lmHead)
    }

    /// With `lmHead` nil, `project(_:)` must fall back to the tied embedding
    /// matrix (`Embedding.asLinear`) and produce the SAME logits a manual
    /// embed-transpose matmul would — not silently run through a
    /// randomly-initialized head.
    func testTiedProjectionMatchesEmbeddingTranspose() throws {
        let config = try JSONDecoder().decode(Qwen35Config.self, from: makeTextConfigJSON(tieWordEmbeddings: true))
        let model = Qwen35ForCausalLM(config)
        eval(model)

        let tokens = MLXArray(Int32(0) ..< Int32(3)).reshaped(1, 3)
        let logits = model(tokens)
        eval(logits)

        guard let embedding = model.model.embedTokens as? Embedding else {
            return XCTFail("expected a plain Embedding for embed_tokens")
        }
        let hidden = model.model(tokens)
        let expected = embedding.asLinear(hidden)
        eval(expected)

        XCTAssertEqual(logits.shape, [1, 3, config.vocabSize])
        let got = logits.asType(.float32).asArray(Float.self)
        let want = expected.asType(.float32).asArray(Float.self)
        XCTAssertEqual(got, want, "tied lm_head projection must equal embed_tokens.asLinear")
    }
}
