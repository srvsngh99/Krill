import XCTest
import MLX
@testable import KrillCore

/// Unit tests for `qwen35VLKeyRewrite`, the fix for a raw HF/torch-format
/// Qwen3.5-VL snapshot (e.g. `huggingface-cli download Qwen/Qwen3.5-4B`, no
/// `mlx_vlm.convert` step) loading via `loadQwen35VL` and producing silent
/// garbage: its top-level key prefixes (`model.language_model.*`,
/// `model.visual.*`, bare `lm_head.*`) never matched
/// `Qwen35VLForConditionalGeneration`'s module tree (`language_model.model.*`,
/// `vision_tower.*`, `language_model.lm_head.*`), and the loader's lax
/// `verify: []` let every key silently fail to bind instead of erroring, so
/// the model decoded from its random Swift initialization.
///
/// Literal key patterns below are taken directly from the real
/// `Qwen/Qwen3.5-4B` `model.safetensors.index.json` (raw) and
/// `mlx-community/Qwen3.5-4B-MLX-4bit`'s (already mlx_vlm-format).
final class Qwen35VLKeyRewriteTests: XCTestCase {
    func testRewritesRawHFTorchKeys() {
        let one = MLXArray([Float(1)])
        let raw = [
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            "model.language_model.norm.weight",
            "model.visual.patch_embed.proj.bias",
            "model.visual.blocks.0.attn.proj.bias",
            "model.visual.merger.linear_fc1.bias",
            "lm_head.weight",
        ]
        let out = qwen35VLKeyRewrite(Dictionary(uniqueKeysWithValues: raw.map { ($0, one) }))

        XCTAssertEqual(out.count, raw.count, "rewrite must not drop or merge keys")
        XCTAssertNotNil(out["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(out["language_model.model.layers.0.input_layernorm.weight"])
        XCTAssertNotNil(out["language_model.model.layers.0.linear_attn.conv1d.weight"])
        XCTAssertNotNil(out["language_model.model.norm.weight"])
        XCTAssertNotNil(out["vision_tower.patch_embed.proj.bias"])
        XCTAssertNotNil(out["vision_tower.blocks.0.attn.proj.bias"])
        XCTAssertNotNil(out["vision_tower.merger.linear_fc1.bias"])
        XCTAssertNotNil(out["language_model.lm_head.weight"])
        // None of the raw prefixes should survive.
        for key in out.keys {
            XCTAssertFalse(key.hasPrefix("model.language_model."))
            XCTAssertFalse(key.hasPrefix("model.visual."))
        }
        XCTAssertNil(out["lm_head.weight"], "bare lm_head must be nested under language_model")
    }

    func testLeavesAlreadyMlxVlmFormatKeysUntouched() {
        let one = MLXArray([Float(1)])
        let mlxVlm = [
            "language_model.model.embed_tokens.weight",
            "language_model.model.embed_tokens.scales",
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            "language_model.model.norm.weight",
            "language_model.lm_head.weight",
            "vision_tower.patch_embed.proj.bias",
            "vision_tower.blocks.0.attn.proj.bias",
        ]
        let out = qwen35VLKeyRewrite(Dictionary(uniqueKeysWithValues: mlxVlm.map { ($0, one) }))
        XCTAssertEqual(Set(out.keys), Set(mlxVlm),
            "an already mlx_vlm-format checkpoint must pass through byte-for-byte")
    }

    func testMTPAndOtherKeysPassThroughUnaffected() {
        // `mtp.*` keys are dropped separately by loadQwen35VL's existing
        // sanitize loop, not by this rewrite - it must leave them alone.
        let one = MLXArray([Float(1)])
        let keys = ["mtp.fc.weight", "mtp.norm.weight"]
        let out = qwen35VLKeyRewrite(Dictionary(uniqueKeysWithValues: keys.map { ($0, one) }))
        XCTAssertEqual(Set(out.keys), Set(keys))
    }

    func testEmptyInputStaysEmpty() {
        XCTAssertTrue(qwen35VLKeyRewrite([:]).isEmpty)
    }
}
