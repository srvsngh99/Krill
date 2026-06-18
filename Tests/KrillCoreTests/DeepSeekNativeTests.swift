import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import KrillCore

/// Native runtime tests for the DeepSeek MoE family (`DeepseekV2ForCausalLM`
/// / `DeepseekV3ForCausalLM`). Pins config parsing (MLA + YaRN + MoE fields),
/// the dense-layer prefix, MLA module structure (q_lora present/absent), the
/// V3 sigmoid gate bias parameter, forward finiteness, and routing on tiny
/// synthetic instances. Logit parity vs mlx-lm (both V2 softmax and V3
/// noaux_tc gating) is covered by the gated `DeepSeekParityTests`.
final class DeepSeekNativeTests: XCTestCase {

    private func tinyConfigJSON(
        modelType: String = "deepseek_v2",
        hidden: Int = 32,
        heads: Int = 2,
        layers: Int = 2,
        vocab: Int = 64,
        qLoraRank: Int? = nil,
        kvLoraRank: Int = 16,
        qkRope: Int = 8,
        qkNope: Int = 16,
        vHeadDim: Int = 16,
        moeIntermediate: Int = 32,
        nRouted: Int = 4,
        topK: Int = 2,
        nShared: Int = 1,
        firstKDense: Int = 1,
        scoringFunc: String = "softmax",
        topkMethod: String = "greedy",
        nGroup: Int = 1,
        topkGroup: Int = 1,
        normTopK: Bool = false
    ) -> [String: Any] {
        var cfg: [String: Any] = [
            "architectures": [modelType == "deepseek_v3"
                ? "DeepseekV3ForCausalLM" : "DeepseekV2ForCausalLM"],
            "model_type": modelType,
            "hidden_size": hidden,
            "intermediate_size": 64,
            "moe_intermediate_size": moeIntermediate,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": heads,
            "vocab_size": vocab,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 256,
            "kv_lora_rank": kvLoraRank,
            "qk_rope_head_dim": qkRope,
            "qk_nope_head_dim": qkNope,
            "v_head_dim": vHeadDim,
            "n_routed_experts": nRouted,
            "n_shared_experts": nShared,
            "num_experts_per_tok": topK,
            "routed_scaling_factor": 1.0,
            "topk_method": topkMethod,
            "scoring_func": scoringFunc,
            "norm_topk_prob": normTopK,
            "n_group": nGroup,
            "topk_group": topkGroup,
            "first_k_dense_replace": firstKDense,
            "moe_layer_freq": 1,
            "rope_scaling": [
                "type": "yarn", "factor": 4.0, "beta_fast": 32, "beta_slow": 1,
                "mscale": 1.0, "mscale_all_dim": 0.0,
                "original_max_position_embeddings": 256,
            ],
            "quantization": ["group_size": 16, "bits": 4],
        ]
        if let qLoraRank { cfg["q_lora_rank"] = qLoraRank }
        return cfg
    }

    private func decode(_ json: [String: Any]) throws -> DeepSeekConfig {
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder().decode(DeepSeekConfig.self, from: data)
    }

    // MARK: - Config

    func testConfigParsesMLAAndMoEFields() throws {
        let cfg = try decode(tinyConfigJSON(
            kvLoraRank: 512, qkRope: 64, qkNope: 128, vHeadDim: 128,
            nRouted: 64, topK: 6, nShared: 2, firstKDense: 1))
        XCTAssertEqual(cfg.kvLoraRank, 512)
        XCTAssertEqual(cfg.qkRopeHeadDim, 64)
        XCTAssertEqual(cfg.qkNopeHeadDim, 128)
        XCTAssertEqual(cfg.vHeadDim, 128)
        XCTAssertEqual(cfg.headDim, 192, "q_head_dim = nope + rope")
        XCTAssertEqual(cfg.nRoutedExperts, 64)
        XCTAssertEqual(cfg.numExpertsPerToken, 6)
        XCTAssertEqual(cfg.nSharedExperts, 2)
        XCTAssertEqual(cfg.ropeScaling.factor, 4.0)
        XCTAssertEqual(cfg.ropeScaling.originalMaxPositionEmbeddings, 256)
    }

    func testIsMoELayerHonorsFirstKDenseReplace() throws {
        let cfg = try decode(tinyConfigJSON(layers: 3, firstKDense: 1))
        XCTAssertFalse(cfg.isMoELayer(0), "Layer 0 is dense (first_k_dense_replace=1)")
        XCTAssertTrue(cfg.isMoELayer(1))
        XCTAssertTrue(cfg.isMoELayer(2))
    }

    // MARK: - MLA module structure

    func testMLAUsesDirectQProjWhenNoLora() throws {
        let cfg = try decode(tinyConfigJSON(qLoraRank: nil))
        let attn = DeepSeekAttention(cfg)
        let keys = Set(attn.parameters().flattened().map { $0.0 })
        XCTAssertTrue(keys.contains("q_proj.weight"), "No q_lora_rank -> direct q_proj")
        XCTAssertFalse(keys.contains { $0.hasPrefix("q_a_proj") })
        XCTAssertTrue(keys.contains("kv_a_proj_with_mqa.weight"))
        XCTAssertTrue(keys.contains("kv_a_layernorm.weight"))
        XCTAssertTrue(keys.contains("kv_b_proj.weight"))
    }

    func testMLAUsesLoraBottleneckWhenConfigured() throws {
        let cfg = try decode(tinyConfigJSON(qLoraRank: 16))
        let attn = DeepSeekAttention(cfg)
        let keys = Set(attn.parameters().flattened().map { $0.0 })
        XCTAssertTrue(keys.contains("q_a_proj.weight"), "q_lora_rank set -> q_a/q_b bottleneck")
        XCTAssertTrue(keys.contains("q_a_layernorm.weight"))
        XCTAssertTrue(keys.contains("q_b_proj.weight"))
        XCTAssertFalse(keys.contains("q_proj.weight"))
    }

    // MARK: - Forward + parameter cache

    func testForwardPassProducesFiniteLogits() throws {
        let cfg = try decode(tinyConfigJSON())
        let model = DeepSeekForCausalLM(cfg)
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 3])
        let logits = model(tokens, caches: nil)
        XCTAssertEqual(logits.shape, [1, 3, 64])
        eval(logits)
        let arr = logits.asArray(Float.self)
        XCTAssertNil(arr.firstIndex { !$0.isFinite }, "Forward must not emit NaN/Inf")
    }

    /// Dense layer 0 exposes a plain MLP; MoE layer 1 exposes the gate +
    /// stacked experts + shared experts. V2 has no e_score_correction_bias.
    func testParameterCacheLayoutV2() throws {
        let cfg = try decode(tinyConfigJSON(modelType: "deepseek_v2", layers: 2, firstKDense: 1))
        let model = DeepSeekForCausalLM(cfg)
        let names = Set(model.parameters().flattened().map { $0.0 })

        // Dense layer 0.
        XCTAssertTrue(names.contains("model.layers.0.mlp.gate_proj.weight"),
            "Dense layer 0 must expose a plain MLP")
        XCTAssertFalse(names.contains("model.layers.0.mlp.gate.weight"),
            "Dense layer 0 must not expose the MoE router")
        // MoE layer 1.
        XCTAssertTrue(names.contains("model.layers.1.mlp.gate.weight"),
            "MoE router weight missing")
        XCTAssertTrue(names.contains("model.layers.1.mlp.switch_mlp.gate_proj.weight"),
            "SwitchGLU gate_proj missing")
        XCTAssertTrue(names.contains("model.layers.1.mlp.shared_experts.gate_proj.weight"),
            "Shared-expert MLP missing")
        // MLA attention keys.
        XCTAssertTrue(names.contains("model.layers.1.self_attn.kv_a_proj_with_mqa.weight"))
        XCTAssertTrue(names.contains("model.layers.1.self_attn.kv_b_proj.weight"))
        // V2 has no sigmoid bias.
        XCTAssertFalse(names.contains { $0.contains("e_score_correction_bias") },
            "V2 (softmax) must not declare e_score_correction_bias")
        XCTAssertTrue(names.contains("lm_head.weight"))
    }

    /// V3 (noaux_tc / sigmoid) declares the e_score_correction_bias selection bias.
    func testV3DeclaresScoreCorrectionBias() throws {
        let cfg = try decode(tinyConfigJSON(
            modelType: "deepseek_v3", scoringFunc: "sigmoid", topkMethod: "noaux_tc",
            normTopK: true))
        let model = DeepSeekForCausalLM(cfg)
        let names = Set(model.parameters().flattened().map { $0.0 })
        XCTAssertTrue(names.contains("model.layers.1.mlp.gate.e_score_correction_bias"),
            "V3 noaux_tc gate must declare e_score_correction_bias")
    }

    // MARK: - Routing

    func testExpertCountsSumToAssignments() throws {
        let cfg = try decode(tinyConfigJSON(
            hidden: 64, layers: 2, nRouted: 8, topK: 4, firstKDense: 0))
        let moe = DeepSeekMoE(cfg)
        let batch = 2, seqLen = 6
        eval(moe(MLXRandom.normal([batch, seqLen, 64]).asType(.float32)))
        XCTAssertEqual(moe.cumulativeExpertCounts.reduce(0, +), batch * seqLen * 4,
            "Cumulative expert counts must sum to N * topK")
    }
}
