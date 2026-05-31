import XCTest
import Foundation
import MLX
@testable import KLMCore

final class EmbeddingModelTests: XCTestCase {
    func testBertConfigDecodingPlainBert() throws {
        let json = """
        {
          "model_type": "bert",
          "hidden_size": 384,
          "num_hidden_layers": 6,
          "num_attention_heads": 12,
          "intermediate_size": 1536,
          "vocab_size": 30522,
          "max_position_embeddings": 512,
          "type_vocab_size": 2,
          "layer_norm_eps": 1e-12,
          "pad_token_id": 0
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(BertEmbeddingConfig.self, from: json)
        XCTAssertEqual(cfg.hiddenSize, 384)
        XCTAssertEqual(cfg.numHiddenLayers, 6)
        XCTAssertEqual(cfg.positionOffset, 0)  // plain BERT: no offset
    }

    func testRobertaConfigAppliesPositionOffset() throws {
        let json = """
        {
          "model_type": "xlm-roberta",
          "hidden_size": 768,
          "num_hidden_layers": 12,
          "num_attention_heads": 12,
          "intermediate_size": 3072,
          "vocab_size": 250002,
          "max_position_embeddings": 514,
          "layer_norm_eps": 1e-5,
          "pad_token_id": 1
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(BertEmbeddingConfig.self, from: json)
        XCTAssertEqual(cfg.positionOffset, 2)  // pad_token_id (1) + 1
        XCTAssertEqual(cfg.typeVocabSize, 2)   // default when absent
    }

    func testMeanPoolingIsL2Normalized() {
        // [1, T=2, H=3]
        let hidden = MLXArray(
            [1.0, 0.0, 0.0,
             3.0, 0.0, 0.0] as [Float]).reshaped(1, 2, 3)
        let v = poolSentenceEmbedding(hidden, pooling: .mean, normalize: true)
        XCTAssertEqual(v.count, 3)
        // mean over T = [2,0,0]; L2-normalized -> [1,0,0]
        XCTAssertEqual(v[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(v[1], 0.0, accuracy: 1e-5)
        let norm = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).squareRoot()
        XCTAssertEqual(norm, 1.0, accuracy: 1e-5)
    }

    func testClsPoolingTakesFirstToken() {
        let hidden = MLXArray(
            [5.0, 0.0, 0.0,
             9.0, 0.0, 0.0] as [Float]).reshaped(1, 2, 3)
        let v = poolSentenceEmbedding(hidden, pooling: .cls, normalize: false)
        XCTAssertEqual(v, [5.0, 0.0, 0.0])
    }

    // MARK: - nomic-bert (RoPE encoder)

    /// nomic configs use GPT-2-style keys (n_embd/n_layer/n_head/n_inner) and
    /// carry rotary params instead of a learned-position table.
    func testNomicConfigDecodingGPT2StyleKeys() throws {
        let json = """
        {
          "model_type": "nomic_bert",
          "architectures": ["NomicBertModel"],
          "n_embd": 768,
          "n_layer": 12,
          "n_head": 12,
          "n_inner": 3072,
          "vocab_size": 30528,
          "type_vocab_size": 2,
          "rotary_emb_base": 1000,
          "rotary_emb_fraction": 1.0,
          "max_position_embeddings": 2048,
          "n_positions": 8192
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(NomicBertConfig.self, from: json)
        XCTAssertEqual(cfg.hiddenSize, 768)
        XCTAssertEqual(cfg.numHiddenLayers, 12)
        XCTAssertEqual(cfg.numAttentionHeads, 12)
        XCTAssertEqual(cfg.intermediateSize, 3072)
        XCTAssertEqual(cfg.headDim, 64)
        XCTAssertEqual(cfg.ropeBase, 1000)
        XCTAssertEqual(cfg.rotaryFraction, 1.0)
        XCTAssertEqual(cfg.maxTokens, 2048)  // trained window, not n_positions
    }

    /// Tolerate HF BERT-style aliases (hidden_size/num_hidden_layers/...) and
    /// fall back to sane rotary defaults when the keys are absent.
    func testNomicConfigDecodingBertStyleKeysAndDefaults() throws {
        let json = """
        {
          "model_type": "nomic_bert",
          "hidden_size": 256,
          "num_hidden_layers": 4,
          "num_attention_heads": 8,
          "intermediate_size": 512,
          "vocab_size": 1000
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(NomicBertConfig.self, from: json)
        XCTAssertEqual(cfg.hiddenSize, 256)
        XCTAssertEqual(cfg.numHiddenLayers, 4)
        XCTAssertEqual(cfg.numAttentionHeads, 8)
        XCTAssertEqual(cfg.typeVocabSize, 2)     // default
        XCTAssertEqual(cfg.ropeBase, 1000)       // default
        XCTAssertEqual(cfg.rotaryFraction, 1.0)  // default
        XCTAssertEqual(cfg.maxTokens, 2048)      // default
    }

    /// Exercises the full nomic wiring (embeddings + emb_ln -> fused-Wqkv RoPE
    /// attention -> SwiGLU MLP -> post-norm) on a tiny random-init model: the
    /// last-hidden-state must be `[1, T, H]` and deterministic. Numerical
    /// parity against the reference forward is validated out-of-band against
    /// real weights (a 12-layer numpy reimplementation matches the Swift
    /// output to cosine 1.0000); this guards the architecture plumbing.
    func testNomicForwardShapeAndDeterminism() throws {
        let json = """
        {
          "model_type": "nomic_bert",
          "n_embd": 32, "n_layer": 2, "n_head": 4, "n_inner": 64,
          "vocab_size": 100, "type_vocab_size": 2,
          "rotary_emb_base": 1000, "rotary_emb_fraction": 1.0,
          "max_position_embeddings": 64
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(NomicBertConfig.self, from: json)
        let model = NomicBertEmbeddingModel(cfg)
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5)

        let out = model.lastHiddenState(tokens)
        out.eval()
        XCTAssertEqual(out.shape, [1, 5, 32])

        // Deterministic: same input -> identical output (no RNG in forward).
        let out2 = model.lastHiddenState(tokens)
        out2.eval()
        XCTAssertTrue(allClose(out, out2, atol: 0).all().item(Bool.self))

        // Pools to a unit-norm sentence vector of width H.
        let vec = poolSentenceEmbedding(out, pooling: .mean, normalize: true)
        XCTAssertEqual(vec.count, 32)
        let norm = vec.reduce(0) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(norm, 1.0, accuracy: 1e-4)
    }

    /// Pins the module's parameter keys to the nomic-bert checkpoint key
    /// template. `EmbeddingEngine` loads nomic weights with `strictVerify`, so a
    /// stray `@ModuleInfo(key:)` typo would only blow up at real-weight load
    /// (no fixture in CI). This catches it from the param tree alone.
    func testNomicParameterKeysMatchCheckpointTemplate() throws {
        let json = """
        {
          "model_type": "nomic_bert",
          "n_embd": 32, "n_layer": 2, "n_head": 4, "n_inner": 64,
          "vocab_size": 100, "type_vocab_size": 2,
          "rotary_emb_base": 1000, "rotary_emb_fraction": 1.0,
          "max_position_embeddings": 64
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(NomicBertConfig.self, from: json)
        let model = NomicBertEmbeddingModel(cfg)

        var expected: Set<String> = [
            "embeddings.word_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
            "emb_ln.weight", "emb_ln.bias",
        ]
        for i in 0 ..< cfg.numHiddenLayers {
            let p = "encoder.layers.\(i)."
            expected.formUnion([
                p + "attn.Wqkv.weight", p + "attn.out_proj.weight",
                p + "mlp.fc11.weight", p + "mlp.fc12.weight", p + "mlp.fc2.weight",
                p + "norm1.weight", p + "norm1.bias",
                p + "norm2.weight", p + "norm2.bias",
            ])
        }

        let actual = Set(model.parameters().flattened().map { $0.0 })
        XCTAssertEqual(actual, expected,
            "param keys diverge from the nomic checkpoint; strictVerify load would fail")
    }

    // MARK: - GTE-v1.5 (RoPE encoder, GeGLU, CLS pooling)

    func testGTEConfigDecoding() throws {
        let json = """
        {
          "model_type": "new", "architectures": ["NewModel"],
          "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
          "intermediate_size": 3072, "vocab_size": 30528,
          "max_position_embeddings": 8192, "rope_theta": 500000, "layer_norm_eps": 1e-12
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(GTEConfig.self, from: json)
        XCTAssertEqual(cfg.headDim, 64)
        XCTAssertEqual(cfg.ropeBase, 500000)
        XCTAssertEqual(cfg.maxTokens, 8192)
    }

    func testGTEForwardShapeAndKeys() throws {
        let json = """
        {
          "model_type": "new",
          "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
          "intermediate_size": 64, "vocab_size": 100,
          "max_position_embeddings": 4096, "rope_theta": 500000, "layer_norm_eps": 1e-12
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(GTEConfig.self, from: json)
        let model = GTEEmbeddingModel(cfg)
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5)
        let out = model.lastHiddenState(tokens)
        out.eval()
        XCTAssertEqual(out.shape, [1, 5, 32])

        var expected: Set<String> = [
            "embeddings.word_embeddings.weight",
            "embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias",
        ]
        for i in 0 ..< cfg.numHiddenLayers {
            let p = "encoder.layer.\(i)."
            expected.formUnion([
                p + "attention.qkv_proj.weight", p + "attention.qkv_proj.bias",
                p + "attention.o_proj.weight", p + "attention.o_proj.bias",
                p + "attn_ln.weight", p + "attn_ln.bias",
                p + "mlp.up_gate_proj.weight",
                p + "mlp.down_proj.weight", p + "mlp.down_proj.bias",
                p + "mlp_ln.weight", p + "mlp_ln.bias",
            ])
        }
        let actual = Set(model.parameters().flattened().map { $0.0 })
        XCTAssertEqual(actual, expected,
            "param keys diverge from the GTE-v1.5 checkpoint; strictVerify load would fail")
    }

    // MARK: - MPNet (relative-attention-bias encoder)

    func testMPNetConfigDecoding() throws {
        let json = """
        {
          "model_type": "mpnet",
          "architectures": ["MPNetForMaskedLM"],
          "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
          "intermediate_size": 3072, "vocab_size": 30527,
          "max_position_embeddings": 514, "relative_attention_num_buckets": 32,
          "layer_norm_eps": 1e-5, "pad_token_id": 1
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(MPNetConfig.self, from: json)
        XCTAssertEqual(cfg.hiddenSize, 768)
        XCTAssertEqual(cfg.headDim, 64)
        XCTAssertEqual(cfg.relativeAttentionNumBuckets, 32)
        XCTAssertEqual(cfg.positionOffset, 2)   // pad_token_id (1) + 1
        XCTAssertEqual(cfg.maxTokens, 512)       // 514 - offset
    }

    func testMPNetForwardShapeAndDeterminism() throws {
        let json = """
        {
          "model_type": "mpnet",
          "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
          "intermediate_size": 64, "vocab_size": 100,
          "max_position_embeddings": 40, "relative_attention_num_buckets": 32,
          "layer_norm_eps": 1e-5, "pad_token_id": 1
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(MPNetConfig.self, from: json)
        let model = MPNetEmbeddingModel(cfg)
        let tokens = MLXArray([Int32(0), 1, 2, 3, 4, 5]).reshaped(1, 6)

        let out = model.lastHiddenState(tokens)
        out.eval()
        XCTAssertEqual(out.shape, [1, 6, 32])
        let out2 = model.lastHiddenState(tokens)
        out2.eval()
        XCTAssertTrue(allClose(out, out2, atol: 0).all().item(Bool.self))
    }

    /// Pins MPNet's module parameter keys to the checkpoint template. The engine
    /// loads MPNet with `strictVerify` (after dropping `pooler.*`/`position_ids`),
    /// so a stray `@ModuleInfo(key:)` would only fail at real-weight load.
    func testMPNetParameterKeysMatchCheckpointTemplate() throws {
        let json = """
        {
          "model_type": "mpnet",
          "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
          "intermediate_size": 64, "vocab_size": 100,
          "max_position_embeddings": 40, "relative_attention_num_buckets": 32,
          "layer_norm_eps": 1e-5, "pad_token_id": 1
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(MPNetConfig.self, from: json)
        let model = MPNetEmbeddingModel(cfg)

        var expected: Set<String> = [
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias",
            "encoder.relative_attention_bias.weight",
        ]
        for i in 0 ..< cfg.numHiddenLayers {
            let p = "encoder.layer.\(i)."
            for proj in ["q", "k", "v", "o"] {
                expected.insert(p + "attention.attn.\(proj).weight")
                expected.insert(p + "attention.attn.\(proj).bias")
            }
            for (mod, suffix) in [("attention.LayerNorm", ["weight", "bias"]),
                                  ("intermediate.dense", ["weight", "bias"]),
                                  ("output.dense", ["weight", "bias"]),
                                  ("output.LayerNorm", ["weight", "bias"])] {
                for s in suffix { expected.insert(p + mod + "." + s) }
            }
        }
        let actual = Set(model.parameters().flattened().map { $0.0 })
        XCTAssertEqual(actual, expected,
            "param keys diverge from the MPNet checkpoint; strictVerify load would fail")
    }

    // MARK: - Decoder-LLM embedders (last-token pooling)

    func testPoolingStringParseToleratesCaseAndSeparators() {
        XCTAssertEqual(EmbeddingPooling.from("mean"), .mean)
        XCTAssertEqual(EmbeddingPooling.from("CLS"), .cls)
        // The camelCase rawValue would defeat a lowercased init(rawValue:);
        // these spellings must all resolve to .lastToken.
        XCTAssertEqual(EmbeddingPooling.from("lasttoken"), .lastToken)
        XCTAssertEqual(EmbeddingPooling.from("last_token"), .lastToken)
        XCTAssertEqual(EmbeddingPooling.from("LastToken"), .lastToken)
        XCTAssertEqual(EmbeddingPooling.from("last"), .lastToken)
        XCTAssertNil(EmbeddingPooling.from("bogus"))
    }

    func testLastTokenPoolingPicksFinalRow() {
        // [1, T=3, H=2]
        let hidden = MLXArray(
            [1.0, 2.0,
             3.0, 4.0,
             9.0, 12.0] as [Float]).reshaped(1, 3, 2)
        let raw = poolSentenceEmbedding(hidden, pooling: .lastToken, normalize: false)
        XCTAssertEqual(raw, [9.0, 12.0])  // final token's hidden state
        // L2-normalized: [9,12] / 15 = [0.6, 0.8]
        let v = poolSentenceEmbedding(hidden, pooling: .lastToken, normalize: true)
        XCTAssertEqual(v[0], 0.6, accuracy: 1e-5)
        XCTAssertEqual(v[1], 0.8, accuracy: 1e-5)
    }

    /// A causal backbone (here Qwen2) must satisfy `SentenceEmbeddingEncoder`,
    /// returning the pre-lm_head hidden state `[1, T, H]` so decoder-LLM
    /// embedders (gte-Qwen2, e5-mistral) reuse the validated causal forward.
    func testQwenForCausalLMActsAsDecoderEmbedder() throws {
        let dict: [String: Any] = [
            "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "num_hidden_layers": 2, "vocab_size": 256,
            "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0,
            "max_position_embeddings": 4096, "model_type": "qwen2", "head_dim": 16,
        ]
        let data = try JSONSerialization.data(withJSONObject: dict)
        let cfg = try JSONDecoder().decode(QwenConfig.self, from: data)

        let encoder: any SentenceEmbeddingEncoder = QwenForCausalLM(cfg)
        let tokens = MLXArray((0 ..< 5).map { Int32($0) }).reshaped(1, 5)
        let hidden = encoder.lastHiddenState(tokens)
        hidden.eval()
        XCTAssertEqual(hidden.shape, [1, 5, 64])

        let vec = poolSentenceEmbedding(hidden, pooling: .lastToken, normalize: true)
        XCTAssertEqual(vec.count, 64)
        let norm = vec.reduce(0) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(norm, 1.0, accuracy: 1e-4)
    }

    /// Gemma backbone conformance (for bge-multilingual-gemma2): the inner model
    /// must expose its final hidden state `[1, T, H]` for last-token pooling.
    func testGemmaForCausalLMActsAsDecoderEmbedder() throws {
        let dict: [String: Any] = [
            "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "num_hidden_layers": 2, "vocab_size": 256,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "max_position_embeddings": 4096, "head_dim": 16,
        ]
        let data = try JSONSerialization.data(withJSONObject: dict)
        let cfg = try JSONDecoder().decode(GemmaConfig.self, from: data)

        let encoder: any SentenceEmbeddingEncoder = GemmaForCausalLM(cfg)
        let tokens = MLXArray((0 ..< 5).map { Int32($0) }).reshaped(1, 5)
        let hidden = encoder.lastHiddenState(tokens)
        hidden.eval()
        XCTAssertEqual(hidden.shape, [1, 5, 64])
    }
}
