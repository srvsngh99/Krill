import XCTest
@testable import KLMCore

/// WS6 foundation: the loader recognizes MoE architectures AND either
/// routes them to a native Swift+MLX runtime (Qwen 3 MoE, Mixtral) or
/// refuses to instantiate the not-yet-ported families (Qwen2-MoE / OLMoE /
/// DeepSeek) instead of silently falling back to a dense text loader that
/// would crash on router/expert keys. Tests pin both halves of the
/// contract.
final class MoELoaderRejectionTests: XCTestCase {

    private func writeConfig(_ json: [String: Any], dirSlug: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-moe-\(dirSlug)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: json)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    /// Mixtral now has a native runtime (`loadMixtral`), so it must reach
    /// the native arm rather than the bridge rejection. With an empty config
    /// dir (no safetensors) the native arm fails specifically with
    /// `WeightLoadError.noSafetensorsFiles` -- proof it routed past the
    /// rejection and into the native loader.
    func testMixtralNativeReachesNativeArm() throws {
        let dir = try writeConfig([
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        ], dirSlug: "mixtral-native")
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            _ = try loadModel(from: dir)
            XCTFail("Expected WeightLoadError.noSafetensorsFiles for an empty config dir")
        } catch let error as WeightLoadError {
            guard case .noSafetensorsFiles = error else {
                XCTFail("Expected noSafetensorsFiles, got \(error)")
                return
            }
            // OK: reached the native arm, failed at weight load.
        } catch let error as ModelLoadError {
            if case .unsupportedArchitecture(let msg) = error {
                XCTFail("Mixtral native arm must not throw unsupportedArchitecture; got: \(msg)")
            } else {
                XCTFail("Unexpected ModelLoadError: \(error)")
            }
        }
    }

    /// Qwen 3 MoE config that is large enough to satisfy the
    /// native loader's config decoder but contains no actual
    /// weights. Used by both the opt-in-on and opt-in-off arms.
    private func writeQwen3MoEConfig(dirSlug: String) throws -> URL {
        return try writeConfig([
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "num_hidden_layers": 48,
            "vocab_size": 151936,
            "head_dim": 128,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "norm_topk_prob": true,
            "tie_word_embeddings": false,
        ], dirSlug: dirSlug)
    }

    func testQwen3MoEReachesNativeArm() throws {
        // Qwen 3 MoE is native and is the only path: the mlx-lm sidecar bridge
        // and its `KRILL_NATIVE_MOE=0` opt-out were deleted. With an empty
        // config dir the loader fails specifically with
        // `WeightLoadError.noSafetensorsFiles`, proving it reached the native
        // arm. The specific error type discriminates "reached native arm and
        // failed at weight load" from an unrelated trap.
        let dir = try writeQwen3MoEConfig(dirSlug: "qwen3moe-native")
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            _ = try loadModel(from: dir)
            XCTFail("Expected WeightLoadError.noSafetensorsFiles "
                + "for an empty config dir on the native arm")
        } catch let error as WeightLoadError {
            guard case .noSafetensorsFiles = error else {
                XCTFail("Expected noSafetensorsFiles, got \(error)")
                return
            }
        } catch let error as ModelLoadError {
            if case .unsupportedArchitecture(let msg) = error {
                XCTFail("Qwen 3 MoE native arm must not throw "
                    + "unsupportedArchitecture; got: \(msg)")
            } else {
                XCTFail("Unexpected ModelLoadError: \(error)")
            }
        }
    }

    /// Qwen 2 MoE now has a native runtime; like Mixtral it must reach the
    /// native arm and fail at weight load (empty config dir) rather than at
    /// the bridge rejection.
    func testQwen2MoENativeReachesNativeArm() throws {
        let dir = try writeConfig([
            "architectures": ["Qwen2MoeForCausalLM"],
            "model_type": "qwen2_moe",
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_hidden_layers": 24,
            "vocab_size": 151936,
            "num_experts": 60,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1408,
            "shared_expert_intermediate_size": 5632,
        ], dirSlug: "qwen2moe-native")
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            _ = try loadModel(from: dir)
            XCTFail("Expected WeightLoadError.noSafetensorsFiles for an empty config dir")
        } catch let error as WeightLoadError {
            guard case .noSafetensorsFiles = error else {
                XCTFail("Expected noSafetensorsFiles, got \(error)")
                return
            }
        } catch let error as ModelLoadError {
            if case .unsupportedArchitecture(let msg) = error {
                XCTFail("Qwen2-MoE native arm must not throw unsupportedArchitecture; got: \(msg)")
            } else {
                XCTFail("Unexpected ModelLoadError: \(error)")
            }
        }
    }

    /// OLMoE now has a native runtime; it must reach the native arm and fail
    /// at weight load (empty config dir) rather than the bridge rejection.
    func testOLMoENativeReachesNativeArm() throws {
        let dir = try writeConfig([
            "architectures": ["OlmoeForCausalLM"],
            "model_type": "olmoe",
            "hidden_size": 2048,
            "intermediate_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_hidden_layers": 16,
            "vocab_size": 50304,
            "num_experts": 64,
            "num_experts_per_tok": 8,
        ], dirSlug: "olmoe-native")
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            _ = try loadModel(from: dir)
            XCTFail("Expected WeightLoadError.noSafetensorsFiles for an empty config dir")
        } catch let error as WeightLoadError {
            guard case .noSafetensorsFiles = error else {
                XCTFail("Expected noSafetensorsFiles, got \(error)")
                return
            }
        } catch let error as ModelLoadError {
            if case .unsupportedArchitecture(let msg) = error {
                XCTFail("OLMoE native arm must not throw unsupportedArchitecture; got: \(msg)")
            } else {
                XCTFail("Unexpected ModelLoadError: \(error)")
            }
        }
    }

    private func deepSeekConfig(arch: String, modelType: String) -> [String: Any] {
        [
            "architectures": [arch],
            "model_type": modelType,
            "hidden_size": 256,
            "intermediate_size": 512,
            "moe_intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 1024,
            "kv_lora_rank": 128,
            "qk_rope_head_dim": 32,
            "qk_nope_head_dim": 64,
            "v_head_dim": 64,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 1,
            "rope_scaling": [
                "type": "yarn", "factor": 4.0, "beta_fast": 32, "beta_slow": 1,
                "mscale": 1.0, "mscale_all_dim": 0.0,
                "original_max_position_embeddings": 256,
            ],
        ]
    }

    /// DeepSeek-V2 / V2-Lite has a native runtime. It is the `.deepseek` family
    /// (dense chat routing), so the server reaches it via the dense engine ->
    /// `loadModel` -> `loadDeepSeek`. With an empty config dir the native arm
    /// fails specifically with `WeightLoadError.noSafetensorsFiles`, proving it
    /// routed into the native loader rather than the old bridge rejection.
    func testDeepSeekV2NativeReachesNativeArm() throws {
        let dir = try writeConfig(
            deepSeekConfig(arch: "DeepseekV2ForCausalLM", modelType: "deepseek_v2"),
            dirSlug: "deepseekv2-native")
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            _ = try loadModel(from: dir)
            XCTFail("Expected WeightLoadError.noSafetensorsFiles for deepseek_v2")
        } catch let error as WeightLoadError {
            guard case .noSafetensorsFiles = error else {
                XCTFail("Expected noSafetensorsFiles, got \(error)")
                return
            }
        } catch let error as ModelLoadError {
            if case .unsupportedArchitecture(let msg) = error {
                XCTFail("DeepSeek-V2 native arm must not throw unsupportedArchitecture; got: \(msg)")
            } else {
                XCTFail("Unexpected ModelLoadError: \(error)")
            }
        }
    }

    /// DeepSeek-V3 uses an absorbed MLA layout the native runtime does not load
    /// yet; it must fail fast with a clear message (not a cryptic strict-verify
    /// error), naming the V2/V2-Lite native support and the backlog follow-up.
    func testDeepSeekV3GivesClearAbsorbedMLAError() throws {
        let dir = try writeConfig(
            deepSeekConfig(arch: "DeepseekV3ForCausalLM", modelType: "deepseek_v3"),
            dirSlug: "deepseekv3-absorbed")
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertThrowsError(try loadModel(from: dir)) { error in
            guard let modelError = error as? ModelLoadError,
                  case .unsupportedArchitecture(let msg) = modelError else {
                XCTFail("Expected unsupportedArchitecture for deepseek_v3, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("absorbed"),
                "V3 rejection must explain the absorbed-MLA limitation")
            XCTAssertTrue(msg.contains("V2-Lite") || msg.contains("V2 / V2-Lite"),
                "V3 rejection must point at the native V2/V2-Lite support")
        }
    }

    func testDenseQwen3IsNotMisroutedToMoE() throws {
        // The dense Qwen 3 architecture must NOT route through the
        // MoE rejection arm; it should reach `loadQwen`. We cannot
        // construct a full quantized Qwen 3 here without weights,
        // but a config-only invocation will fail at weight loading
        // rather than at the MoE arm. The error class is the
        // discriminator: MoE rejection throws
        // `unsupportedArchitecture` with WS6 in the message; the
        // dense path throws something else (or succeeds if weights
        // exist).
        let dir = try writeConfig([
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "head_dim": 128,
        ], dirSlug: "qwen3dense")
        defer { try? FileManager.default.removeItem(at: dir) }

        // Expect a failure at weight loading (no safetensors in the
        // temp dir), NOT the WS6 MoE rejection.
        do {
            _ = try loadModel(from: dir)
            XCTFail("Expected weight-load failure for empty config dir")
        } catch let error as ModelLoadError {
            if case .unsupportedArchitecture(let msg) = error {
                XCTAssertFalse(msg.contains("MoE bridge"),
                    "Dense Qwen 3 must NOT route through the MoE rejection arm")
            }
            // Any other ModelLoadError is fine (e.g. invalid config
            // path) - that means the family routed correctly.
        } catch {
            // Non-ModelLoadError (weight loader error) is the
            // expected path - the family routed to loadQwen which
            // then failed on missing weights.
        }
    }
}
