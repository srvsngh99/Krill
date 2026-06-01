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

    /// Scoped env-var setter so the test process restores the prior
    /// value (or unset) after the assertion block, even on failure.
    private func withEnv(_ key: String, _ value: String?, _ body: () throws -> Void) rethrows {
        let prior = ProcessInfo.processInfo.environment[key]
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
        defer {
            if let prior {
                setenv(key, prior, 1)
            } else {
                unsetenv(key)
            }
        }
        try body()
    }

    func testQwen3MoENativeIsTheDefault() throws {
        // Native is now the default: with KRILL_NATIVE_MOE unset (and
        // also when explicitly "1"), the loader takes the native arm.
        // With an empty config dir (no safetensors) the failure is
        // specifically `WeightLoadError.noSafetensorsFiles`, which proves
        // the family routed past the bridge rejection and into the native
        // loader. Asserting the SPECIFIC error type is what discriminates
        // "reached native arm and failed there" from "got an unrelated
        // runtime trap that the test would have swallowed".
        let dir = try writeQwen3MoEConfig(dirSlug: "qwen3moe-default-native")
        defer { try? FileManager.default.removeItem(at: dir) }

        func assertReachesNativeArm() throws {
            do {
                _ = try loadModel(from: dir)
                XCTFail("Expected WeightLoadError.noSafetensorsFiles "
                    + "for an empty config dir on the native arm")
            } catch let error as WeightLoadError {
                if case .noSafetensorsFiles = error {
                    // OK: reached the native arm, failed at weight load.
                } else {
                    XCTFail("Expected noSafetensorsFiles, got \(error)")
                }
            } catch let error as ModelLoadError {
                if case .unsupportedArchitecture(let msg) = error {
                    XCTFail("Native arm must not throw "
                        + "unsupportedArchitecture by default; got: \(msg)")
                } else {
                    XCTFail("Unexpected ModelLoadError: \(error)")
                }
            }
        }

        try withEnv("KRILL_NATIVE_MOE", nil) { try assertReachesNativeArm() }
        try withEnv("KRILL_NATIVE_MOE", "1") { try assertReachesNativeArm() }
    }

    func testQwen3MoEOptOutRoutesToBridge() throws {
        // KRILL_NATIVE_MOE=0 is the opt-out: the native arm refuses
        // and emits the documented redirect to the legacy bridge. Pins
        // the opt-out contract so the env-gate cannot silently drop it.
        let dir = try writeQwen3MoEConfig(dirSlug: "qwen3moe-optout")
        defer { try? FileManager.default.removeItem(at: dir) }

        try withEnv("KRILL_NATIVE_MOE", "0") {
            XCTAssertThrowsError(try loadModel(from: dir)) { error in
                guard let modelError = error as? ModelLoadError,
                      case .unsupportedArchitecture(let msg) = modelError else {
                    XCTFail("Expected unsupportedArchitecture, got \(error)")
                    return
                }
                XCTAssertTrue(msg.contains("KRILL_NATIVE_MOE"),
                    "Opt-out rejection must name the env var so users "
                    + "know how to restore the native default")
                XCTAssertTrue(msg.contains("MoE bridge") || msg.contains("MoEEngine"),
                    "Opt-out rejection must redirect to the bridge")
            }
        }
    }

    func testUnportedMoEFamiliesStillRouteToBridge() throws {
        // The not-yet-ported MoE families (Qwen2-MoE / OLMoE) keep the
        // bridge fallback until their native ports land. This pins the
        // contract so a native PR cannot silently drop the bridge
        // rejection for an unmigrated family.
        for (arch, modelType, slug) in [
            ("Qwen2MoeForCausalLM", "qwen2_moe", "qwen2moe-still-bridge"),
            ("OlmoeForCausalLM", "olmoe", "olmoe-still-bridge"),
        ] {
            let dir = try writeConfig([
                "architectures": [arch],
                "model_type": modelType,
                "hidden_size": 2048,
                "vocab_size": 151936,
                "num_experts": 60,
                "num_experts_per_tok": 4,
            ], dirSlug: slug)
            defer { try? FileManager.default.removeItem(at: dir) }

            XCTAssertThrowsError(try loadModel(from: dir)) { error in
                guard let modelError = error as? ModelLoadError,
                      case .unsupportedArchitecture(let msg) = modelError else {
                    XCTFail("Expected unsupportedArchitecture for \(modelType), got \(error)")
                    return
                }
                XCTAssertTrue(msg.contains("MoE bridge"),
                    "\(modelType) must still route through the MoE bridge (no native runtime yet)")
            }
        }
    }

    func testDeepSeekV3IsRejectedWithMoEMessage() throws {
        // DeepSeek V3 was migrated into the unified MoE
        // rejection in WS6 foundation; the WS6 runtime PR keeps
        // it on the same redirect path.
        let dir = try writeConfig([
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "vocab_size": 129280,
        ], dirSlug: "deepseekv3")
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertThrowsError(try loadModel(from: dir)) { error in
            guard let modelError = error as? ModelLoadError,
                  case .unsupportedArchitecture(let msg) = modelError else {
                XCTFail("Expected unsupportedArchitecture, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("MoE bridge"),
                "DeepSeek V3 should route through the unified MoE bridge redirect")
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
