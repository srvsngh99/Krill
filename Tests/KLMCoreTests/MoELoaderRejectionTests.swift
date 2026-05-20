import XCTest
@testable import KLMCore

/// WS6 foundation: the loader recognizes MoE architectures AND
/// refuses to instantiate them (no silent fallback to a dense text
/// loader that would crash on router/expert keys). Tests pin both
/// halves of the contract for Mixtral, Qwen 3 MoE, and DeepSeek-V3.
final class MoELoaderRejectionTests: XCTestCase {

    private func writeConfig(_ json: [String: Any], dirSlug: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-moe-\(dirSlug)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: json)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    func testMixtralIsRejectedWithDocumentedError() throws {
        let dir = try writeConfig([
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "hidden_size": 4096,
            "vocab_size": 32000,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        ], dirSlug: "mixtral")
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertThrowsError(try loadModel(from: dir)) { error in
            guard let modelError = error as? ModelLoadError,
                  case .unsupportedArchitecture(let msg) = modelError else {
                XCTFail("Expected unsupportedArchitecture, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("MoE bridge"),
                "Error must redirect users to the MoE bridge runtime")
            XCTAssertTrue(msg.lowercased().contains("mixture-of-experts"),
                "Error must name the family for users debugging the rejection")
            XCTAssertTrue(msg.contains("/api/chat") || msg.contains("/v1/chat"),
                "Error must point at the chat-completion endpoints that handle MoE routing")
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

    func testQwen3MoENativeOptInReachesNativeArm() throws {
        // With KRILL_NATIVE_MOE=1 set, the loader takes the native
        // arm; with an empty config dir (no safetensors) the
        // failure is specifically `WeightLoadError.noSafetensorsFiles`,
        // which proves the family routed past the bridge rejection
        // and into the native loader. Asserting the SPECIFIC error
        // type is what discriminates "reached native arm and failed
        // there" from "got an unrelated runtime trap that the test
        // would have swallowed".
        let dir = try writeQwen3MoEConfig(dirSlug: "qwen3moe-optin")
        defer { try? FileManager.default.removeItem(at: dir) }

        try withEnv("KRILL_NATIVE_MOE", "1") {
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
                        + "unsupportedArchitecture when KRILL_NATIVE_MOE=1; "
                        + "got: \(msg)")
                } else {
                    XCTFail("Unexpected ModelLoadError: \(error)")
                }
            }
        }
    }

    func testQwen3MoEDefaultRoutesToBridge() throws {
        // Without KRILL_NATIVE_MOE set, the native arm refuses
        // and emits the documented redirect. Pins the "default
        // is bridge" contract so the env-gate cannot silently
        // flip without updating this test.
        let dir = try writeQwen3MoEConfig(dirSlug: "qwen3moe-default")
        defer { try? FileManager.default.removeItem(at: dir) }

        try withEnv("KRILL_NATIVE_MOE", nil) {
            XCTAssertThrowsError(try loadModel(from: dir)) { error in
                guard let modelError = error as? ModelLoadError,
                      case .unsupportedArchitecture(let msg) = modelError else {
                    XCTFail("Expected unsupportedArchitecture, got \(error)")
                    return
                }
                XCTAssertTrue(msg.contains("KRILL_NATIVE_MOE"),
                    "Default Qwen 3 MoE rejection must name the opt-in "
                    + "env var so users know how to enable native")
                XCTAssertTrue(msg.contains("MoE bridge") || msg.contains("MoEEngine"),
                    "Default rejection must redirect to the bridge")
            }
        }
    }

    func testMixtralStillRoutesToBridge() throws {
        // Mixtral (and other non-Qwen3 MoE families) keeps the
        // bridge fallback until its native port lands. This pins
        // the contract so the WS6 native PR cannot silently drop
        // the bridge rejection for unmigrated MoE families.
        let dir = try writeConfig([
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "hidden_size": 4096,
            "vocab_size": 32000,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        ], dirSlug: "mixtral-still-bridge")
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertThrowsError(try loadModel(from: dir)) { error in
            guard let modelError = error as? ModelLoadError,
                  case .unsupportedArchitecture(let msg) = modelError else {
                XCTFail("Expected unsupportedArchitecture, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("MoE bridge"),
                "Mixtral must still route through the MoE bridge "
                + "(no native runtime yet)")
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
