import XCTest
import MLX
import KrillCache
@testable import KrillCore

/// Logit-parity check for the native Nanbeige 4.2 runtime
/// (`NanbeigeForCausalLM`, model_type "nanbeige") against the UPSTREAM PyTorch
/// reference (`modeling_nanbeige.py`). mlx-lm has no nanbeige port, so the
/// oracle is the authors' own code rather than a reimplementation of it.
///
/// Gated on `KRILL_NANBEIGE_PARITY_DIR`, a directory produced by
/// `tools/verify_nanbeige_parity.py` holding a tiny random Nanbeige checkpoint
/// (config.json + model.safetensors) plus `reference_logits.json`. Skipped when
/// the env var is unset - the fixture is generated on demand, not committed,
/// mirroring the other parity tests.
///
/// It pins the four things most likely to be wrong in a LOOPED-transformer port:
///
///  1. **The loop runs.** `num_loops=2` executes the stack twice over the same
///     weights. Running it once changes every logit.
///  2. **Cache sizing.** The engine allocates from `LoadedModel.numLayers`, which
///     for a looped model must be `num_loops * num_hidden_layers` (4 here, not
///     2) - one KV slot per layer PER LOOP, as the reference indexes them.
///  3. **Per-loop cache isolation on decode.** The incremental case prefills all
///     but the last token and then steps it; if loop 2 overwrote loop 1's keys
///     and values, the stepped logits diverge from the prefill's last-token
///     logits even though a single full forward still looks fine.
///  4. **head_dim decoupled from hidden_size.** The fixture sets
///     `4 heads * 24 dims != 64 hidden`, so a runtime deriving
///     `hidden_size / num_heads` mis-shapes q/o and fails to load at all.
final class NanbeigeParityTests: XCTestCase {

    private struct Reference: Decodable {
        let tokens: [Int]
        let vocab_size: Int
        let num_loops: Int
        let num_hidden_layers: Int
        let expected_cache_entries: Int
        let last_token_logits: [Float]
        let argmax: Int
        let step_logits: [Float]
        /// Absolute logit tolerance for this fixture. The synthetic fixture is
        /// fp32 on both sides and holds to 2e-3; a REAL bf16 checkpoint compared
        /// against an fp32 reference drifts more, because `num_loops` doubles the
        /// layer executions that accumulate rounding (44 for Nanbeige4.2-3B).
        /// Argmax agreement and cosine > 0.9999 are the invariants that matter;
        /// this bound only rejects gross divergence.
        let max_abs_tolerance: Double?
    }

    private func fixture() throws -> (URL, Reference) {
        guard let dirPath = ProcessInfo.processInfo.environment["KRILL_NANBEIGE_PARITY_DIR"] else {
            throw XCTSkip("Set KRILL_NANBEIGE_PARITY_DIR (see tools/verify_nanbeige_parity.py)")
        }
        let dir = URL(fileURLWithPath: dirPath)
        let refData = try Data(contentsOf: dir.appendingPathComponent("reference_logits.json"))
        return (dir, try JSONDecoder().decode(Reference.self, from: refData))
    }

    /// Compare two logit vectors: argmax agreement, cosine ~1, small max-abs.
    private func assertMatches(
        _ got: [Float], _ want: [Float], label: String, maxAbsTolerance: Double = 2e-3,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.count, want.count, "\(label): vocab size mismatch",
                       file: file, line: line)
        guard got.count == want.count, !got.isEmpty else { return }

        var gotMax = 0, wantMax = 0
        for i in 1 ..< got.count {
            if got[i] > got[gotMax] { gotMax = i }
            if want[i] > want[wantMax] { wantMax = i }
        }
        XCTAssertEqual(gotMax, wantMax,
            "\(label): native argmax \(gotMax) != reference argmax \(wantMax)",
            file: file, line: line)

        var dot: Double = 0, na: Double = 0, nb: Double = 0, maxAbs: Double = 0
        for i in 0 ..< got.count {
            let a = Double(got[i]), b = Double(want[i])
            dot += a * b; na += a * a; nb += b * b
            maxAbs = max(maxAbs, abs(a - b))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        XCTAssertGreaterThan(cosine, 0.9999,
            "\(label): cosine \(cosine) too low", file: file, line: line)
        // MLX and PyTorch order reductions differently, so allow a small
        // absolute drift; the caller widens this for a bf16 checkpoint.
        XCTAssertLessThan(maxAbs, maxAbsTolerance,
            "\(label): max abs logit diff \(maxAbs) too large", file: file, line: line)
    }

    /// Full-sequence prefill vs the reference's last-token logits.
    func testNativeNanbeigeMatchesReferenceLogits() throws {
        let (dir, ref) = try fixture()
        let loaded = try loadModel(from: dir)

        XCTAssertEqual(loaded.family, "nanbeige",
            "a nanbeige checkpoint must route to the nanbeige rule, not a dense fallback")
        XCTAssertEqual(loaded.vocabSize, ref.vocab_size)
        // The looped model needs one cache per layer PER LOOP.
        XCTAssertEqual(loaded.numLayers, ref.expected_cache_entries,
            "looped model must report num_loops * num_hidden_layers "
            + "(\(ref.num_loops) * \(ref.num_hidden_layers)) KV caches")

        let tokens = MLXArray(ref.tokens.map { Int32($0) }).reshaped([1, ref.tokens.count])
        let logits = loaded.forward(tokens, nil)          // [1, L, V]
        let last = logits[0, ref.tokens.count - 1, 0...]  // [V]
        eval(last)
        assertMatches(last.asArray(Float.self), ref.last_token_logits, label: "prefill",
                      maxAbsTolerance: ref.max_abs_tolerance ?? 2e-3)
    }

    /// Incremental decode against a populated cache. This is the case that
    /// actually exercises the per-loop cache indexing.
    func testNativeNanbeigeCachedDecodeMatchesReference() throws {
        let (dir, ref) = try fixture()
        let loaded = try loadModel(from: dir)

        let caches = makeKVCaches(numLayers: loaded.numLayers)
        let prefixIds = ref.tokens.dropLast().map { Int32($0) }
        let prefix = MLXArray(prefixIds).reshaped([1, prefixIds.count])
        let prefixLogits = loaded.forward(prefix, caches)
        eval(prefixLogits)

        // Every cache must have advanced by the prefix length - if loop 2 wrote
        // into loop 1's slots, some would be double-length and others empty.
        for (i, c) in caches.enumerated() {
            XCTAssertEqual(c.sequenceLength, prefixIds.count,
                "cache \(i) holds \(c.sequenceLength) tokens, expected \(prefixIds.count)")
        }

        let stepId = MLXArray([Int32(ref.tokens[ref.tokens.count - 1])]).reshaped([1, 1])
        let stepLogits = loaded.forward(stepId, caches)   // [1, 1, V]
        let step = stepLogits[0, 0, 0...]
        eval(step)
        assertMatches(step.asArray(Float.self), ref.step_logits, label: "cached decode",
                      maxAbsTolerance: ref.max_abs_tolerance ?? 2e-3)
    }

    /// A checkpoint enabling an upstream research feature with no native runtime
    /// must be REFUSED at load, not silently run with the flag ignored (which
    /// would emit confidently wrong logits).
    func testUnsupportedNanbeigeFeatureIsRejected() throws {
        let json = """
        {
          "architectures": ["NanbeigeForCausalLM"],
          "model_type": "nanbeige",
          "hidden_size": 64, "intermediate_size": 128,
          "num_hidden_layers": 2, "num_attention_heads": 4,
          "num_key_value_heads": 2, "head_dim": 24, "vocab_size": 128,
          "num_loops": 2, "enable_hyper_connection": true
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(NanbeigeConfig.self, from: json)
        XCTAssertThrowsError(try config.validate()) { error in
            XCTAssertTrue("\(error)".contains("hyper_connection"),
                "error should name the unsupported feature, got: \(error)")
        }
    }

    /// `head_dim` must come from the config, not from hidden_size / num_heads.
    func testHeadDimIsReadFromConfig() throws {
        let json = """
        {
          "architectures": ["NanbeigeForCausalLM"],
          "model_type": "nanbeige",
          "hidden_size": 3072, "intermediate_size": 10752,
          "num_hidden_layers": 22, "num_attention_heads": 48,
          "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 166144,
          "num_loops": 2
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(NanbeigeConfig.self, from: json)
        try config.validate()
        XCTAssertEqual(config.headDim, 128, "head_dim must be read, not derived (would be 64)")
        XCTAssertEqual(config.blockConfig.headDim, 128,
            "the block config handed to Attention must carry the explicit head_dim")
        XCTAssertEqual(NanbeigeForCausalLM(config).cacheCount, 44,
            "22 layers x 2 loops = 44 KV cache slots")
    }
}
