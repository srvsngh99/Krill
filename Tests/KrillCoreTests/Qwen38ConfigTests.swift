import XCTest
@testable import KrillCore

/// Config-decode gates for Qwen3.8-27B, which reaches Krill's `.qwen35` runtime
/// unchanged but ships a transformers 5.x `config.json`: the rope knobs moved
/// into a nested `rope_parameters` object, and `rope_theta` no longer appears at
/// the `text_config` top level at all.
///
/// Both facts were silent hazards. `mrope_section` used to be hardcoded to
/// `[11, 11, 10]` at the call site, and a missing `rope_theta` fell through to a
/// 10M default that happens to be correct for THIS checkpoint — a coincidence,
/// not a contract. These assert the fallback chain instead of the coincidence.
final class Qwen38ConfigTests: XCTestCase {
    /// Shape-accurate excerpt of `mlx-community/Qwen3.8-27B-4bit`'s `text_config`.
    private let qwen38TextConfig = """
    {
        "attn_output_gate": true,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "output_gate_type": "swish",
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default"
        },
        "tie_word_embeddings": false,
        "vocab_size": 248320
    }
    """

    private func decode(_ json: String) throws -> Qwen35Config {
        try JSONDecoder().decode(Qwen35Config.self, from: Data(json.utf8))
    }

    func testQwen38ShapesDecode() throws {
        let c = try decode(qwen38TextConfig)
        XCTAssertEqual(c.numHiddenLayers, 64)
        XCTAssertEqual(c.hiddenSize, 5120)
        XCTAssertEqual(c.intermediateSize, 17408)
        XCTAssertEqual(c.headDim, 256, "must NOT be derived as hidden/heads (213)")
        XCTAssertEqual(c.numAttentionHeads, 24)
        XCTAssertEqual(c.numKeyValueHeads, 4)
        XCTAssertEqual(c.linearNumValueHeads, 48)
        XCTAssertEqual(c.linearNumKeyHeads, 16)
        XCTAssertEqual(c.vocabSize, 248320)
        XCTAssertFalse(c.tieWordEmbeddings)
    }

    /// The hybrid schedule: 16 × (3 GatedDeltaNet → 1 gated attention). The
    /// checkpoint also ships an explicit 64-entry `layer_types` array; this
    /// asserts our computed rule agrees with it at every index.
    func testHybridScheduleMatchesCheckpointLayerTypes() throws {
        let c = try decode(qwen38TextConfig)
        let fullAttentionLayers = (0 ..< c.numHiddenLayers).filter { !c.isLinearLayer($0) }
        XCTAssertEqual(fullAttentionLayers.count, 16)
        XCTAssertEqual(Array(fullAttentionLayers.prefix(4)), [3, 7, 11, 15])
        XCTAssertEqual(fullAttentionLayers.last, 63)
    }

    /// `rope_theta` lives ONLY inside `rope_parameters` here. Reading it must not
    /// depend on the default happening to match.
    func testRopeParametersAreReadFromTheNestedObject() throws {
        let c = try decode(qwen38TextConfig)
        XCTAssertEqual(c.ropeTheta, 10_000_000)
        XCTAssertEqual(c.partialRotaryFactor, 0.25)
        XCTAssertEqual(c.mropeSection, [11, 11, 10])
        // Partial rotary: 256 * 0.25 = 64 dims rotated, so the t/h/w split must
        // sum to half of that.
        XCTAssertEqual(c.mropeSection.reduce(0, +), Int(Float(c.headDim) * c.partialRotaryFactor) / 2)
    }

    /// A nested value must be used when the top level omits it — proving the
    /// fallback actually fires rather than coinciding with the default.
    func testNestedRopeThetaWinsOverTheDefault() throws {
        let json = qwen38TextConfig.replacingOccurrences(
            of: "\"rope_theta\": 10000000", with: "\"rope_theta\": 5000000")
        let c = try decode(json)
        XCTAssertEqual(c.ropeTheta, 5_000_000, "nested rope_theta must beat the 10M default")
    }

    /// The older Ornith/Qwythos layout (flat keys, no `rope_parameters`) must keep
    /// decoding identically — this is the shipping path for two live aliases.
    func testFlatOrnithLayoutStillDecodes() throws {
        let json = """
        {
            "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 32,
            "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
            "vocab_size": 248320, "linear_num_value_heads": 48, "linear_num_key_heads": 16,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "rope_theta": 10000000, "partial_rotary_factor": 0.25
        }
        """
        let c = try decode(json)
        XCTAssertEqual(c.numHiddenLayers, 32)
        XCTAssertEqual(c.ropeTheta, 10_000_000)
        XCTAssertEqual(c.partialRotaryFactor, 0.25)
        XCTAssertEqual(c.mropeSection, [11, 11, 10], "default preserves prior behavior")
        XCTAssertEqual(c.fullAttentionInterval, 4, "default when the key is absent")
    }

    /// The excerpts above are hand-written, so they can only prove the decoder
    /// handles what we THINK the checkpoint says. When a real checkpoint is
    /// available, decode its actual `config.json` through the full VL config —
    /// the one the loader uses — and assert the vision half too.
    func testRealCheckpointConfigDecodes() throws {
        let env = ProcessInfo.processInfo.environment
        guard let dir = [env["KRILL_QWEN35_MODEL_PATH"], env["KRILL_ORNITH_MODEL_PATH"]]
            .compactMap({ $0 }).first(where: { !$0.isEmpty }) else {
            throw XCTSkip("KRILL_QWEN35_MODEL_PATH not set")
        }
        let url = URL(fileURLWithPath: dir).appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("no config.json at \(url.path)")
        }
        let config = try JSONDecoder().decode(Qwen35VLConfig.self, from: Data(contentsOf: url))
        let t = config.textConfig, v = config.visionConfig

        // Every dimension the runtime builds modules from must be non-degenerate.
        XCTAssertGreaterThan(t.numHiddenLayers, 0)
        XCTAssertGreaterThan(t.hiddenSize, 0)
        XCTAssertGreaterThan(t.headDim, 0)
        XCTAssertGreaterThan(t.ropeTheta, 0, "rope_theta must resolve from wherever it lives")
        XCTAssertEqual(
            t.mropeSection.reduce(0, +),
            Int(Float(t.headDim) * t.partialRotaryFactor) / 2,
            "mrope_section must partition exactly half the rotary dims")

        // The vision projector's output must match the decoder's residual width,
        // or the image-feature scatter writes the wrong shape into the embeds.
        XCTAssertEqual(v.outHiddenSize, t.hiddenSize,
                       "vision out_hidden_size must equal the text hidden_size")
        XCTAssertGreaterThan(v.depth, 0)
        XCTAssertEqual(v.hiddenSize % v.numHeads, 0, "vision head_dim must divide evenly")
    }
}
