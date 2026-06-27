import XCTest
import MLX
import KrillTokenizer
import KrillCache
@testable import KrillCore

/// Real-checkpoint regression guards for the native qwen3_5 (Ornith-9B) runtime.
/// All gate on the int4 checkpoint being present on the Trench SSD (skip in CI /
/// on machines without it). They pin three things that the synthetic-fixture
/// parity tests cannot: that the int4 LOAD + forward matches mlx_vlm, that
/// incremental SSM-cached decode matches a cacheless reference on the real
/// 32-layer model, and that the chat-template tokenization matches HF.
final class Qwen35RealCheckpointTests: XCTestCase {
    private let path = "/Volumes/Trench/caches/ornith/ornith-9b-int4"

    /// Recorded mlx_vlm reference argmax for a fixed token sequence.
    func testRealPrefillArgmaxVsMLXVLM() throws {
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Ornith int4 checkpoint not present")
        }
        let model = try loadModel(from: URL(fileURLWithPath: path))
        let ids = [248045, 74455, 198, 9707, 11, 1246, 525, 498, 3351, 30]
        let tokens = MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
        let logits = model.forward(tokens, nil)
        MLX.eval(logits)
        // mlx_vlm: pos0=25, pos4=264, pos9=198
        for (pos, refArg) in [(0, 25), (4, 264), (9, 198)] {
            let am = Int(argMax(logits[0, pos]).item(Int32.self))
            XCTAssertEqual(am, refArg, "prefill argmax mismatch at pos \(pos)")
        }
    }

    /// Incremental SSM-cached decode must match a cacheless reference on the real
    /// model: prefill, then 5 single-token steps == cacheless forward over the
    /// growing prefix. Guards the GatedDeltaNet conv/SSM state threading.
    func testRealCachedDecodeMatchesCacheless() throws {
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Ornith int4 checkpoint not present")
        }
        let model = try loadModel(from: URL(fileURLWithPath: path))
        var seq = [248045, 846, 198, 33963, 2250, 10104, 364, 18839, 10419, 1345,
                   20316, 13, 248046, 198, 248045, 74455, 198, 248068, 198]
        let caches = makeKVCaches(spec: model.cacheSpec, numLayers: model.numLayers)
        let pre = model.forward(MLXArray(seq.map { Int32($0) }).reshaped(1, seq.count), caches)
        MLX.eval(pre)
        var next = Int(argMax(pre[0, seq.count - 1]).item(Int32.self))
        for _ in 0 ..< 5 {
            seq.append(next)
            let step = model.forward(MLXArray([Int32(next)]).reshaped(1, 1), caches)
            MLX.eval(step)
            let cachedNext = Int(argMax(step[0, 0]).item(Int32.self))
            let full = model.forward(MLXArray(seq.map { Int32($0) }).reshaped(1, seq.count), nil)
            MLX.eval(full)
            let cachelessNext = Int(argMax(full[0, seq.count - 1]).item(Int32.self))
            XCTAssertEqual(cachedNext, cachelessNext, "cached decode diverged from cacheless")
            next = cachelessNext
        }
    }

    /// Krill's chat-template tokenization must match HF `apply_chat_template`
    /// (add_generation_prompt) on the custom 248k-vocab Ornith tokenizer,
    /// including the auto-appended `<think>` priming.
    func testKrillTokenizationMatchesHF() async throws {
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Ornith int4 checkpoint not present")
        }
        let tok = try await KrillTokenizer(from: URL(fileURLWithPath: path))
        let msgs = [["role": "user", "content": "Give three tips for staying focused while studying."]]
        let hf = [248045, 846, 198, 33963, 2250, 10104, 364, 18839, 10419, 1345,
                  20316, 13, 248046, 198, 248045, 74455, 198, 248068, 198]
        XCTAssertEqual(tok.applyChatTemplateTokens(messages: msgs), hf)
    }
}
