import XCTest
import Foundation
import MLX
import KLMCache
@testable import KLMCore

/// Root-cause harness for the gemma-4-e2b long-context degeneration: feed the
/// EXACT mlx-vlm-tokenized ids (BOS + gemma turns, 1090 tok) and compare both
/// the prefill last-token argmax and a 32-step greedy continuation against the
/// mlx-vlm oracle (`tools/.../make_ref.py` -> reference_logits.json).
final class Gemma4LongCtxParityTests: XCTestCase {
    struct Reference: Decodable {
        let tokens: [Int]
        let vocab_size: Int
        let last_token_logits: [Float]
        let argmax: Int
        let greedy_ids: [Int]
    }

    func testLongCtxParity() throws {
        guard let dirPath = ProcessInfo.processInfo.environment["KLM_GEMMA4_PARITY_DIR"] else {
            throw XCTSkip("Set KLM_GEMMA4_PARITY_DIR (see tools/verify_gemma4_longctx_parity.py)")
        }
        let dir = URL(fileURLWithPath: dirPath)
        let ref = try JSONDecoder().decode(
            Reference.self, from: Data(contentsOf: dir.appendingPathComponent("reference_logits.json")))
        let loaded = try loadModel(from: dir)

        // TEACHER-FORCED decode through the KV cache (the real generation path,
        // which engages KV sharing). Feeding the oracle's own tokens keeps every
        // step on the identical prefix, so Krill's argmax at step i must equal
        // the oracle's greedy_ids[i]. This is the gate that the KV-shared decode
        // RoPE-offset bug fails: rotating shared-layer Q at offset 0 instead of
        // the donor's true position degrades long-context decode into garbage.
        let caches: [KVCache] = (0 ..< loaded.numLayers).map { _ in KVCache() }
        let n = ref.tokens.count
        var step = loaded.forward(MLXArray(ref.tokens.map { Int32($0) }).reshaped([1, n]), caches)
        var agree = 0, firstDiv = -1
        var krillTF: [Int] = []
        for i in 0 ..< ref.greedy_ids.count {
            let lg = step[0, step.dim(1) - 1, 0...]
            eval(lg)
            let arr = lg.asArray(Float.self)
            var mi = 0
            for j in 1 ..< arr.count where arr[j] > arr[mi] { mi = j }
            krillTF.append(mi)
            if mi == ref.greedy_ids[i] { agree += 1 } else if firstDiv < 0 { firstDiv = i }
            step = loaded.forward(MLXArray([Int32(ref.greedy_ids[i])]).reshaped([1, 1]), caches)
        }
        print("KRILL  TF: \(krillTF.prefix(20))")
        print("ORACLE TF: \(ref.greedy_ids.prefix(20))")
        print("TEACHER-FORCED AGREEMENT: \(agree)/\(ref.greedy_ids.count)  first-disagreement=\(firstDiv)")

        // The prefill argmax (first teacher-forced step) MUST match the oracle.
        XCTAssertEqual(krillTF.first, ref.argmax,
            "prefill argmax \(String(describing: krillTF.first)) != oracle \(ref.argmax)")
        // Teacher-forced agreement: tolerate a few 4-bit GEMM tie-flips between
        // MLX-Swift and MLX-Python but require the long-context decode to track
        // the oracle (the bug drove this near zero, garbage from step ~2).
        XCTAssertGreaterThanOrEqual(agree, (ref.greedy_ids.count * 7) / 8,
            "teacher-forced agreement \(agree)/\(ref.greedy_ids.count) too low - long-context decode diverged")
    }
}
