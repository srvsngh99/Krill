import XCTest
import MLX
import MLXNN
@testable import KrillCore

/// Proves the incremental-decode path (conv-state + SSM recurrent state for
/// linear layers, KVCache for full layers) matches the cacheless full-sequence
/// forward — i.e. prefill-then-step generation is correct. Reuses the /tmp/q35-fwd
/// fixture (4-layer hybrid: GatedDeltaNet 0-2, full-attn 3).
final class Qwen35DecodeCacheTests: XCTestCase {
    func testIncrementalDecodeMatchesCachelessForward() throws {
        let base = "/tmp/q35-fwd"
        guard FileManager.default.fileExists(atPath: base + "/io.safetensors") else {
            throw XCTSkip("forward fixture missing — run the python oracle first")
        }
        let config = try JSONDecoder().decode(
            Qwen35Config.self, from: Data(contentsOf: URL(fileURLWithPath: base + "/config.json")))
        let model = Qwen35ForCausalLM(config)
        let weights = try MLX.loadArrays(url: URL(fileURLWithPath: base + "/weights.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights.map { ($0.key, $0.value) }),
                         verify: [.all])
        eval(model)

        let io = try MLX.loadArrays(url: URL(fileURLWithPath: base + "/io.safetensors"))
        let tokens = io["tokens"]!.asType(.int32)          // [1, L]
        let L = tokens.dim(1)

        // Reference: one cacheless forward over the whole sequence.
        let ref = model(tokens)                            // [1, L, V]
        eval(ref)

        // Incremental: prefill the first P tokens, then decode the rest one at a time.
        let caches = model.makeCaches()
        let P = L - 2
        _ = model(tokens[0..., 0 ..< P], caches: caches)
        for t in P ..< L {
            let step = model(tokens[0..., t ..< (t + 1)], caches: caches)   // [1,1,V]
            eval(step)
            let got = step[0, 0, 0...].asType(.float32).asArray(Float.self)
            let want = ref[0, t, 0...].asType(.float32).asArray(Float.self)
            var dot = 0.0, na = 0.0, nb = 0.0, maxAbs = 0.0
            var gi = 0, gMax = -Double.infinity, ri = 0, rMax = -Double.infinity
            for i in 0 ..< got.count {
                let x = Double(got[i]), y = Double(want[i])
                dot += x * y; na += x * x; nb += y * y; maxAbs = max(maxAbs, abs(x - y))
                if x > gMax { gMax = x; gi = i }
                if y > rMax { rMax = y; ri = i }
            }
            let cosine = dot / (na.squareRoot() * nb.squareRoot())
            XCTAssertEqual(gi, ri, "pos \(t): argmax mismatch \(gi) vs \(ri)")
            XCTAssertGreaterThan(cosine, 0.9999, "pos \(t): cosine \(cosine)")
            XCTAssertLessThan(maxAbs, 1e-2, "pos \(t): max-abs \(maxAbs)")
        }
    }
}
