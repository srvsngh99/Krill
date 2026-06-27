import XCTest
import MLX
import MLXNN
@testable import KrillCore

/// End-to-end prefill logit-parity for the native qwen3_5 text decoder vs
/// mlx-lm. Exercises BOTH layer classes: the tiny 4-layer fixture has linear
/// (GatedDeltaNet) layers 0-2 and a full-attention layer 3. mRoPE is configured
/// as standard RoPE (text-only path; image mRoPE deferred). Fixture produced by
/// the python oracle into /tmp/q35-fwd. Both runtimes are MLX, so parity is tight.
final class Qwen35ForwardParityTests: XCTestCase {
    func testQwen35PrefillMatchesMLXLM() throws {
        let base = "/tmp/q35-fwd"
        guard FileManager.default.fileExists(atPath: base + "/io.safetensors") else {
            throw XCTSkip("forward fixture missing — run the python oracle first")
        }
        let cfgData = try Data(contentsOf: URL(fileURLWithPath: base + "/config.json"))
        let config = try JSONDecoder().decode(Qwen35Config.self, from: cfgData)
        let model = Qwen35ForCausalLM(config)

        let weights = try MLX.loadArrays(url: URL(fileURLWithPath: base + "/weights.safetensors"))
        let nested = ModuleParameters.unflattened(weights.map { ($0.key, $0.value) })
        try model.update(parameters: nested, verify: [.all])
        eval(model)

        let io = try MLX.loadArrays(url: URL(fileURLWithPath: base + "/io.safetensors"))
        let tokens = io["tokens"]!.asType(.int32)
        let refLogits = io["logits"]!
        let L = tokens.dim(1)

        let logits = model(tokens)
        eval(logits)

        let got = logits[0, L - 1, 0...].asType(.float32).asArray(Float.self)
        let ref = refLogits[0, L - 1, 0...].asType(.float32).asArray(Float.self)
        XCTAssertEqual(got.count, ref.count, "vocab size")

        var dot = 0.0, na = 0.0, nb = 0.0, maxAbs = 0.0
        var gi = 0, gMax = -Double.infinity, ri = 0, rMax = -Double.infinity
        for i in 0 ..< got.count {
            let x = Double(got[i]), y = Double(ref[i])
            dot += x * y; na += x * x; nb += y * y
            maxAbs = max(maxAbs, abs(x - y))
            if x > gMax { gMax = x; gi = i }
            if y > rMax { rMax = y; ri = i }
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        XCTAssertEqual(gi, ri, "argmax mismatch got \(gi) ref \(ri)")
        XCTAssertGreaterThan(cosine, 0.9999, "logit cosine \(cosine) too low")
        XCTAssertLessThan(maxAbs, 1e-2, "max-abs logit diff \(maxAbs) too large")
    }
}
