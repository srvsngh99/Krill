import XCTest
import Foundation
import MLX
import MLXNN
@testable import KrillCore

/// Parity gate for the native MoonViT vision tower + `mlp1` connector against
/// NVIDIA LocateAnything-3B's custom PyTorch reference (`modeling_vit.py`
/// `MoonVitPretrainedModel`). Fixture (tiny synthetic config + random weights +
/// a random patch batch and the oracle's outputs) is produced by
/// `tools/verify_locateanything_parity.py` and committed under
/// `Fixtures/locateanything`. Green = the Conv2d patch embed (run as a matmul),
/// the bicubically-interpolated learnable position embedding, the 2D complex
/// rotary full-attention blocks, the 2×2 patch merge, and the connector all
/// match the oracle.
final class LocateAnythingVisionParityTests: XCTestCase {
    private var fixtureDir: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/locateanything")
    }

    func testVisionPathMatchesReference() throws {
        let dir = fixtureDir
        guard FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("weights.safetensors").path) else {
            throw XCTSkip("locateanything fixture missing — run tools/verify_locateanything_parity.py")
        }

        // Config + shapes from meta.json.
        let metaData = try Data(contentsOf: dir.appendingPathComponent("meta.json"))
        let meta = try JSONSerialization.jsonObject(with: metaData) as! [String: Any]
        let configData = try JSONSerialization.data(withJSONObject: meta["config"]!)
        let config = try JSONDecoder().decode(MoonViTVisionConfig.self, from: configData)
        let grid = (meta["grid_hws"] as! [[Int]]).map { (h: $0[0], w: $0[1]) }
        let vitHidden = meta["vit_hidden"] as! Int
        let llmHidden = meta["llm_hidden"] as! Int

        // Build models.
        let tower = MoonViTVisionModel(config)
        let connector = LocateAnythingConnector(vitHidden: vitHidden, llmHidden: llmHidden)

        // Load + sanitize weights, split into vision_model.* / mlp1.* subtrees.
        let raw = try MLX.loadArrays(url: dir.appendingPathComponent("weights.safetensors"))
        let weights = MoonViTVisionModel.sanitize(raw)
        var visionW: [(String, MLXArray)] = []
        var connW: [(String, MLXArray)] = []
        for (k, v) in weights {
            if k.hasPrefix("vision_model.") {
                visionW.append((String(k.dropFirst("vision_model.".count)), v))
            } else if k.hasPrefix("mlp1.") {
                connW.append((String(k.dropFirst("mlp1.".count)), v))
            }
        }
        connW = LocateAnythingConnector.remapKeys(connW)
        try tower.update(parameters: ModuleParameters.unflattened(visionW), verify: [.all])
        try connector.update(parameters: ModuleParameters.unflattened(connW), verify: [.all])

        // Run + compare both stages.
        let io = try MLX.loadArrays(url: dir.appendingPathComponent("io.safetensors"))
        let pixelValues = io["pixel_values"]!
        let expectedVit = io["vit_output"]!
        let expectedConn = io["connector_output"]!

        let vitOut = tower(pixelValues, grids: grid)
        XCTAssertEqual(vitOut.shape, expectedVit.shape, "vit output shape")
        compare(vitOut, expectedVit, "moonvit_tower")

        let connOut = connector(vitOut)
        XCTAssertEqual(connOut.shape, expectedConn.shape, "connector output shape")
        compare(connOut, expectedConn, "connector")
    }

    private func compare(_ got: MLXArray, _ ref: MLXArray, _ name: String) {
        let gv = got.asType(.float32).flattened().asArray(Float.self)
        let rv = ref.asType(.float32).flattened().asArray(Float.self)
        XCTAssertEqual(gv.count, rv.count, "\(name) element count")
        var dot = 0.0, na = 0.0, nb = 0.0, maxAbs = 0.0
        for i in 0 ..< gv.count {
            let x = Double(gv[i]), y = Double(rv[i])
            dot += x * y; na += x * x; nb += y * y
            maxAbs = max(maxAbs, abs(x - y))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        XCTAssertGreaterThan(cosine, 0.9999, "\(name) cosine \(cosine) too low")
        XCTAssertLessThan(maxAbs, 1e-3, "\(name) max-abs \(maxAbs) too large")
    }
}
