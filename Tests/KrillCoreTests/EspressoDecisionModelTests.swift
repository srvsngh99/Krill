import XCTest
import MLX
import KrillRuntime
@testable import KrillCore

final class EspressoDecisionModelTests: XCTestCase {
    func testLocalAttentionAcrossBlockBoundaries() throws {
        guard MLXMetalRuntime.canInitializeMLXForTests else { throw XCTSkip("MLX Metal runtime is not available") }
        let length = 259
        let width = 128
        let q = MLXArray.zeros([1, 1, length, 64])
        let v = broadcast(MLXArray((0 ..< length).map(Float.init), [1, 1, length, 1]), to: q.shape)
        let output = EspressoAttentionWindow(length: length, width: width).attend(q, q, v, scale: 0.125)
        let values = output[0, 0, 0..., 0].asArray(Float.self)
        XCTAssertEqual(values.count, length)
        for (i, value) in values.enumerated() {
            let expected = Float(max(i - width, 0) + min(i + width, length - 1)) / 2
            XCTAssertEqual(value, expected, accuracy: 2e-4, "token \(i)")
        }
    }

    func testChoiceRuleAndOutOfScopeMapping() {
        let rule = EspressoDecisionRule(deltaNone: 0.25, bPass: -0.15, tau: 0.88, minOptions: 3)
        let none = EspressoDecisionModel.choiceKey("option_2", description: "Out of scope: no match")
        XCTAssertEqual(none, "none_of_these")
        XCTAssertEqual(EspressoDecisionModel.choiceKey("option_2", description: "out_of_scope"), "option_2")
        let scoped = rule.apply([0.46, 0.1, 0.44], keys: ["a", "b", none])
        XCTAssertEqual(scoped.best, 2)
        XCTAssertTrue(scoped.review)
        XCTAssertEqual(scoped.pNone, scoped.scores[2])
        XCTAssertEqual(scoped.scores.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertFalse(rule.apply([0.1, 0.9], keys: ["a", none]).review)
        XCTAssertEqual(rule.apply([0.51, 0.49], keys: ["PASS", "FAIL"]).best, 1)
    }

    func testReferenceParity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["KRILL_ESPRESSO_MODEL_PATH"], let dataPath = env["KRILL_ESPRESSO_PARITY_ROWS"] else {
            throw XCTSkip("Run scripts/espresso_parity.py with the model, inputs and results paths")
        }
        try MLXMetalRuntime.validateForNativeInference()
        let model = try await DecisionModel.load(directory: URL(fileURLWithPath: path))
        XCTAssertEqual(model.family, .espresso)
        try checkDecisionReferenceParity(model, name: "Espresso", dataPath: dataPath)
    }
}
