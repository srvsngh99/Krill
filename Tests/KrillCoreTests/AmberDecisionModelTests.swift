import XCTest
import MLX
import KrillRuntime
@testable import KrillCore

final class AmberDecisionModelTests: XCTestCase {
    func testSequenceLayoutAndLimit() throws {
        let sequence = try AmberSequence(instructions: [4, 5], state: [6], options: [[7, 8], [], [9]],
                                         cls: 1, sep: 2, mask: 3, limit: 13)
        XCTAssertEqual(sequence.ids, [1, 4, 5, 2, 6, 2, 3, 7, 8, 3, 3, 9, 2])
        XCTAssertEqual(sequence.markers, [6, 9, 10])
        XCTAssertEqual(sequence.starts, [7, 10, 11])
        XCTAssertEqual(sequence.ends, [9, 10, 12])
        XCTAssertThrowsError(try AmberSequence(instructions: [4, 5], state: [6], options: [[7, 8], [], [9]],
                                               cls: 1, sep: 2, mask: 3, limit: 12))
    }

    func testSequenceFeatures() throws {
        guard MLXMetalRuntime.canInitializeMLXForTests else { throw XCTSkip("MLX Metal runtime is not available") }
        let sequence = try AmberSequence(instructions: [], state: [], options: [[4, 5], []],
                                         cls: 1, sep: 2, mask: 3, limit: 8)
        let states = MLXArray((0 ..< 16).map(Float.init), [8, 2])
        XCTAssertEqual(sequence.features(states).asArray(Float.self), [6, 7, 9, 10, 0, 1, 12, 13, 0, 0, 0, 1])
    }

    func testReferenceParity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["KRILL_AMBER_MODEL_PATH"], let dataPath = env["KRILL_AMBER_PARITY_ROWS"] else {
            throw XCTSkip("Run scripts/amber_parity.py with the model, inputs and results paths")
        }
        try MLXMetalRuntime.validateForNativeInference()
        let model = try await DecisionModel.load(directory: URL(fileURLWithPath: path))
        XCTAssertEqual(model.family, .amber)
        try checkDecisionReferenceParity(model, name: "Amber", dataPath: dataPath)
    }
}
