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

    private struct Reference: Decodable {
        let state: String
        let instructions: String
        let options: [String]
        let choice: Int
        let probabilities: [Float]
    }

    func testReferenceParity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["KRILL_AMBER_MODEL_PATH"], let dataPath = env["KRILL_AMBER_PARITY_ROWS"] else {
            throw XCTSkip("Run scripts/amber_parity.py with the model, inputs and results paths")
        }
        try MLXMetalRuntime.validateForNativeInference()
        let model = try await DecisionModel.load(directory: URL(fileURLWithPath: path))
        XCTAssertTrue(model.isAmber)
        let decoder = JSONDecoder()
        let rows = try String(contentsOfFile: dataPath, encoding: .utf8).split(separator: "\n").map {
            try decoder.decode(Reference.self, from: Data($0.utf8))
        }
        XCTAssertEqual(rows.count, 200)
        let warmup = try XCTUnwrap(rows.first)
        _ = try model.candidateProbabilities(text: warmup.state, questions: [warmup.instructions], options: warmup.options)
        var matched = 0
        var worst: Float = 0
        var milliseconds: [Double] = []
        for (index, row) in rows.enumerated() {
            let start = DispatchTime.now().uptimeNanoseconds
            let result = try model.candidateProbabilities(text: row.state, questions: [row.instructions], options: row.options)
            milliseconds.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            let p = try XCTUnwrap(result.probabilities.first)
            XCTAssertEqual(p.count, row.options.count)
            XCTAssertEqual(p.count, row.probabilities.count)
            XCTAssertTrue(p.allSatisfy { $0.isFinite && (0 ... 1).contains($0) })
            XCTAssertEqual(p.reduce(0, +), 1, accuracy: 1e-5)
            let best = p.indices.reduce(0) { p[$1] > p[$0] ? $1 : $0 }
            if best == row.choice { matched += 1 } else { print("Amber choice differs on row \(index + 1)") }
            for (actual, expected) in zip(p, row.probabilities) { worst = max(worst, abs(actual - expected)) }
        }
        milliseconds.sort()
        let middle = milliseconds.count / 2
        let median = (milliseconds[middle - 1] + milliseconds[middle]) / 2
        print("Amber parity: \(matched)/\(rows.count) choices agree, max absolute probability difference \(worst), median \(median) ms/question")
        XCTAssertGreaterThanOrEqual(matched, 198)
    }
}
