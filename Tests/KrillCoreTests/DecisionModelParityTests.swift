import XCTest
import MLX
import KrillRuntime
import KrillTokenizer
@testable import KrillCore

/// Golden parity of the native decision model against the Python reference. Needs a bundle:
/// `KRILL_DECISION_MODEL_PATH=/path/to/bundle swift test --filter DecisionModelParityTests`
final class DecisionModelParityTests: XCTestCase {
    struct Case: Decodable {
        let id: String
        let kind: String
        let text: String
        let question: String?
        let options: [String]?
        let allowNone: Bool?
        let stateIds: [Int]
        let scores: [Float]?
        let pNone: Float?
        let index: Int?
        let pYes: Float?
        let answer: String?

        enum CodingKeys: String, CodingKey {
            case id, kind, text, question, options, scores, index, answer
            case allowNone = "allow_none", stateIds = "state_ids", pNone = "p_none", pYes = "p_yes"
        }
    }

    struct Golden: Decodable {
        let cases: [Case]
        let batch: [String]
    }

    private let tolerance: Float = 1e-3

    private func fixture() async throws -> (model: DecisionModel, dir: URL, golden: Golden) {
        guard let path = ProcessInfo.processInfo.environment["KRILL_DECISION_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KRILL_DECISION_MODEL_PATH not set; skipping decision-model parity test")
        }
        guard MLXMetalRuntime.canInitializeMLXForTests else {
            throw XCTSkip("MLX Metal runtime is not available in this test process")
        }
        let file = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/decision/golden.json")
        let golden = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: file))
        let dir = URL(fileURLWithPath: path)
        return (try await DecisionModel.load(directory: dir), dir, golden)
    }

    func testTokenizerMatchesReference() async throws {
        let (_, dir, golden) = try await fixture()
        let tokenizer = try await KrillTokenizer(from: dir)
        for c in golden.cases {
            let state = c.question.flatMap { c.kind == "choice" ? "\(c.text)\n\n\($0)" : nil } ?? c.text
            XCTAssertEqual(tokenizer.encode(state), c.stateIds, "tokens differ for \(c.id)")
        }
    }

    func testDecisionsAndProbabilitiesMatchReference() async throws {
        let (model, _, golden) = try await fixture()
        var worst: Float = 0
        var matched = 0
        for c in golden.cases {
            if c.kind == "yesno" {
                let r = try model.yesNo(texts: [c.text], question: c.question!)[0]
                XCTAssertEqual(r.answer, c.answer, c.id)
                worst = max(worst, abs(r.pYes - c.pYes!))
                if r.answer == c.answer { matched += 1 }
                continue
            }
            let r = try model.choose([c.text], options: c.options!, question: c.question,
                                     allowNone: c.allowNone!)[0]
            XCTAssertEqual(r.index, c.index, "decision differs for \(c.id)")
            if r.index == c.index { matched += 1 }
            XCTAssertEqual(r.scores.count, c.scores!.count, c.id)
            for (a, b) in zip(r.scores, c.scores!) { worst = max(worst, abs(a - b)) }
            worst = max(worst, abs(r.pNone - c.pNone!))
        }
        print("decision parity: \(matched)/\(golden.cases.count) decisions matched, worst probability difference \(worst)")
        XCTAssertLessThan(worst, tolerance)
    }

    func testBatchEqualsSingle() async throws {
        let (model, _, golden) = try await fixture()
        let rows = golden.cases.filter { golden.batch.contains($0.id) }
        let first = rows[0]
        let batched = try model.choose(rows.map(\.text), options: first.options!, question: first.question)
        var worst: Float = 0
        for (row, b) in zip(rows, batched) {
            let single = try model.choose([row.text], options: row.options!, question: row.question)[0]
            XCTAssertEqual(b.index, single.index, row.id)
            XCTAssertEqual(b.index, row.index, row.id)
            for (x, y) in zip(b.scores, single.scores) { worst = max(worst, abs(x - y)) }
            worst = max(worst, abs(b.pNone - single.pNone))
        }
        print("batch vs single: worst probability difference \(worst)")
        XCTAssertLessThan(worst, tolerance)
    }

    func testOverLimitInputsThrow() async throws {
        let (model, _, _) = try await fixture()
        let long = String(repeating: "word ", count: 3000)
        XCTAssertThrowsError(try model.choose([long], options: ["a", "b"]))
        XCTAssertThrowsError(try model.choose(["hi"], options: [String(repeating: "word ", count: 100)]))
    }
}
