import Foundation
import XCTest
import KrillRuntime
import KrillTokenizer
@testable import KrillCore

final class DMJEPADecisionModelTests: XCTestCase {
    func testPromptRendering() {
        XCTAssertEqual(DMJEPAPrompt.state(" ", instructions: " Pick an intent. "),
                       "<|state|>\nnone\n<|instructions|>\nPick an intent.")
        XCTAssertEqual(DMJEPAPrompt.state(" line 1\nline 2 ", instructions: ""),
                       "<|state|>\nline 1\nline 2\n<|instructions|>\n")
        for label in ["option_10", "k_card", "10", "²"] {
            XCTAssertEqual(DMJEPAPrompt.option(label: label, criterion: " card_arrival "), "<|option|> card_arrival")
        }
        XCTAssertEqual(DMJEPAPrompt.option(label: " card ", criterion: " A lost card "), "<|option|> card: A lost card")
        XCTAssertEqual(DMJEPAPrompt.option(label: "card"), "<|option|> card")
        XCTAssertEqual(DMJEPAPrompt.option(label: "\u{1c}card\u{200b}"), "<|option|> card\u{200b}")
    }

    private struct Reference: Decodable {
        let state: String
        let instructions: String
        let labels: [String]
        let options: [String]
        let choice: Int
        let probabilities: [Float]
    }

    func testReferenceParity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["KRILL_DMJEPA_MODEL_PATH"], let dataPath = env["KRILL_DMJEPA_PARITY_ROWS"] else {
            throw XCTSkip("Run scripts/dmjepa_parity.py with the model, tokenizer, inputs and results paths")
        }
        try MLXMetalRuntime.validateForNativeInference()
        let directory = URL(fileURLWithPath: path)
        XCTAssertTrue(DecisionModel.isBundle(directory))
        let model = try await DecisionModel.load(directory: directory)
        XCTAssertTrue(model.isDMJEPA)
        let tokenizer = try await KrillTokenizer(from: directory)
        XCTAssertEqual(tokenizer.encode(DMJEPAPrompt.state("", instructions: "Pick an intent.")),
                       [50281, 29, 93, 3409, 49651, 187, 15422, 187, 29, 93, 6839, 6477, 49651, 187, 36942, 271, 6860, 15, 50282])
        XCTAssertEqual(tokenizer.encode(DMJEPAPrompt.option(label: "option_0", criterion: "card_arrival")),
                       [50281, 29, 93, 7872, 49651, 3120, 64, 274, 49182, 50282])
        let decoder = JSONDecoder()
        let rows = try String(contentsOfFile: dataPath, encoding: .utf8).split(separator: "\n").map {
            try decoder.decode(Reference.self, from: Data($0.utf8))
        }
        XCTAssertEqual(rows.count, 200)
        func probabilities(_ row: Reference) throws -> [Float] {
            let result = try model.candidateProbabilities(text: row.state, questions: [row.instructions],
                                                          options: row.options, optionLabels: row.labels)
            return try XCTUnwrap(result.probabilities.first)
        }
        _ = try probabilities(XCTUnwrap(rows.first))
        var matched = 0
        var worst: Float = 0
        var milliseconds: [Double] = []
        for (index, row) in rows.enumerated() {
            let start = DispatchTime.now().uptimeNanoseconds
            let p = try probabilities(row)
            milliseconds.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            XCTAssertEqual(p.count, row.options.count)
            XCTAssertEqual(p.count, row.probabilities.count)
            XCTAssertTrue(p.allSatisfy { $0.isFinite && (0 ... 1).contains($0) })
            XCTAssertEqual(p.reduce(0, +), 1, accuracy: 1e-5)
            let best = p.indices.dropFirst().reduce(0) { p[$1] > p[$0] ? $1 : $0 }
            if best == row.choice { matched += 1 } else { print("DM-JEPA choice differs on row \(index + 1)") }
            for (actual, expected) in zip(p, row.probabilities) { worst = max(worst, abs(actual - expected)) }
        }
        milliseconds.sort()
        let middle = milliseconds.count / 2
        let median = (milliseconds[middle - 1] + milliseconds[middle]) / 2
        print("DM-JEPA parity: \(matched)/\(rows.count) choices agree, max absolute probability difference \(worst), median \(median) ms/question")
        XCTAssertGreaterThanOrEqual(matched, 198)
    }
}
