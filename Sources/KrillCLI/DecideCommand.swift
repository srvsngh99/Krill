import ArgumentParser
import Foundation
import KrillCore
import KrillRegistry

/// Run a decision-model bundle: pick one option (or none), or answer yes/no.
struct DecideCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decide",
        abstract: "Pick one of several options (or none), or answer a yes/no question"
    )

    @Option(name: .long, help: "Path to a decision-model bundle directory, or the name of an installed one")
    var model: String

    @Option(name: .long, help: "Text to decide about")
    var text: String?

    @Option(name: .long, help: "Question appended to the text")
    var question: String?

    @Option(name: .long, help: "An option to choose from (repeat for each option)")
    var option: [String] = []

    @Flag(name: .customLong("no-none"), help: "Always pick an option, never \"none of these\"")
    var noNone = false

    @Option(name: .customLong("yes-no"), help: "Answer this yes/no question instead of choosing")
    var yesNo: String?

    @Option(name: .customLong("json-in"), help: "File (or - for stdin) of JSON lines: text, question, options, allow_none, yes_no")
    var jsonIn: String?

    func run() async throws {
        let dir = FileManager.default.fileExists(atPath: model)
            ? URL(fileURLWithPath: model) : Registry.configured().modelPath(model)
        guard DecisionModel.isBundle(dir) else {
            print("Error: '\(model)' is not a decision-model bundle directory.")
            throw ExitCode.failure
        }
        let decider = try await DecisionModel.load(directory: dir)
        do {
            if let jsonIn {
                let input = jsonIn == "-"
                    ? String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
                    : try String(contentsOfFile: jsonIn, encoding: .utf8)
                for (n, line) in input.split(whereSeparator: \.isNewline).enumerated() {
                    guard let call = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else {
                        throw DecisionError.invalid("--json-in line \(n + 1) is not a JSON object")
                    }
                    try emit(decider, text: call["text"] as? String, question: call["question"] as? String,
                             options: call["options"] as? [String] ?? [],
                             allowNone: call["allow_none"] as? Bool ?? true, yesNo: call["yes_no"] as? String)
                }
            } else {
                try emit(decider, text: text, question: question, options: option, allowNone: !noNone, yesNo: yesNo)
            }
        } catch let error as DecisionError {
            print("Error: \(error)")
            throw ExitCode.failure
        }
    }

    private func emit(_ decider: DecisionModel, text: String?, question: String?, options: [String],
                      allowNone: Bool, yesNo: String?) throws {
        guard let text else { throw DecisionError.invalid("text is required") }
        let result: [String: Any]
        if let yesNo {
            let r = try decider.yesNo(text: text, question: yesNo)
            result = ["answer": r.answer, "p_yes": r.pYes]
        } else {
            let r = try decider.choose([text], options: options, question: question, allowNone: allowNone)[0]
            result = ["label": r.label as Any? ?? NSNull(), "index": r.index as Any? ?? NSNull(),
                      "scores": r.scores, "p_none": r.pNone]
        }
        let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }
}
