import ArgumentParser
import Foundation
import KrillCore
import KrillEngine
import KrillRegistry

struct TuneCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tune",
        abstract: "Fine-tune an installed reader decision model and install the result"
    )

    @Option(name: .long, help: "Name of an installed reader decision model")
    var model: String

    @Option(name: .long, help: "Training examples in a JSONL file")
    var data: String

    @Option(name: .long, help: "Name to install the tuned model under")
    var name: String

    @Option(name: .long, help: "Maximum number of training epochs")
    var epochs = 30

    @Option(name: .long, help: "Adam learning rate")
    var lr: Float = 3e-3

    @Option(name: .long, help: "Holdout fraction when there are at least 20 examples (0 disables it)")
    var holdout = 0.2

    @Option(name: .long, help: "Seed for deterministic split and batch ordering")
    var seed = 0

    @Flag(name: .long, help: "Install even if held-out accuracy decreases")
    var force = false

    func run() async throws {
        let registry = Registry.configured()
        guard let base = registry.getModel(model) else { throw ValidationError("decision model not found: '\(model)'") }
        guard base.family == .decision else {
            throw ValidationError("'\(model)' (family \(base.family.rawValue)) is not a decision model")
        }

        do {
            try registry.requireAvailableModelName(name)
            let input = try Data(contentsOf: URL(fileURLWithPath: data))
            let lines = input.isEmpty ? [] : input.split(separator: 0x0a, omittingEmptySubsequences: false)
            let examples = try (input.last == 0x0a ? lines.dropLast() : lines[...]).enumerated().map { i, line in
                let row: Any
                do { row = try JSONSerialization.jsonObject(with: Data(line), options: [.fragmentsAllowed]) }
                catch { throw DecisionError.invalid("\(data):\(i + 1): invalid JSON") }
                return try DecisionTuneExample(json: row as? [String: Any], number: i + 1)
            }
            let options = DecisionTuneOptions(epochs: epochs, lr: lr, holdout: holdout, seed: seed, force: force)
            try options.validate(examples: examples)
            let engine = DecisionEngine()
            let loaded = try await engine.load(directory: registry.modelPath(model), forTuning: true)
            let report = try engine.tune(loaded, base: base, name: name, examples: examples,
                                         options: options, registry: registry)
            let json = try JSONSerialization.data(withJSONObject: report.jsonObject, options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: json, as: UTF8.self))
            guard report.accepted else {
                FileHandle.standardError.write(Data("Held-out accuracy decreased. Use --force to install anyway.\n".utf8))
                throw ExitCode.failure
            }
            let quotedName = name.replacingOccurrences(of: "'", with: "'\\''")
            let separator = name.hasPrefix("-") ? "=" : " "
            print("krill decide --model\(separator)'\(quotedName)' --json-in -")
        } catch let error as DecisionError {
            throw ValidationError(error.description)
        } catch let error as Registry.RegistryError {
            throw ValidationError(error.description)
        }
    }
}
