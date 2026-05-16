import ArgumentParser
import Foundation
import KLMRegistry

struct ShowCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "show",
        abstract: "Show a model's metadata, parameters, template, and system prompt"
    )

    @Argument(help: "Model name")
    var name: String

    @Flag(name: .long, help: "Print only the Modelfile")
    var modelfile = false
    @Flag(name: .long, help: "Print only the parameters")
    var parameters = false
    @Flag(name: .long, help: "Print only the template")
    var template = false
    @Flag(name: .long, help: "Print only the system prompt")
    var system = false

    func run() throws {
        let registry = Registry()
        guard let m = registry.getModel(name) else {
            print("Error: model '\(name)' not found")
            throw ExitCode.failure
        }
        let ov = m.overrides
        let mf = """
        FROM \(m.source)
        \(ov?.parameters.map { "PARAMETER \($0.key) \($0.value)" }.joined(separator: "\n") ?? "")
        \(ov?.system.map { "SYSTEM \"\"\"\n\($0)\n\"\"\"" } ?? "")
        \(ov?.template.map { "TEMPLATE \"\"\"\n\($0)\n\"\"\"" } ?? "")
        """.trimmingCharacters(in: .whitespacesAndNewlines)

        if modelfile { print(mf); return }
        if parameters {
            print((ov?.parameters ?? [:]).map { "\($0.key) \($0.value)" }
                .joined(separator: "\n"))
            return
        }
        if template { print(ov?.template ?? m.chatTemplate); return }
        if system { print(ov?.system ?? ""); return }

        print("Model:   \(m.name)")
        print("Family:  \(m.family.rawValue)")
        print("Params:  \(m.params)")
        print("Quant:   \(m.quant)")
        print("Context: \(m.context)")
        print("Source:  \(m.source)")
        if let s = ov?.system { print("System:  \(s)") }
        if let ov, !ov.parameters.isEmpty { print("Parameters: \(ov.parameters)") }
        if let t = ov?.template { print("Template:\n\(t)") }
        if let l = ov?.license { print("License: \(l.prefix(80))") }
    }
}
