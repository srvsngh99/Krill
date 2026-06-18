import ArgumentParser
import Foundation
import KrillRegistry

struct CreateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "create",
        abstract: "Create a customized model from a Modelfile"
    )

    @Argument(help: "Name for the new model")
    var name: String

    @Option(name: [.short, .long], help: "Path to the Modelfile")
    var file: String = "Modelfile"

    func run() async throws {
        let url = URL(fileURLWithPath: file)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            print("Error: cannot read Modelfile at \(file)")
            throw ExitCode.failure
        }
        let modelfile: Modelfile
        do {
            modelfile = try ModelfileParser.parse(text)
        } catch {
            print("Error: \(error)")
            throw ExitCode.failure
        }
        let registry = Registry()
        do {
            let m = try registry.createModel(name: name, from: modelfile)
            if let w = modelfile.adapterWarning { print("warning: \(w)") }
            print("Created '\(m.name)' from '\(modelfile.from)'.")
            if let s = m.overrides?.system { print("  SYSTEM: \(s.prefix(60))…") }
            if !(m.overrides?.parameters.isEmpty ?? true) {
                print("  PARAMETERS: \(m.overrides!.parameters)")
            }
            print("Run with: krill run \(m.name)")
        } catch {
            print("Error: \(error)")
            throw ExitCode.failure
        }
    }
}
