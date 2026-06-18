import ArgumentParser
import Foundation
import KLMRegistry

struct RemoveCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rm",
        abstract: "Remove an installed model"
    )

    @Argument(help: "Name of the model to remove")
    var model: String

    func run() throws {
        let registry = Registry()

        guard registry.hasModel(model) else {
            print("Error: model '\(model)' not found")
            print("Run 'krill list' to see installed models.")
            throw ExitCode.failure
        }

        try registry.removeModel(model)
        print("Removed \(model)")
    }
}
