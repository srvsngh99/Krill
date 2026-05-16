import ArgumentParser
import Foundation
import KLMRegistry

struct CpCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "cp",
        abstract: "Copy a model to a new name (weights are referenced, not duplicated)"
    )

    @Argument(help: "Source model name")
    var source: String

    @Argument(help: "Destination model name")
    var destination: String

    func run() throws {
        let registry = Registry()
        guard let src = registry.getModel(source) else {
            print("Error: model '\(source)' not found")
            throw ExitCode.failure
        }
        let fm = FileManager.default
        let dst = registry.modelPath(destination)
        if fm.fileExists(atPath: dst.path) { try? fm.removeItem(at: dst) }
        do {
            let srcDir = registry.modelPath(source).resolvingSymlinksInPath()
            try fm.createDirectory(at: dst, withIntermediateDirectories: true)
            for entry in (try? fm.contentsOfDirectory(atPath: srcDir.path)) ?? [] {
                try? fm.createSymbolicLink(
                    at: dst.appendingPathComponent(entry),
                    withDestinationURL: srcDir.appendingPathComponent(entry))
            }
            let copied = ModelManifest(
                name: destination, family: src.family, params: src.params,
                quant: src.quant, source: src.source, context: src.context,
                files: src.files, draftPair: src.draftPair,
                chatTemplate: src.chatTemplate, sizeBytes: src.sizeBytes,
                pulledAt: Date(), overrides: src.overrides)
            try registry.saveManifest(copied)
            print("Copied '\(source)' -> '\(destination)'.")
        } catch {
            print("Error: \(error)")
            throw ExitCode.failure
        }
    }
}
