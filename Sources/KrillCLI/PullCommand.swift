import ArgumentParser
import Foundation
import KrillRegistry

struct PullCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a model from HuggingFace Hub"
    )

    @Argument(help: "Model name (alias like 'llama-3.2-3b' or HF repo 'org/name')")
    var model: String

    @Flag(name: .long, help: "Re-download even if already installed")
    var force: Bool = false

    func run() async throws {
        let registry = Registry()
        let catalog = ModelCatalogStore(baseDir: registry.baseDir)

        guard let resolved = AliasMap.resolve(model, catalog: catalog) else {
            print("Error: unknown model '\(model)'")
            print("\nAvailable models:")
            for (name, info) in AliasMap.allAliases {
                print("  \(name.padding(toLength: 20, withPad: " ", startingAt: 0)) \(info.params) \(info.family.rawValue)")
            }
            if let extra = catalog.load()?.models, !extra.isEmpty {
                print("\nFrom catalog (krill catalog list):")
                for entry in extra.sorted(by: { $0.alias < $1.alias }) {
                    print("  \(entry.alias.padding(toLength: 20, withPad: " ", startingAt: 0)) \(entry.params) \(entry.family.rawValue)")
                }
            }
            print("\nOr use a full HuggingFace repo path: krill pull org/model-name")
            throw ExitCode.failure
        }

        print(Ansi.bold(">_ ") + "Pulling " + Ansi.bold(resolved.name) + Ansi.dim(" from \(resolved.repo)"))
        print(Ansi.dim("   \(resolved.family.rawValue) \u{00B7} \(resolved.params) \u{00B7} \(resolved.quant)"))
        print()

        let puller = Puller(registry: registry)

        let startTime = CFAbsoluteTimeGetCurrent()

        let manifest = try await puller.pull(resolved, force: force) { downloaded, total, file in
            if file == "done" {
                print()
            } else {
                let pct = total > 0 ? Int(Double(downloaded) / Double(total) * 100) : 0
                let barW = 24
                let filled = max(0, min(barW, pct * barW / 100))
                let bar = String(repeating: "=", count: filled)
                    + String(repeating: " ", count: barW - filled)
                let line = "  " + Ansi.dim("[\(bar)]") + String(format: " %3d%%  ", pct) + Ansi.dim(file)
                print(Ansi.clearLine + line, terminator: "")
                fflush(stdout)
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let sizeMB = Double(manifest.sizeBytes) / 1_048_576

        print(Ansi.bold("Done.") + " \(manifest.name) installed.")
        print(Ansi.dim(String(format: "   %.0f MB \u{00B7} %.1fs", sizeMB, elapsed)))
        print(Ansi.dim("   \(registry.modelPath(manifest.name).path)"))
        print("\nRun with: " + Ansi.bold("krill run \(manifest.name)"))
    }
}
