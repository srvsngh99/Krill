import ArgumentParser
import Foundation
import KrillRegistry

struct ListCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "Show installed models"
    )

    func run() throws {
        let registry = Registry()
        let models = registry.listModels()

        if models.isEmpty {
            print("No models installed.")
            print("\nPull a model with: krill pull gemma-4-e2b")
            return
        }

        // Header: name column sized to the longest name so nothing is
        // truncated into the SIZE column.
        let nameW = max(22, (models.map { $0.name.count }.max() ?? 0) + 2)
        let sizeW = 10
        let familyW = max(10, (models.map { $0.family.rawValue.count }.max() ?? 0) + 2)
        let paramsW = 8
        let quantW = 6

        print(
            "NAME".padding(toLength: nameW, withPad: " ", startingAt: 0)
            + "SIZE".padding(toLength: sizeW, withPad: " ", startingAt: 0)
            + "FAMILY".padding(toLength: familyW, withPad: " ", startingAt: 0)
            + "PARAMS".padding(toLength: paramsW, withPad: " ", startingAt: 0)
            + "QUANT".padding(toLength: quantW, withPad: " ", startingAt: 0)
        )

        for model in models {
            let sizeMB = Double(registry.diskUsage(of: model)) / 1_048_576
            let sizeStr: String
            if sizeMB >= 1024 {
                sizeStr = String(format: "%.1f GB", sizeMB / 1024)
            } else {
                sizeStr = String(format: "%.0f MB", sizeMB)
            }

            print(
                model.name.padding(toLength: nameW, withPad: " ", startingAt: 0)
                + sizeStr.padding(toLength: sizeW, withPad: " ", startingAt: 0)
                + model.family.rawValue.padding(toLength: familyW, withPad: " ", startingAt: 0)
                + model.params.padding(toLength: paramsW, withPad: " ", startingAt: 0)
                + model.quant.padding(toLength: quantW, withPad: " ", startingAt: 0)
            )
        }

        let totalGB = Double(registry.totalDiskUsage()) / 1_073_741_824
        print(String(format: "\n%d model(s), %.1f GB total", models.count, totalGB))
    }
}
