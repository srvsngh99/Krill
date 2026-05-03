import ArgumentParser
import Foundation
import KLMRegistry

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
            print("\nPull a model with: krillm pull llama-3.2-3b")
            return
        }

        // Header
        let nameW = 22
        let sizeW = 10
        let familyW = 10
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
            let sizeMB = Double(model.sizeBytes) / 1_048_576
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
