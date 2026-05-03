import ArgumentParser
import Foundation
import KLMRegistry

struct QuantizeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "quantize",
        abstract: "Convert a HuggingFace model to MLX format with quantization"
    )

    @Argument(help: "HuggingFace model path (e.g., 'meta-llama/Llama-3.1-8B-Instruct')")
    var hfPath: String

    @Option(name: .long, help: "Quantization bits (4 or 8)")
    var bits: Int = 4

    @Option(name: .long, help: "Quantization group size")
    var groupSize: Int = 64

    @Option(name: .long, help: "Output name for the quantized model in registry")
    var name: String?

    func run() async throws {
        // Check Python + mlx-lm availability
        let pythonPath = try findPython()
        try checkMLXLM(python: pythonPath)

        let outputName = name ?? inferName(from: hfPath, bits: bits)
        let registry = Registry()
        let outputDir = registry.modelPath(outputName)

        print("Quantizing \(hfPath)")
        print("  Bits: \(bits), Group size: \(groupSize)")
        print("  Output: \(outputName)")
        print()

        // Create output directory
        try FileManager.default.createDirectory(
            at: outputDir, withIntermediateDirectories: true)

        // Shell out to mlx-lm convert
        let script = """
        import mlx_lm
        mlx_lm.convert(
            hf_path="\(hfPath)",
            mlx_path="\(outputDir.path)",
            quantize=True,
            q_bits=\(bits),
            q_group_size=\(groupSize),
        )
        print("DONE")
        """

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = ["-c", script]

        let pipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = pipe
        process.standardError = errPipe

        print("Converting (this may take a few minutes)...")
        try process.run()
        process.waitUntilExit()

        let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let errOutput = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""

        guard process.terminationStatus == 0, output.contains("DONE") else {
            print("Error: quantization failed")
            if !errOutput.isEmpty {
                print(errOutput.prefix(500))
            }
            throw ExitCode.failure
        }

        // Calculate size and register manifest
        let totalSize = directorySize(outputDir)

        // Detect family from the output config.json
        var family: ModelFamily = .llama
        let configURL = outputDir.appendingPathComponent("config.json")
        if let configData = try? Data(contentsOf: configURL),
           let json = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let detected = ModelFamily.detect(from: json) {
            family = detected
        }

        // Get file list
        let files = try FileManager.default.contentsOfDirectory(
            at: outputDir, includingPropertiesForKeys: [.fileSizeKey])
        let modelFiles = files.map { url -> ModelFile in
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return ModelFile(path: url.lastPathComponent, sha256: "local", sizeBytes: Int64(size))
        }

        let manifest = ModelManifest(
            name: outputName,
            family: family,
            params: "?",
            quant: "\(bits)bit",
            source: hfPath,
            context: 4096,
            files: modelFiles,
            chatTemplate: family.rawValue,
            sizeBytes: totalSize
        )
        try registry.saveManifest(manifest)

        let sizeMB = Double(totalSize) / 1_048_576
        print()
        print("Done! Model quantized and registered.")
        print(String(format: "  Size: %.0f MB", sizeMB))
        print("  Run with: krillm run \(outputName)")
    }
}

// MARK: - Helpers

private func findPython() throws -> String {
    let candidates = ["python3", "python", "/usr/bin/python3", "/usr/local/bin/python3"]
    for candidate in candidates {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = [candidate]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        try? process.run()
        process.waitUntilExit()
        if process.terminationStatus == 0 {
            let path = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !path.isEmpty { return path }
        }
    }
    print("Error: Python 3 not found. Install Python to use quantize.")
    print("  brew install python3")
    throw ExitCode.failure
}

private func checkMLXLM(python: String) throws {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: python)
    process.arguments = ["-c", "import mlx_lm; print(mlx_lm.__version__)"]
    process.standardOutput = FileHandle.nullDevice
    process.standardError = FileHandle.nullDevice
    try? process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        print("Error: mlx-lm not installed. Install it with:")
        print("  pip install mlx-lm")
        throw ExitCode.failure
    }
}

private func inferName(from hfPath: String, bits: Int) -> String {
    let base = hfPath.split(separator: "/").last.map(String.init) ?? hfPath
    let cleaned = base.lowercased()
        .replacingOccurrences(of: "-instruct", with: "")
        .replacingOccurrences(of: "-chat", with: "")
    return "\(cleaned)-\(bits)bit"
}

private func directorySize(_ url: URL) -> Int64 {
    let fm = FileManager.default
    guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
        return 0
    }
    var total: Int64 = 0
    for case let fileURL as URL in enumerator {
        let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        total += Int64(size)
    }
    return total
}
