import ArgumentParser
import Foundation
import KLMCore
import KLMRegistry

struct QuantizeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "quantize",
        abstract: "Quantize a dense model to MLX format, natively (no Python / mlx-lm)"
    )

    @Argument(help: "Source model: a local directory, or a HuggingFace repo id already in the local HF cache (e.g. 'mlx-community/GLM-4-9B-0414-bf16').")
    var source: String

    @Option(name: .long, help: "Quantization bits (4 or 8)")
    var bits: Int = 4

    @Option(name: .long, help: "Quantization group size")
    var groupSize: Int = 64

    @Option(name: .long, help: "Quantization mode: affine (default), nvfp4, mxfp4, mxfp8")
    var mode: String = "affine"

    @Option(name: .long, help: "Storage dtype for scales/biases/norms: fp16 (default, mlx-community convention), bf16, fp32")
    var dtype: String = "fp16"

    @Option(name: .long, help: "A 4-bit build of this model (local dir or HF repo id) to learn the per-module quantized set from. Required for MoE / vision / Gemma; reproduces that build's coverage exactly.")
    var reference: String?

    @Option(name: .long, parsing: .singleValue, help: "Module-path substring to quantize at the protect precision (repeatable), e.g. --protect down_proj --protect o_proj.")
    var protect: [String] = []

    @Option(name: .long, help: "Protect precision bits (default 8)")
    var protectBits: Int = 8

    @Option(name: .long, help: "Protect precision group size (default 64)")
    var protectGroupSize: Int = 64

    @Option(name: .long, help: "Protect precision mode: affine (default), mxfp8")
    var protectMode: String = "affine"

    @Flag(name: .long, inversion: .prefixedNo, help: "Auto-protect the vision/audio projectors at the protect precision in reference mode (default on).")
    var protectVision: Bool = true

    @Option(name: .long, help: "Output name for the quantized model in registry")
    var name: String?

    func run() async throws {
        guard let sourceDir = resolveSource(source) else {
            print("Error: could not find source model '\(source)'.")
            print("  Pass a local directory, or a HuggingFace repo id already downloaded")
            print("  into ~/.cache/huggingface/hub (pull it first with huggingface-cli or krill).")
            throw ExitCode.failure
        }

        var referenceDir: URL?
        if let reference {
            guard let r = resolveSource(reference) else {
                print("Error: could not find --reference model '\(reference)'.")
                throw ExitCode.failure
            }
            referenceDir = r
        }

        let outputName = name ?? inferName(from: source, bits: bits)
        let registry = Registry()
        let outputDir = registry.modelPath(outputName)

        print("Quantizing \(sourceDir.lastPathComponent)")
        print("  Bits: \(bits), Group size: \(groupSize), Mode: \(mode)")
        if let referenceDir { print("  Reference: \(referenceDir.lastPathComponent)") }
        print("  Output: \(outputName)")
        print("Converting (this may take a few minutes)...")

        do {
            let n = try CheckpointQuantizer.quantize(
                sourceDir: sourceDir, outputDir: outputDir,
                bits: bits, groupSize: groupSize, mode: mode, dtype: dtype,
                referenceDir: referenceDir,
                protect: protect,
                protectBits: protectBits, protectGroupSize: protectGroupSize,
                protectMode: protectMode, autoProtectVision: protectVision,
                log: { print("  \($0)") })
            print("  wrote \(n) quantized tensors")
        } catch {
            print("Error: quantization failed: \(error)")
            // Don't leave a half-written model registered.
            try? FileManager.default.removeItem(at: outputDir)
            throw ExitCode.failure
        }

        // Register the manifest (size, family from the emitted config.json).
        let totalSize = directorySize(outputDir)
        var family: ModelFamily = .llama
        let configURL = outputDir.appendingPathComponent("config.json")
        if let configData = try? Data(contentsOf: configURL),
           let json = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let detected = ModelFamily.detect(from: json) {
            family = detected
        }

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
            quant: mode == "affine" ? "\(bits)bit" : mode,
            source: source,
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
        print("  Run with: krill run \(outputName)")
    }
}

// MARK: - Helpers

/// Resolve a source argument to a model directory: an existing local path, or a
/// HuggingFace repo id (`org/name`) already snapshot-downloaded into the local
/// HF cache (`~/.cache/huggingface/hub/models--org--name/snapshots/<latest>`).
private func resolveSource(_ source: String) -> URL? {
    let fm = FileManager.default
    var isDir: ObjCBool = false
    if fm.fileExists(atPath: source, isDirectory: &isDir), isDir.boolValue {
        return URL(fileURLWithPath: source, isDirectory: true)
    }
    guard source.contains("/") else { return nil }
    let repoSlug = "models--" + source.replacingOccurrences(of: "/", with: "--")
    let home = fm.homeDirectoryForCurrentUser
    let snapshots = home
        .appendingPathComponent(".cache/huggingface/hub")
        .appendingPathComponent(repoSlug)
        .appendingPathComponent("snapshots")
    guard let entries = try? fm.contentsOfDirectory(
        at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey]) else {
        return nil
    }
    // Most-recently-modified snapshot that actually has a config.json.
    let withConfig = entries.filter {
        fm.fileExists(atPath: $0.appendingPathComponent("config.json").path)
    }
    return withConfig.sorted {
        let a = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        let b = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        return a > b
    }.first
}

private func inferName(from source: String, bits: Int) -> String {
    let base = source.split(separator: "/").last.map(String.init) ?? source
    let cleaned = base.lowercased()
        .replacingOccurrences(of: "-instruct", with: "")
        .replacingOccurrences(of: "-chat", with: "")
        .replacingOccurrences(of: "-bf16", with: "")
        .replacingOccurrences(of: "-fp16", with: "")
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
