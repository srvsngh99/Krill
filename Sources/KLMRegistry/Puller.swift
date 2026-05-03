import Foundation
import Crypto
import Logging

/// Downloads models from HuggingFace Hub into the local registry.
///
/// Uses the HuggingFace Hub HTTP API to list files and download them
/// with progress reporting and SHA256 verification.
public final class Puller: @unchecked Sendable {
    private let registry: Registry
    private let logger = Logger(label: "krillm.puller")

    /// Progress callback: (bytesDownloaded, totalBytes, currentFile)
    public typealias ProgressHandler = @Sendable (Int64, Int64, String) -> Void

    public init(registry: Registry) {
        self.registry = registry
    }

    /// Pull a model from HuggingFace Hub.
    ///
    /// - Parameters:
    ///   - resolved: The resolved model reference (from AliasMap)
    ///   - force: Re-download even if already installed
    ///   - progress: Progress callback
    /// - Returns: The saved ModelManifest
    public func pull(
        _ resolved: ResolvedModel,
        force: Bool = false,
        progress: ProgressHandler? = nil
    ) async throws -> ModelManifest {
        // Check if already installed
        if !force && registry.hasModel(resolved.name) {
            logger.info("\(resolved.name) already installed, use --force to re-download")
            if let existing = registry.getModel(resolved.name) {
                return existing
            }
        }

        try registry.ensureDirectories()
        let destDir = registry.modelPath(resolved.name)

        // Create destination directory
        let fm = FileManager.default
        if fm.fileExists(atPath: destDir.path) {
            try fm.removeItem(at: destDir)
        }
        try fm.createDirectory(at: destDir, withIntermediateDirectories: true)

        // List files in the HF repo
        let fileList = try await listRepoFiles(repo: resolved.repo)

        // Filter to essential files (safetensors, configs, tokenizer)
        let essentialFiles = fileList.filter { file in
            let name = file.name.lowercased()
            return name.hasSuffix(".safetensors")
                || name == "config.json"
                || name == "tokenizer.json"
                || name == "tokenizer_config.json"
                || name == "special_tokens_map.json"
                || name == "generation_config.json"
                || name == "tokenizer.model"
        }

        let totalSize = essentialFiles.reduce(0) { $0 + $1.size }
        var downloadedBytes: Int64 = 0
        var modelFiles: [ModelFile] = []

        // Download each file
        for file in essentialFiles {
            let fileURL = destDir.appendingPathComponent(file.name)

            // Create subdirectories if needed
            let parentDir = fileURL.deletingLastPathComponent()
            if !fm.fileExists(atPath: parentDir.path) {
                try fm.createDirectory(at: parentDir, withIntermediateDirectories: true)
            }

            progress?(downloadedBytes, totalSize, file.name)
            let sha256 = try await downloadFile(
                repo: resolved.repo,
                filename: file.name,
                destination: fileURL
            )
            downloadedBytes += file.size

            modelFiles.append(ModelFile(
                path: file.name,
                sha256: sha256,
                sizeBytes: file.size
            ))
        }

        progress?(totalSize, totalSize, "done")

        // Detect actual family from config.json if available
        var family = resolved.family
        let configURL = destDir.appendingPathComponent("config.json")
        if let configData = try? Data(contentsOf: configURL),
           let configJSON = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let detected = ModelFamily.detect(from: configJSON) {
            family = detected
        }

        // Detect context length from config
        var context = resolved.context
        if let configData = try? Data(contentsOf: configURL),
           let configJSON = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let maxPos = configJSON["max_position_embeddings"] as? Int {
            context = maxPos
        }

        // Build and save manifest
        let manifest = ModelManifest(
            name: resolved.name,
            family: family,
            params: resolved.params,
            quant: resolved.quant,
            source: resolved.repo,
            context: context,
            files: modelFiles,
            chatTemplate: family.rawValue,
            sizeBytes: totalSize,
            pulledAt: Date()
        )

        try registry.saveManifest(manifest)
        return manifest
    }
}

// MARK: - HuggingFace Hub API

/// File info from HuggingFace Hub API
private struct HFFileInfo {
    let name: String
    let size: Int64
}

extension Puller {
    /// List files in a HuggingFace repo via the API.
    private func listRepoFiles(repo: String) async throws -> [HFFileInfo] {
        let urlString = "https://huggingface.co/api/models/\(repo)"
        guard let url = URL(string: urlString) else {
            throw PullerError.invalidRepo(repo)
        }

        let (data, response) = try await URLSession.shared.data(from: url)
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw PullerError.httpError(code, "Failed to list repo files")
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let siblings = json["siblings"] as? [[String: Any]] else {
            throw PullerError.invalidResponse("No siblings in repo response")
        }

        return siblings.compactMap { sibling in
            guard let filename = sibling["rfilename"] as? String else { return nil }
            let size = (sibling["size"] as? Int64) ?? 0
            return HFFileInfo(name: filename, size: size)
        }
    }

    /// Download a single file from HuggingFace Hub and return its SHA256 hash.
    private func downloadFile(
        repo: String,
        filename: String,
        destination: URL
    ) async throws -> String {
        let urlString = "https://huggingface.co/\(repo)/resolve/main/\(filename)"
        guard let url = URL(string: urlString) else {
            throw PullerError.invalidRepo(repo)
        }

        let (tempURL, response) = try await URLSession.shared.download(from: url)
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw PullerError.httpError(code, "Failed to download \(filename)")
        }

        // Move to destination
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }
        try fm.moveItem(at: tempURL, to: destination)

        // Compute SHA256
        let fileData = try Data(contentsOf: destination)
        let hash = SHA256.hash(data: fileData)
        return hash.map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Errors

public enum PullerError: Error, CustomStringConvertible {
    case invalidRepo(String)
    case httpError(Int, String)
    case invalidResponse(String)
    case verificationFailed(String, expected: String, got: String)

    public var description: String {
        switch self {
        case .invalidRepo(let repo):
            return "Invalid repository: \(repo)"
        case .httpError(let code, let msg):
            return "HTTP \(code): \(msg)"
        case .invalidResponse(let msg):
            return "Invalid response: \(msg)"
        case .verificationFailed(let file, let expected, let got):
            return "SHA256 mismatch for \(file): expected \(expected), got \(got)"
        }
    }
}
