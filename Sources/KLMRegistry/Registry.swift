import Foundation
import Logging

/// Local model registry managing ~/.krillm/models.
///
/// Layout:
/// ```
/// ~/.krillm/
///   models/
///     manifests/       # <name>.json manifest files
///     blobs/           # model weight directories (named by model identifier)
///   cache/             # prefix cache (Phase 3)
///   config.toml        # user configuration (Phase 2+)
/// ```
public final class Registry: Sendable {
    public let baseDir: URL
    public let modelsDir: URL
    public let manifestsDir: URL
    public let blobsDir: URL

    private let logger = Logger(label: "krillm.registry")

    /// Initialize registry at the default or custom location.
    ///
    /// `modelsDir` (e.g. `OLLAMA_MODELS` / `KRILL_MODELS_DIR`) points directly
    /// at the models directory and, when given, wins over `baseDir`; the
    /// manifests/blobs subtree is rooted there exactly like Ollama. Otherwise
    /// the models dir is derived as `baseDir/models`.
    public init(baseDir: URL? = nil, modelsDir: URL? = nil) {
        if let modelsDir {
            self.modelsDir = modelsDir
            self.baseDir = modelsDir.deletingLastPathComponent()
        } else {
            let base = baseDir ?? Registry.defaultBaseDir()
            self.baseDir = base
            self.modelsDir = base.appendingPathComponent("models")
        }
        self.manifestsDir = self.modelsDir.appendingPathComponent("manifests")
        self.blobsDir = self.modelsDir.appendingPathComponent("blobs")
    }

    /// Default base directory: ~/.krillm
    public static func defaultBaseDir() -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".krillm")
    }

    /// Ensure the directory structure exists.
    public func ensureDirectories() throws {
        let fm = FileManager.default
        try fm.createDirectory(at: manifestsDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: blobsDir, withIntermediateDirectories: true)
    }

    /// List all installed models.
    public func listModels() -> [ModelManifest] {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: manifestsDir, includingPropertiesForKeys: nil
        ) else {
            return []
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        return files
            .filter { $0.pathExtension == "json" }
            .compactMap { url in
                guard let data = try? Data(contentsOf: url),
                      let manifest = try? decoder.decode(ModelManifest.self, from: data) else {
                    return nil
                }
                return manifest
            }
            .sorted { $0.name < $1.name }
    }

    /// Reject model names that could escape the registry directory or
    /// otherwise resolve to an unintended path. A model name is an opaque
    /// identifier, never a path: no separators, no `..`, no leading `.`,
    /// not absolute, length-bounded.
    public static func isValidModelName(_ name: String) -> Bool {
        guard !name.isEmpty, name.count <= 200 else { return false }
        if name.hasPrefix(".") || name.hasPrefix("/") || name.hasPrefix("~") {
            return false
        }
        if name.contains("/") || name.contains("\\") || name.contains("..") {
            return false
        }
        // Disallow control characters and path-significant whitespace.
        return !name.unicodeScalars.contains { $0.value < 0x20 || $0 == "\u{7f}" }
    }

    public enum RegistryError: Error, CustomStringConvertible {
        case invalidModelName(String)
        public var description: String {
            switch self {
            case .invalidModelName(let n):
                return "Invalid model name '\(n)': must not contain path separators, '..', or leading '.'"
            }
        }
    }

    static func requireValidName(_ name: String) throws {
        guard isValidModelName(name) else {
            throw RegistryError.invalidModelName(name)
        }
    }

    /// Get a specific model by name.
    public func getModel(_ name: String) -> ModelManifest? {
        guard Self.isValidModelName(name) else { return nil }
        let manifestURL = manifestsDir.appendingPathComponent("\(name).json")
        guard let data = try? Data(contentsOf: manifestURL) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(ModelManifest.self, from: data)
    }

    /// Get the local file path for a model's weights directory.
    public func modelPath(_ name: String) -> URL {
        blobsDir.appendingPathComponent(name)
    }

    /// Check if a model is installed.
    public func hasModel(_ name: String) -> Bool {
        guard Self.isValidModelName(name) else { return false }
        let manifestURL = manifestsDir.appendingPathComponent("\(name).json")
        return FileManager.default.fileExists(atPath: manifestURL.path)
    }

    /// Save a model manifest after pulling.
    public func saveManifest(_ manifest: ModelManifest) throws {
        try Self.requireValidName(manifest.name)
        try ensureDirectories()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(manifest)
        let url = manifestsDir.appendingPathComponent("\(manifest.name).json")
        try data.write(to: url)
        logger.info("Saved manifest for \(manifest.name)")
    }

    /// Remove a model (manifest + blob directory).
    public func removeModel(_ name: String) throws {
        try Self.requireValidName(name)
        let fm = FileManager.default
        let manifestURL = manifestsDir.appendingPathComponent("\(name).json")
        let blobDir = blobsDir.appendingPathComponent(name)

        if fm.fileExists(atPath: blobDir.path) {
            try fm.removeItem(at: blobDir)
        }
        if fm.fileExists(atPath: manifestURL.path) {
            try fm.removeItem(at: manifestURL)
        }
        logger.info("Removed model \(name)")
    }

    /// Total disk usage of all installed models.
    public func totalDiskUsage() -> Int64 {
        listModels().reduce(0) { $0 + $1.sizeBytes }
    }
}
