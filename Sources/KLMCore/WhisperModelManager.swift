import Foundation

// MARK: - Whisper model management (consent + download)

/// Resolves and (on consent) downloads the native Whisper weights. Models live
/// under `~/.krillm/models/whisper-<sku>` as raw HuggingFace files
/// (model.safetensors + vocab.json + config.json); `WhisperRuntime` remaps the
/// HF key layout at load, so no Python conversion step is needed.
public enum WhisperModelManager {

    /// A downloadable English dictation SKU.
    public struct SKU: Sendable {
        public let id: String          // e.g. "base.en"
        public let repo: String        // HuggingFace repo
        public let approxMB: Int       // download size (fp32)
    }

    public static let skus: [SKU] = [
        SKU(id: "tiny.en", repo: "openai/whisper-tiny.en", approxMB: 151),
        SKU(id: "base.en", repo: "openai/whisper-base.en", approxMB: 290),
        SKU(id: "small.en", repo: "openai/whisper-small.en", approxMB: 967),
    ]

    /// Balanced default: noticeably better than tiny, far lighter than small.
    public static let defaultSKU = "base.en"

    public static func sku(_ id: String) -> SKU? { skus.first { $0.id == id } }

    /// `~/.krillm/models/whisper-<sku>`.
    public static func modelDir(_ skuID: String) -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".krillm/models/whisper-\(skuID)")
    }

    public static func isInstalled(_ skuID: String) -> Bool {
        let dir = modelDir(skuID)
        return ["model.safetensors", "vocab.json", "config.json"].allSatisfy {
            FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
    }

    public enum DownloadError: Error, CustomStringConvertible {
        case unknownSKU(String)
        case httpStatus(String, Int)
        public var description: String {
            switch self {
            case .unknownSKU(let s): return "unknown Whisper model '\(s)'"
            case .httpStatus(let f, let c): return "download of \(f) failed (HTTP \(c))"
            }
        }
    }

    /// Download the three files for `skuID` into its model dir. `progress` is
    /// called with a human-readable status line (e.g. "model.safetensors 42%").
    /// Files are written atomically; a partial dir is removed on failure.
    public static func download(
        _ skuID: String,
        progress: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws {
        guard let sku = sku(skuID) else { throw DownloadError.unknownSKU(skuID) }
        let dir = modelDir(skuID)
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        let files = ["config.json", "vocab.json", "model.safetensors"]
        do {
            for file in files {
                let url = URL(string: "https://huggingface.co/\(sku.repo)/resolve/main/\(file)")!
                progress("\(file): starting")
                try await fetch(url, to: dir.appendingPathComponent(file), label: file, progress: progress)
            }
        } catch {
            try? fm.removeItem(at: dir)
            throw error
        }
    }

    private static func fetch(
        _ url: URL, to dest: URL, label: String,
        progress: @escaping @Sendable (String) -> Void
    ) async throws {
        var req = URLRequest(url: url)
        req.setValue("krillm", forHTTPHeaderField: "User-Agent")
        let (tempURL, response) = try await URLSession.shared.download(for: req)
        let fm = FileManager.default
        if let http = response as? HTTPURLResponse, !(200 ... 299).contains(http.statusCode) {
            try? fm.removeItem(at: tempURL)   // don't leak the URLSession temp file
            throw DownloadError.httpStatus(label, http.statusCode)
        }
        if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
        try fm.moveItem(at: tempURL, to: dest)
        progress("\(label): done")
    }
}
