import Foundation

#if canImport(Metal)
import Metal
#endif

public enum MLXMetalRuntimeError: Error, CustomStringConvertible, LocalizedError {
    case noMetalDevice
    case metallibNotFound([URL])

    public var description: String {
        switch self {
        case .noMetalDevice:
            return "MLX cannot initialize because this process cannot see a Metal GPU device."
        case .metallibNotFound(let candidates):
            let searched = candidates.map(\.path).joined(separator: "\n  ")
            return """
            MLX Metal runtime library was not found. Build it with `make metallib` or `make release`.
            Searched:
              \(searched)
            """
        }
    }

    public var errorDescription: String? { description }
}

public struct MLXMetalResourceLocator: Sendable {
    public static let swiftPMBundleName = "mlx-swift_Cmlx"

    public struct Context: Sendable {
        public var executableDirectory: URL
        public var mainBundleURL: URL?
        public var bundleResourceURLs: [URL]
        public var frameworkResourceURLs: [URL]
        public var currentDirectory: URL

        public init(
            executableDirectory: URL,
            mainBundleURL: URL?,
            bundleResourceURLs: [URL],
            frameworkResourceURLs: [URL],
            currentDirectory: URL
        ) {
            self.executableDirectory = executableDirectory
            self.mainBundleURL = mainBundleURL
            self.bundleResourceURLs = bundleResourceURLs
            self.frameworkResourceURLs = frameworkResourceURLs
            self.currentDirectory = currentDirectory
        }
    }

    public static func defaultContext() -> Context {
        let executableDirectory = Bundle.main.executableURL?
            .deletingLastPathComponent()
            ?? URL(fileURLWithPath: CommandLine.arguments.first ?? ".")
                .deletingLastPathComponent()

        let frameworkResourceURLs = Bundle.allFrameworks.compactMap { bundle -> URL? in
            bundle.bundleIdentifier == swiftPMBundleName ? bundle.resourceURL : nil
        }

        return Context(
            executableDirectory: executableDirectory,
            mainBundleURL: Bundle.main.bundleURL,
            bundleResourceURLs: Bundle.allBundles.compactMap(\.resourceURL),
            frameworkResourceURLs: frameworkResourceURLs,
            currentDirectory: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        )
    }

    public static func candidateURLs(context: Context = defaultContext()) -> [URL] {
        var candidates: [URL] = []
        candidates.append(context.executableDirectory.appendingPathComponent("mlx.metallib"))
        candidates.append(
            context.executableDirectory
                .appendingPathComponent("Resources")
                .appendingPathComponent("mlx.metallib")
        )

        if let mainBundleURL = context.mainBundleURL {
            candidates.append(swiftPMBundleDefaultMetallib(relativeTo: mainBundleURL))
        }

        for resourceURL in context.bundleResourceURLs {
            candidates.append(swiftPMBundleDefaultMetallib(relativeTo: resourceURL))
        }

        for resourceURL in context.frameworkResourceURLs {
            candidates.append(resourceURL.appendingPathComponent("default.metallib"))
        }

        candidates.append(
            context.executableDirectory
                .appendingPathComponent("Resources")
                .appendingPathComponent("default.metallib")
        )
        candidates.append(context.currentDirectory.appendingPathComponent("default.metallib"))

        var seen = Set<String>()
        return candidates.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }

    public static func firstAvailableResourceURL(
        context: Context = defaultContext(),
        fileManager: FileManager = .default
    ) -> URL? {
        candidateURLs(context: context).first {
            fileManager.fileExists(atPath: $0.path)
        }
    }

    private static func swiftPMBundleDefaultMetallib(relativeTo url: URL) -> URL {
        url.appendingPathComponent("\(swiftPMBundleName).bundle")
            .appendingPathComponent("default.metallib")
    }
}

public enum MLXMetalRuntime {
    public static func validateForNativeInference() throws {
        #if canImport(Metal)
        guard !MTLCopyAllDevices().isEmpty else {
            throw MLXMetalRuntimeError.noMetalDevice
        }
        #else
        throw MLXMetalRuntimeError.noMetalDevice
        #endif

        let candidates = MLXMetalResourceLocator.candidateURLs()
        guard MLXMetalResourceLocator.firstAvailableResourceURL() != nil else {
            throw MLXMetalRuntimeError.metallibNotFound(candidates)
        }
    }

    public static var canInitializeMLXForTests: Bool {
        #if canImport(Metal)
        guard !MTLCopyAllDevices().isEmpty else {
            return false
        }
        #else
        return false
        #endif

        return MLXMetalResourceLocator.firstAvailableResourceURL() != nil
    }
}
