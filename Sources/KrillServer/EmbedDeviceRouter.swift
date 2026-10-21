import Foundation
import KrillEngine
import KrillRegistry

/// Decides where one embedding request runs: the MLX build the client named (GPU) or the
/// Neural Engine package. Pure, so the fallback cases are unit-tested without hardware.
///
/// Any embedder or reranker may have a sibling package `<model>-ane-lin8` (or `-fp16`); with `ane`/`auto` a request
/// for `<model>` runs there, and a model without a package stays on the GPU. Every quantized EmbeddingGemma 2 text
/// alias shares the `embeddinggemma-2` package. A request that names an ANE package itself
/// (`bge-base-en-ane-lin8`) runs there regardless of the setting.
public enum EmbedDeviceRouter {
    /// Package variants in preference order: int8 first, fp16 if only that is installed.
    public static let variants = ["lin8", "fp16"]
    /// The EmbeddingGemma 2 packages (kept for callers that predate per-model packages).
    public static let packageNames = variants.map { "embeddinggemma-2-ane-\($0)" }

    /// The model whose package serves `requested`: quantized EmbeddingGemma 2 aliases share the
    /// `embeddinggemma-2` package, every other model has its own.
    public static func packageBase(for requested: String) -> String {
        isEG2Text(requested) ? "embeddinggemma-2" : requested.lowercased()
    }

    /// Candidate package names for `requested`, preferred first.
    public static func packageNames(for requested: String) -> [String] {
        variants.map { "\(packageBase(for: requested))-ane-\($0)" }
    }

    /// Process-wide "warn once per model" ledger for an `ane` setting that could not be honoured.
    public final class WarnOnce: @unchecked Sendable {
        private let lock = NSLock()
        private var seen = Set<String>()
        public init() {}
        /// True the first time `model` is seen.
        public func first(_ model: String) -> Bool {
            lock.lock(); defer { lock.unlock() }
            return seen.insert(model.lowercased()).inserted
        }
    }
    public static let warned = WarnOnce()

    public struct Decision: Equatable, Sendable {
        /// Directory to load into the embed engine.
        public let directory: URL
        /// For an ANE decision: the build to use for what the package cannot serve (the model the
        /// client asked for). nil for the GPU path.
        public let fallbackDirectory: URL?
        /// Where the request is expected to run (`.gpu` or `.ane`).
        public let device: EmbedDevice
        /// Why a requested/configured `ane` was not honoured (logged loudly once per server).
        public let warning: String?
    }

    /// True for the EmbeddingGemma 2 builds that share the text path (not an ANE package name).
    public static func isEG2Text(_ name: String) -> Bool {
        let n = name.lowercased()
        return n.hasPrefix("embeddinggemma-2") && !n.contains("-ane-")
    }

    public static func isANEPackageName(_ name: String) -> Bool { name.lowercased().contains("-ane-") }

    /// The full-precision (bf16 weights, fp32 compute) EmbeddingGemma 2 build.
    public static func isFullPrecision(_ name: String) -> Bool { name.lowercased() == "embeddinggemma-2" }

    /// - Parameters:
    ///   - requested: the `model` the client sent (registry name).
    ///   - requestedDir: its directory.
    ///   - setting: the server's `embed_device`.
    ///   - override: a per-request `device` field, if the client sent one.
    ///   - installedPackage: directory of the first installed ANE package, or nil.
    ///   - coreML: whether this platform can run Core ML.
    public static func route(requested: String, requestedDir: URL, setting: EmbedDevice,
                             override: EmbedDevice? = nil, installedPackage: URL?,
                             coreML: Bool) -> Decision {
        let gpu = Decision(directory: requestedDir, fallbackDirectory: nil, device: .gpu, warning: nil)
        if isANEPackageName(requested) {
            return Decision(directory: requestedDir, fallbackDirectory: nil, device: .ane, warning: nil)
        }
        // The full-precision build is the corpus/reference embedder: the server-wide setting never
        // moves it; only a request that explicitly asks for `ane`/`auto` does.
        if isFullPrecision(requested), override == nil { return gpu }
        let want = override ?? setting
        switch want {
        case .gpu:
            return gpu
        case .ane, .auto:
            if coreML, let pkg = installedPackage {
                return Decision(directory: pkg, fallbackDirectory: requestedDir, device: .ane, warning: nil)
            }
            guard want == .ane else { return gpu }   // auto: quietly the GPU
            let why = coreML
                ? "no ANE package is installed for \(requested) (build it: make ane-package MODEL=\(packageBase(for: requested)); see docs/ANE_EMBEDDER.md)"
                : "Core ML is not available on this platform"
            return Decision(directory: requestedDir, fallbackDirectory: nil, device: .gpu,
                            warning: "embed_device = ane but \(why); serving \(requested) on the GPU")
        }
    }

    /// What the setting resolves to right now, for the status endpoint and the startup banner.
    public static func effective(setting: EmbedDevice, packageInstalled: Bool, coreML: Bool) -> EmbedDevice {
        switch setting {
        case .gpu: return .gpu
        case .ane, .auto: return (coreML && packageInstalled) ? .ane : .gpu
        }
    }

    /// Directory of the first installed ANE package in `registry` for `model`, if any.
    public static func installedPackage(for model: String, in registry: Registry) -> URL? {
        for n in packageNames(for: model) where registry.hasModel(n) {
            let d = registry.modelPath(n)
            if ANEPackageManifest.exists(in: d) { return d }
        }
        return nil
    }

    /// Every installed ANE package as (model it serves, package directory), sorted by model.
    public static func installedPackages(in registry: Registry) -> [(model: String, directory: URL)] {
        var out: [(String, URL)] = []
        for m in registry.listModels() where isANEPackageName(m.name) {
            let d = registry.modelPath(m.name)
            guard ANEPackageManifest.exists(in: d), let mf = try? ANEPackageManifest.load(directory: d) else { continue }
            out.append((mf.model ?? "embeddinggemma-2", d))
        }
        return out.sorted { $0.0 < $1.0 }.map { (model: $0.0, directory: $0.1) }
    }

    /// Directory of the first installed EmbeddingGemma 2 package, if any.
    public static func installedPackage(in registry: Registry) -> URL? { installedPackage(for: "embeddinggemma-2", in: registry) }
}
