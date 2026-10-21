import Foundation
import KrillEngine
import KrillRegistry

/// Decides where one embedding request runs: the MLX build the client named (GPU) or the
/// Neural Engine package. Pure, so the fallback cases are unit-tested without hardware.
///
/// Only EmbeddingGemma 2 text builds are routed (`embeddinggemma-2`, `embeddinggemma-2-mlx-4bit-dyn-text`,
/// ...). Every other embedder is left alone. A request that names an ANE package itself
/// (`embeddinggemma-2-ane-lin8`) runs there regardless of the setting.
public enum EmbedDeviceRouter {
    /// ANE packages in preference order: int8 first, fp16 if only that is installed.
    public static let packageNames = ["embeddinggemma-2-ane-lin8", "embeddinggemma-2-ane-fp16"]

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
        guard isEG2Text(requested) else { return gpu }
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
                ? "the ANE package is not installed (build it: make ane-package; see docs/ANE_EMBEDDER.md)"
                : "Core ML is not available on this platform"
            return Decision(directory: requestedDir, fallbackDirectory: nil, device: .gpu,
                            warning: "embed_device = ane but \(why); serving embeddings on the GPU")
        }
    }

    /// What the setting resolves to right now, for the status endpoint and the startup banner.
    public static func effective(setting: EmbedDevice, packageInstalled: Bool, coreML: Bool) -> EmbedDevice {
        switch setting {
        case .gpu: return .gpu
        case .ane, .auto: return (coreML && packageInstalled) ? .ane : .gpu
        }
    }

    /// Directory of the first installed ANE package in `registry`, if any.
    public static func installedPackage(in registry: Registry) -> URL? {
        for n in packageNames where registry.hasModel(n) {
            let d = registry.modelPath(n)
            if ANEPackageManifest.exists(in: d) { return d }
        }
        return nil
    }
}
