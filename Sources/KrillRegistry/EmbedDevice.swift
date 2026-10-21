import Foundation

/// Where the EmbeddingGemma 2 text embedder runs.
///
/// - `gpu`: the MLX path (default; the behaviour before the Neural Engine backend existed).
/// - `ane`: the Core ML Neural Engine package serves EG2 text requests; text over the
///   largest bucket and every image/audio/video input still go to the MLX path.
/// - `auto`: `ane` when this platform has Core ML and the ANE package is installed,
///   otherwise `gpu`.
///
/// Set with `embed_device` in `config.toml` or `KRILL_EMBED_DEVICE`; read at server
/// start (a change needs a restart). See docs/ANE_EMBEDDER.md.
public enum EmbedDevice: String, Sendable, CaseIterable, Equatable {
    case gpu, ane, auto

    public static let envVar = "KRILL_EMBED_DEVICE"
    public static let configKey = "embed_device"

    /// Case-insensitive, whitespace-tolerant; nil for anything else.
    public static func parse(_ raw: String?) -> EmbedDevice? {
        guard let raw else { return nil }
        let v = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return EmbedDevice(rawValue: v)
    }

    public static var validValues: String { allCases.map(\.rawValue).joined(separator: " | ") }
}
