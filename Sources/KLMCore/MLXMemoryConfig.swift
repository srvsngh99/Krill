import Foundation
import MLX

/// Bounds the MLX Metal buffer-recycling pool so it does not inflate the
/// process `phys_footprint` during inference.
///
/// MLX keeps freed intermediate buffers in a recycling pool instead of
/// returning them to the OS, sized by Metal's `recommendedMaxWorkingSetSize`
/// (≈16 GB on a 24 GB M4 Pro). Those cached pages are resident and counted by
/// `phys_footprint` / `RSIZE` — the exact figure the release benchmark samples
/// for `memory_ratio` — even though MLX considers them "free". Left
/// unconstrained the pool can grow into the multi-GB range over a run.
///
/// Capping the cache forces MLX to release recycled buffers back to the OS on
/// deallocation while still allowing reuse of the small, fixed-size per-token
/// decode buffers (well under the cap), so decode throughput is preserved.
public enum MLXMemoryConfig {
    /// Environment override for the cache cap, in megabytes.
    ///
    /// - Unset: use ``defaultCacheLimitMB``.
    /// - `0`: disable the cap entirely (legacy unbounded MLX behavior).
    /// - Positive integer: cap the recycling pool at that many MB.
    public static let envVar = "KRILL_MLX_CACHE_LIMIT_MB"

    /// Default cap when the environment variable is unset.
    ///
    /// 256 MB comfortably covers Gemma 4 e2b's fixed-size decode-step buffers
    /// (hidden states, single-token logits, attention scratch) so the hot loop
    /// still recycles rather than re-allocates, while preventing the pool from
    /// growing into the multi-GB range that dominated `phys_footprint`.
    public static let defaultCacheLimitMB = 256

    /// Pure resolution of the configured cap from an environment, in MB.
    ///
    /// Returns `nil` when the cap is explicitly disabled (`...=0`). Invalid
    /// values fall back to ``defaultCacheLimitMB`` and emit a warning.
    /// No MLX/Metal side effects — safe to unit test.
    public static func resolveCacheLimitMB(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        guard let raw = environment[envVar]?.trimmingCharacters(in: .whitespaces),
              !raw.isEmpty else {
            return defaultCacheLimitMB
        }
        guard let parsed = Int(raw), parsed >= 0 else {
            FileHandle.standardError.write(Data(
                "[KrillLM] ignoring invalid \(envVar)=\(raw); using default \(defaultCacheLimitMB) MB\n".utf8))
            return defaultCacheLimitMB
        }
        return parsed == 0 ? nil : parsed
    }

    /// Resolve and apply the cache limit to the MLX runtime. Idempotent; safe
    /// to call on every model load. Returns the limit applied in bytes, or
    /// `nil` if the cap was explicitly disabled.
    @discardableResult
    public static func apply(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        guard let mb = resolveCacheLimitMB(environment: environment) else {
            // Explicit opt-out: leave MLX's default (device-scaled) cache limit.
            return nil
        }
        let bytes = mb * 1024 * 1024
        MLX.Memory.cacheLimit = bytes
        return bytes
    }
}
