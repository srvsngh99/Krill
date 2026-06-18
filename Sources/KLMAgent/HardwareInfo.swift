import Foundation

/// A snapshot of the user's machine, captured once per agent session.
///
/// The struct is plain data (no methods that re-shell out) so it can be
/// embedded verbatim in the system prompt as JSON, persisted in a
/// fixture for tests, or passed across `Sendable` boundaries.
public struct HardwareInfo: Equatable, Sendable, Codable {
    /// CPU architecture (`arm64` on Apple Silicon, `x86_64` on Intel Mac).
    public let arch: String
    /// Marketing brand string (e.g. `Apple M4 Pro`).
    public let chip: String
    /// Total physical RAM in bytes.
    public let totalRAMBytes: UInt64
    /// Free RAM in bytes at the moment of capture (free + inactive pages).
    public let freeRAMBytes: UInt64
    /// Logical CPU core count.
    public let cpuCores: Int
    /// GPU core count when known. `nil` on Intel Macs where the integer
    /// is meaningless / on parse failure.
    public let gpuCores: Int?
    /// Free disk space in bytes on the registry partition (~/.krill).
    public let freeDiskBytes: UInt64
    /// `true` iff arch is `arm64`; serves as a proxy for "the MLX/Metal
    /// fast path is available".
    public let metalAvailable: Bool

    public init(
        arch: String, chip: String,
        totalRAMBytes: UInt64, freeRAMBytes: UInt64,
        cpuCores: Int, gpuCores: Int?,
        freeDiskBytes: UInt64, metalAvailable: Bool
    ) {
        self.arch = arch
        self.chip = chip
        self.totalRAMBytes = totalRAMBytes
        self.freeRAMBytes = freeRAMBytes
        self.cpuCores = cpuCores
        self.gpuCores = gpuCores
        self.freeDiskBytes = freeDiskBytes
        self.metalAvailable = metalAvailable
    }

    /// RAM in gigabytes (1024^3), rounded to one decimal place.
    public var totalRAMGB: Double {
        Double(totalRAMBytes) / 1_073_741_824.0
    }

    public var freeRAMGB: Double {
        Double(freeRAMBytes) / 1_073_741_824.0
    }

    public var freeDiskGB: Double {
        Double(freeDiskBytes) / 1_073_741_824.0
    }
}

/// Classification of how comfortably a given model fits on a given
/// machine. Tier rules live in `HardwareInfo.classifyFit`.
public enum FitClassification: String, Equatable, Sendable, Codable {
    case comfortable
    case tight
    case risky
    case wontFit = "wont_fit"

    /// Human-readable label shown next to a model in the recommender
    /// shortlist and in `model_info` output.
    public var label: String {
        switch self {
        case .comfortable: return "comfortable fit"
        case .tight: return "tight fit"
        case .risky: return "risky fit"
        case .wontFit: return "won't fit"
        }
    }

    /// Severity tier consumed by `OperatorEvent.warning`.
    public var severity: WarningSeverity {
        switch self {
        case .comfortable: return .info
        case .tight: return .warn
        case .risky: return .risky
        case .wontFit: return .wontFit
        }
    }
}

public extension HardwareInfo {
    /// Capture the current machine state by shelling out to sysctl /
    /// uname / vm_stat / system_profiler / statfs. Each individual probe
    /// falls back to a conservative default on failure rather than
    /// throwing - the operator agent works degraded rather than refuses
    /// to launch on a weird host.
    static func current() -> HardwareInfo {
        let probe = SysctlProbe.posix
        return HardwareInfo(
            arch: probe.arch(),
            chip: probe.chip(),
            totalRAMBytes: probe.totalRAMBytes(),
            freeRAMBytes: probe.freeRAMBytes(),
            cpuCores: probe.cpuCores(),
            gpuCores: probe.gpuCores(),
            freeDiskBytes: probe.freeDiskBytes(
                Self.defaultRegistryPath()),
            metalAvailable: probe.arch() == "arm64"
        )
    }

    /// Classify how a model of `sizeBytes` (the on-disk size, typically
    /// `ModelManifest.sizeBytes` or a `CatalogEntry`-derived estimate)
    /// fits on this hardware.
    ///
    /// The headroom heuristic (§2.4): a 4-bit dense LM of `m` GB needs
    /// roughly `m * 1.6` GB at runtime once KV cache, Metal pool, and OS
    /// headroom are factored in. Vision/audio inflate that further; the
    /// recommender can pass an inflated `sizeBytes` for those families
    /// rather than baking family-specific factors here.
    func classifyFit(modelSizeBytes: UInt64) -> FitClassification {
        let m = Double(modelSizeBytes) / 1_073_741_824.0
        let r = totalRAMGB
        guard r > 0 else { return .wontFit }
        let need = m * 1.6
        if need < r * 0.5 { return .comfortable }
        if need < r * 0.8 { return .tight }
        if need < r * 0.95 { return .risky }
        return .wontFit
    }

    static func defaultRegistryPath() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".krill")
    }
}

/// Indirection over the four shell-outs so unit tests can swap in
/// canned outputs without launching `sysctl` for real.
internal struct SysctlProbe: Sendable {
    let arch: @Sendable () -> String
    let chip: @Sendable () -> String
    let totalRAMBytes: @Sendable () -> UInt64
    let freeRAMBytes: @Sendable () -> UInt64
    let cpuCores: @Sendable () -> Int
    let gpuCores: @Sendable () -> Int?
    let freeDiskBytes: @Sendable (URL) -> UInt64

    static let posix = SysctlProbe(
        arch: {
            if let s = SysctlProbe.runShell(
                "/usr/bin/uname", ["-m"])?.trimmingCharacters(
                    in: .whitespacesAndNewlines), !s.isEmpty {
                return s
            }
            return "unknown"
        },
        chip: {
            SysctlProbe.runShell(
                "/usr/sbin/sysctl", ["-n", "machdep.cpu.brand_string"]
            )?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "unknown"
        },
        totalRAMBytes: {
            UInt64(SysctlProbe.runShell(
                "/usr/sbin/sysctl", ["-n", "hw.memsize"]
            )?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "") ?? 0
        },
        freeRAMBytes: {
            parseVMStatFree(SysctlProbe.runShell(
                "/usr/bin/vm_stat", []) ?? "")
        },
        cpuCores: {
            Int(SysctlProbe.runShell(
                "/usr/sbin/sysctl", ["-n", "hw.ncpu"]
            )?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "") ?? 0
        },
        gpuCores: {
            parseGPUCores(SysctlProbe.runShell(
                "/usr/sbin/system_profiler", ["SPDisplaysDataType"]) ?? "")
        },
        freeDiskBytes: { url in
            // statfs at the *parent* directory when the path itself does
            // not exist yet (a fresh install before `~/.krill` is created).
            let path = FileManager.default.fileExists(atPath: url.path)
                ? url.path
                : url.deletingLastPathComponent().path
            guard let attrs = try? FileManager.default
                .attributesOfFileSystem(forPath: path),
                  let free = attrs[.systemFreeSize] as? NSNumber
            else { return 0 }
            return free.uint64Value
        }
    )

    /// `vm_stat` output reports page counts; the page size is reported
    /// in the header. Sums free + inactive pages × page size.
    static func parseVMStatFree(_ output: String) -> UInt64 {
        var pageSize: UInt64 = 4096
        var freePages: UInt64 = 0
        var inactivePages: UInt64 = 0
        for line in output.split(separator: "\n") {
            let l = String(line)
            if l.contains("page size of"),
               let n = extractFirstInteger(in: l) {
                pageSize = n
            } else if l.hasPrefix("Pages free:"),
                      let n = extractFirstInteger(in: l) {
                freePages = n
            } else if l.hasPrefix("Pages inactive:"),
                      let n = extractFirstInteger(in: l) {
                inactivePages = n
            }
        }
        return (freePages + inactivePages) * pageSize
    }

    /// Parse "Total Number of Cores: N" out of `system_profiler
    /// SPDisplaysDataType`. Returns nil if no match.
    static func parseGPUCores(_ output: String) -> Int? {
        for line in output.split(separator: "\n") {
            let l = String(line).trimmingCharacters(in: .whitespaces)
            if l.hasPrefix("Total Number of Cores:") {
                let v = l.dropFirst("Total Number of Cores:".count)
                    .trimmingCharacters(in: .whitespaces)
                if let n = Int(v) { return n }
            }
        }
        return nil
    }

    private static func extractFirstInteger(in s: String) -> UInt64? {
        var digits = ""
        for ch in s {
            if ch.isNumber { digits.append(ch) }
            else if !digits.isEmpty { break }
        }
        return UInt64(digits)
    }

    private static func runShell(_ path: String, _ args: [String]) -> String? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        // Drop stderr to /dev/null rather than an undrained Pipe(): the
        // OS pipe buffer is ~64 KB, and if the subprocess writes more
        // than that to stderr without anyone reading, the subprocess
        // blocks on write and waitUntilExit deadlocks. None of the
        // sysctl/uname/vm_stat/system_profiler/statfs probes need
        // stderr.
        p.standardError = FileHandle.nullDevice
        do { try p.run() } catch { return nil }
        // Drain stdout BEFORE waitUntilExit. readDataToEndOfFile reads
        // until the write end closes (when the subprocess exits), so
        // the pipe is continuously drained instead of filling. The
        // earlier "wait, then read" ordering deadlocked any probe
        // whose stdout exceeded the pipe buffer (real risk:
        // system_profiler on multi-display / eGPU hosts).
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return String(data: data, encoding: .utf8)
    }
}
