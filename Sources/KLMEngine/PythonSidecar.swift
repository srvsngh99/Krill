import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Shared plumbing for KrillLM's Python-sidecar bridges.
///
/// WS5 retired the Qwen 2.5-VL bridge (the family is now native
/// Swift+MLX). `MoEEngine` is the remaining sidecar - until the
/// native MoE runtime fully replaces it - so the generic pieces
/// (`LineReader`, the error type, the venv interpreter path) live
/// here rather than in a family-specific engine file.
public enum PythonSidecar {
    /// Default interpreter for KrillLM's bundled sidecar venv. The
    /// installer creates `~/.krillm/venv`; individual engines layer
    /// their own env override (e.g. `KRILLM_MOE_PYTHON`) on top.
    public static var defaultVenvPython: String {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".krillm/venv/bin/python3").path
    }
}

/// Errors raised by a Python-sidecar bridge.
public enum VLMError: Error, CustomStringConvertible {
    case notLoaded
    case bridgeNotReady(String)
    case bridgeCrashed(String)
    case generation(String)

    public var description: String {
        switch self {
        case .notLoaded:
            return "sidecar bridge not loaded"
        case .bridgeNotReady(let m):
            return "sidecar bridge failed to start: \(m)"
        case .bridgeCrashed(let m):
            return "sidecar bridge crashed: \(m)"
        case .generation(let m):
            return "sidecar bridge generation failed: \(m)"
        }
    }
}

/// Minimal line-buffered reader over a `FileHandle`. Uses
/// `poll(2)` to enforce a per-call deadline so an alive-but-silent
/// bridge (e.g. an mlx deadlock) does not hang the server's
/// generation queue indefinitely.
final class LineReader {
    private let handle: FileHandle
    private var buffer: Data = Data()

    init(handle: FileHandle) {
        self.handle = handle
    }

    /// Read one `\n`-terminated line. Returns nil on EOF.
    /// Throws `VLMError.bridgeCrashed` if no newline arrives
    /// within `timeout` seconds (the bridge is alive but silent).
    func readLine(timeout: TimeInterval) throws -> String? {
        let deadline = Date().addingTimeInterval(timeout)
        while true {
            if let nlIdx = buffer.firstIndex(of: 0x0a) {
                let lineData = buffer[..<nlIdx]
                buffer.removeSubrange(0 ..< (nlIdx + 1))
                return String(data: Data(lineData), encoding: .utf8) ?? ""
            }
            // Block until the fd is readable OR the deadline
            // expires. `poll(2)` is preferred over `select(2)`
            // (no fd_set size cap) and is available on macOS.
            let remaining = deadline.timeIntervalSinceNow
            if remaining <= 0 {
                throw VLMError.bridgeCrashed(
                    "bridge produced no output within \(Int(timeout))s")
            }
            var pfd = pollfd(
                fd: handle.fileDescriptor,
                events: Int16(POLLIN), revents: 0)
            let timeoutMs = Int32(min(remaining * 1000, Double(Int32.max)))
            let rc = poll(&pfd, 1, timeoutMs)
            if rc == 0 {
                throw VLMError.bridgeCrashed(
                    "bridge produced no output within \(Int(timeout))s")
            }
            if rc < 0 {
                if errno == EINTR { continue }
                throw VLMError.bridgeCrashed(
                    "poll(2) on bridge stdout failed: errno=\(errno)")
            }
            let chunk = handle.availableData
            if chunk.isEmpty {
                return nil
            }
            buffer.append(chunk)
        }
    }
}
