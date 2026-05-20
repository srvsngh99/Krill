import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Compatible-fallback runtime for Qwen 2.5-VL. Spawns a long-lived
/// Python sidecar (`tools/qwen25vl_bridge.py` running under
/// `~/.krillm/venv/bin/python3`) and forwards one (text, image)
/// request per turn over stdin/stdout. This is the same bridge
/// shape Gemma 4 audio used before WS1 landed the native path.
///
/// This is NOT the native Swift+MLX vision tower. It exists so
/// Qwen 2.5-VL is usable today as a compatible-fallback tier
/// (`SupportTier.compatibleFallback`) - the registry advertises
/// the limitation. The native port (custom mRoPE, window-
/// attention vision tower, patch merger, masked embedding
/// injection) is a follow-up to this PR.
public final class Qwen25VLEngine: @unchecked Sendable {
    /// Default mlx-vlm venv interpreter shipped with KrillLM's
    /// installer. Override with `KRILLM_VLM_PYTHON` for development
    /// or to point at a custom mlx-vlm install.
    public static var defaultPython: String {
        ProcessInfo.processInfo.environment["KRILLM_VLM_PYTHON"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".krillm/venv/bin/python3").path
    }

    /// Path to `tools/qwen25vl_bridge.py` shipped alongside the
    /// `krillm` binary. Override with `KRILLM_VLM_BRIDGE` to point
    /// at a development copy.
    public static func defaultBridgeScript() -> String {
        if let override = ProcessInfo.processInfo.environment["KRILLM_VLM_BRIDGE"] {
            return override
        }
        // Resolve relative to the binary so installed builds find
        // the bridge next to themselves. Falls back to the repo
        // tree for `swift run` / `swift test`.
        let exe = URL(fileURLWithPath: CommandLine.arguments[0])
            .deletingLastPathComponent()
        let candidates = [
            exe.appendingPathComponent("qwen25vl_bridge.py"),
            exe.appendingPathComponent("../tools/qwen25vl_bridge.py"),
            URL(fileURLWithPath: "tools/qwen25vl_bridge.py"),
        ]
        for c in candidates where FileManager.default.fileExists(atPath: c.path) {
            return c.path
        }
        return "tools/qwen25vl_bridge.py"
    }

    private var process: Process?
    private var stdin: FileHandle?
    private var stdoutReader: LineReader?
    private var loadedDir: URL?
    private var nextRequestId: Int = 1
    private let lock = NSLock()
    /// Held for the entire duration of a `generate` call (write
    /// request + read response). The sidecar's stdin/stdout are a
    /// single half-duplex pipe; interleaving two concurrent
    /// requests would corrupt both. KLMServer's `genQueue` also
    /// serializes at the HTTP layer, but defending here too lets
    /// other callers (tests, CLI) be safe without re-implementing
    /// that queue.
    private let generateLock = NSLock()

    public init() {}

    public var loadedModelName: String? {
        withLock { loadedDir?.lastPathComponent }
    }

    public func isLoaded(directory: URL) -> Bool {
        withLock {
            loadedDir?.standardizedFileURL == directory.standardizedFileURL
                && process?.isRunning == true
        }
    }

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock(); defer { lock.unlock() }
        return body()
    }

    public struct GenerateResult: Sendable {
        public let text: String
        public let promptTokens: Int
        public let completionTokens: Int
    }

    /// Convenience wrapper for callers that have a single user
    /// prompt string. Equivalent to
    /// `generate(messages: [{role: user, content: prompt}], ...)`.
    public func generate(
        prompt: String, imagePath: String?, maxTokens: Int = 256
    ) throws -> GenerateResult {
        try generate(
            messages: [["role": "user", "content": prompt]],
            imagePath: imagePath, maxTokens: maxTokens)
    }

    /// Spawn the sidecar and wait for its `{"ready": true}` ack.
    public func load(directory: URL) async throws {
        if isLoaded(directory: directory) { return }
        try shutdown()

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: Self.defaultPython)
        proc.arguments = [Self.defaultBridgeScript(), "--model", directory.path]
        let inPipe = Pipe()
        let outPipe = Pipe()
        proc.standardInput = inPipe
        proc.standardOutput = outPipe
        // Forward stderr to our stderr so mlx-vlm load warnings
        // surface in the server log instead of vanishing.
        proc.standardError = FileHandle.standardError

        try proc.run()
        let reader = LineReader(handle: outPipe.fileHandleForReading)
        // Wait for the ready frame; bail out if the bridge dies
        // first.
        guard let firstLine = try reader.readLine(timeout: 60) else {
            proc.terminate()
            throw VLMError.bridgeNotReady("bridge produced no output")
        }
        let frame = try parseFrame(firstLine)
        if let err = frame["error"] as? String {
            proc.terminate()
            throw VLMError.bridgeNotReady(err)
        }
        guard frame["ready"] as? Bool == true else {
            proc.terminate()
            throw VLMError.bridgeNotReady("unexpected frame: \(firstLine)")
        }

        withLock {
            self.process = proc
            self.stdin = inPipe.fileHandleForWriting
            self.stdoutReader = reader
            self.loadedDir = directory
            self.nextRequestId = 1
        }
    }

    /// Send one (messages, image) request and read until `done`.
    /// Serialized via `generateLock` so concurrent callers do not
    /// interleave writes to the sidecar's half-duplex stdin/stdout.
    /// `messages` carries the full role-tagged chat history; the
    /// bridge renders it through Qwen 2.5-VL's chat template
    /// (system / user / assistant turns preserved).
    public func generate(
        messages: [[String: String]],
        imagePath: String?, maxTokens: Int = 256
    ) throws -> GenerateResult {
        generateLock.lock()
        defer { generateLock.unlock() }
        lock.lock()
        guard let stdin = self.stdin, let reader = self.stdoutReader else {
            lock.unlock()
            throw VLMError.notLoaded
        }
        let id = nextRequestId
        nextRequestId += 1
        lock.unlock()

        var req: [String: Any] = [
            "id": id, "messages": messages, "max_tokens": maxTokens,
        ]
        if let imagePath {
            req["image_path"] = imagePath
        } else {
            req["image_path"] = NSNull()
        }
        let data = try JSONSerialization.data(withJSONObject: req)
        try stdin.write(contentsOf: data + Data("\n".utf8))

        var fullText = ""
        var promptTokens = 0
        var completionTokens = 0
        while true {
            guard let line = try reader.readLine(timeout: 600) else {
                throw VLMError.bridgeCrashed("bridge closed stdout mid-stream")
            }
            let frame = try parseFrame(line)
            if let err = frame["error"] as? String {
                throw VLMError.generation(err)
            }
            if let token = frame["token"] as? String {
                fullText += token
            }
            if frame["done"] as? Bool == true {
                promptTokens = (frame["prompt_tokens"] as? Int) ?? 0
                completionTokens = (frame["completion_tokens"] as? Int) ?? 0
                break
            }
        }
        return GenerateResult(
            text: fullText,
            promptTokens: promptTokens,
            completionTokens: completionTokens)
    }

    public func shutdown() throws {
        let (proc, _) = withLock { () -> (Process?, Bool) in
            let p = process
            self.process = nil
            self.stdin?.closeFile()
            self.stdin = nil
            self.stdoutReader = nil
            self.loadedDir = nil
            return (p, true)
        }
        if let proc, proc.isRunning {
            proc.terminate()
            proc.waitUntilExit()
        }
    }

    deinit {
        try? shutdown()
    }

    private func parseFrame(_ line: String) throws -> [String: Any] {
        guard let data = line.data(using: .utf8),
              let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw VLMError.bridgeCrashed("bridge emitted non-JSON: \(line)")
        }
        return obj
    }
}

public enum VLMError: Error, CustomStringConvertible {
    case notLoaded
    case bridgeNotReady(String)
    case bridgeCrashed(String)
    case generation(String)

    public var description: String {
        switch self {
        case .notLoaded:
            return "Qwen 2.5-VL bridge not loaded"
        case .bridgeNotReady(let m):
            return "Qwen 2.5-VL bridge failed to start: \(m)"
        case .bridgeCrashed(let m):
            return "Qwen 2.5-VL bridge crashed: \(m)"
        case .generation(let m):
            return "Qwen 2.5-VL bridge generation failed: \(m)"
        }
    }
}

/// Minimal line-buffered reader over a `FileHandle`. Uses
/// `poll(2)` to enforce a per-call deadline so an alive-but-silent
/// bridge (e.g. an mlx-vlm deadlock) does not hang the server's
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
