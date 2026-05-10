import Foundation

/// Python mlx-vlm fallback for complex models (Gemma 4) where native Swift
/// implementation is still being debugged.
///
/// Talks to a long-running Python helper process (`tools/mlx_vlm_sidecar.py`)
/// over line-delimited JSON on stdin/stdout. The helper loads the model once
/// and answers many requests, avoiding per-call Python and mlx-vlm import
/// costs.
///
/// Python resolution order:
///   1. `KRILLM_PYTHON` environment variable (explicit override)
///   2. `~/.krillm/venv/bin/python3` (managed venv)
///   3. System `python3` via `/usr/bin/env`
public final class PythonFallback: @unchecked Sendable {

    private let modelPath: String

    public init(modelPath: String) {
        self.modelPath = modelPath
    }

    public struct Availability: Equatable, Sendable {
        public let pythonCommand: String
        public let isAvailable: Bool
        public let detail: String
    }

    /// Candidate venv paths to search, in priority order.
    private static let venvSearchPaths: [String] = {
        var paths: [String] = []
        if let envPython = ProcessInfo.processInfo.environment["KRILLM_PYTHON"] {
            paths.append(envPython)
        }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        paths.append("\(home)/.krillm/venv/bin/python3")
        paths.append("\(home)/.venv/bin/python3")
        paths.append("/opt/homebrew/bin/python3")
        return paths
    }()

    /// Resolved Python executable path. Checks venvs first, falls back to system python3.
    private static let pythonPath: String? = {
        for path in venvSearchPaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        return "/usr/bin/env"
    }()

    private static let pythonArgs: [String] = {
        for path in venvSearchPaths {
            if FileManager.default.fileExists(atPath: path) {
                return []
            }
        }
        return ["python3"]
    }()

    /// Check if Python mlx-vlm is available.
    public static var isAvailable: Bool {
        checkAvailability().isAvailable
    }

    /// Check Python and mlx-vlm availability, including a human-readable reason.
    public static func checkAvailability() -> Availability {
        guard let python = pythonPath else {
            return Availability(
                pythonCommand: "",
                isAvailable: false,
                detail: "No Python executable found")
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = pythonArgs + ["-c", "from mlx_vlm import load; print('ok')"]
        process.standardOutput = FileHandle.nullDevice
        let errPipe = Pipe()
        process.standardError = errPipe
        do {
            try process.run()
        } catch {
            return Availability(
                pythonCommand: ([python] + pythonArgs).joined(separator: " "),
                isAvailable: false,
                detail: "Unable to run Python: \(error.localizedDescription)")
        }
        process.waitUntilExit()
        if process.terminationStatus == 0 {
            return Availability(
                pythonCommand: ([python] + pythonArgs).joined(separator: " "),
                isAvailable: true,
                detail: "mlx-vlm available")
        }
        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
        let err = String(data: errData, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let detail = err?.isEmpty == false
            ? err!
            : "Python mlx-vlm not installed (pip install mlx-vlm)"
        return Availability(
            pythonCommand: ([python] + pythonArgs).joined(separator: " "),
            isAvailable: false,
            detail: detail)
    }

    /// Generate text using Python mlx-vlm. Routes through the persistent
    /// helper for this model path, spawning one if necessary.
    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 512,
        imagePath: String? = nil,
        audioPath: String? = nil
    ) async throws -> String {
        let composed: String
        if let sys = systemPrompt, !sys.isEmpty {
            composed = sys + "\n\n" + prompt
        } else {
            composed = prompt
        }
        let helper = try await HelperRegistry.shared.helper(forModelPath: modelPath)
        return try await helper.send(
            prompt: composed,
            maxTokens: maxTokens,
            imagePath: imagePath,
            audioPath: audioPath)
    }

    // MARK: - Helper script resolution

    /// Resolves `tools/mlx_vlm_sidecar.py`. Honors `KLM_MLX_VLM_SIDECAR`
    /// override, otherwise walks up from the current working directory
    /// looking for `tools/mlx_vlm_sidecar.py`.
    static func resolveHelperScriptPath() -> String? {
        if let override = ProcessInfo.processInfo.environment["KLM_MLX_VLM_SIDECAR"],
           !override.isEmpty,
           FileManager.default.fileExists(atPath: override) {
            return override
        }
        let fm = FileManager.default
        var dir = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
        for _ in 0..<8 {
            let candidate = dir.appendingPathComponent("tools/mlx_vlm_sidecar.py").path
            if fm.fileExists(atPath: candidate) {
                return candidate
            }
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path { break }
            dir = parent
        }
        return nil
    }

    fileprivate static func helperResolutionError() -> FallbackError {
        let cwd = FileManager.default.currentDirectoryPath
        let env = ProcessInfo.processInfo.environment["KLM_MLX_VLM_SIDECAR"] ?? "<unset>"
        return .helperNotFound(
            "Could not locate tools/mlx_vlm_sidecar.py. " +
            "Searched up from cwd=\(cwd); KLM_MLX_VLM_SIDECAR=\(env). " +
            "Set KLM_MLX_VLM_SIDECAR to the absolute path of the helper script.")
    }
}

public enum FallbackError: Error, CustomStringConvertible {
    case pythonFailed(String)
    case notAvailable
    case helperNotFound(String)
    case helperCrashed(String)
    case protocolError(String)

    public var description: String {
        switch self {
        case .pythonFailed(let msg): return "Python fallback failed: \(msg.prefix(2000))"
        case .notAvailable: return "Python mlx-vlm not installed (pip install mlx-vlm)"
        case .helperNotFound(let msg): return msg
        case .helperCrashed(let msg): return "Python sidecar crashed: \(msg.prefix(2000))"
        case .protocolError(let msg): return "Python sidecar protocol error: \(msg.prefix(2000))"
        }
    }
}

// MARK: - Helper registry

/// Registry of long-running helper processes, keyed by model path. Spawn
/// requests for the same model path coalesce onto a single in-flight task to
/// prevent concurrent callers from each spawning their own helper.
actor HelperRegistry {
    static let shared = HelperRegistry()

    private var helpers: [String: PythonHelper] = [:]
    private var inflight: [String: Task<PythonHelper, Error>] = [:]

    func helper(forModelPath modelPath: String) async throws -> PythonHelper {
        if let existing = helpers[modelPath], await existing.isAlive {
            return existing
        }
        if let task = inflight[modelPath] {
            return try await task.value
        }
        let task = Task<PythonHelper, Error> {
            try await PythonHelper.spawn(modelPath: modelPath)
        }
        inflight[modelPath] = task
        defer { inflight.removeValue(forKey: modelPath) }
        let helper = try await task.value
        helpers[modelPath] = helper
        return helper
    }

    /// Test hook: drop the cached helper for a given model path so the next
    /// call respawns. Used by tests to simulate restart-after-crash.
    func reset(modelPath: String) async {
        if let h = helpers.removeValue(forKey: modelPath) {
            await h.terminate()
        }
    }

    func resetAll() async {
        let all = helpers
        helpers.removeAll()
        for (_, h) in all { await h.terminate() }
    }
}

// MARK: - PythonHelper

/// One persistent Python sidecar process. Serializes requests through stdin
/// and dispatches stdout responses to the matching pending continuation by id.
actor PythonHelper {

    private let process: Process
    private let stdinHandle: FileHandle
    private let stdoutHandle: FileHandle
    private var pending: [String: CheckedContinuation<String, Error>] = [:]
    private var nextId: UInt64 = 0
    private var alive: Bool = true

    var isAlive: Bool { alive }

    private init(process: Process, stdinHandle: FileHandle, stdoutHandle: FileHandle) {
        self.process = process
        self.stdinHandle = stdinHandle
        self.stdoutHandle = stdoutHandle
    }

    static func spawn(modelPath: String) async throws -> PythonHelper {
        guard let scriptPath = PythonFallback.resolveHelperScriptPath() else {
            throw PythonFallback.helperResolutionError()
        }
        guard let python = pythonExecutable() else {
            throw FallbackError.notAvailable
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = pythonArguments() + [scriptPath, "--model-path", modelPath]

        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr

        // Drain stderr so it never blocks the helper. Buffer the tail for
        // diagnostics if the process dies.
        let stderrBuffer = StderrBuffer()
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty { return }
            stderrBuffer.append(chunk)
        }

        do {
            try process.run()
        } catch {
            throw FallbackError.helperCrashed("failed to spawn: \(error.localizedDescription)")
        }

        // Wait for the READY line on stderr before returning. Time out so
        // a stuck helper doesn't hang the caller forever.
        try await stderrBuffer.waitForReady(timeout: 60)

        let helper = PythonHelper(
            process: process,
            stdinHandle: stdin.fileHandleForWriting,
            stdoutHandle: stdout.fileHandleForReading)
        await helper.startReader(stderrBuffer: stderrBuffer)
        return helper
    }

    private func startReader(stderrBuffer: StderrBuffer) {
        let handle = self.stdoutHandle
        let lineSink = LineSink { [weak self] line in
            guard let self = self else { return }
            Task { await self.dispatch(line: line) }
        } onEOF: { [weak self] in
            guard let self = self else { return }
            let tail = stderrBuffer.tail()
            Task { await self.handleEOF(reason: "EOF on helper stdout", stderrTail: tail) }
        }
        handle.readabilityHandler = { fh in
            let chunk = fh.availableData
            if chunk.isEmpty {
                lineSink.eof()
                fh.readabilityHandler = nil
                return
            }
            lineSink.feed(chunk)
        }
    }

    private func dispatch(line: String) {
        guard let data = line.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let id = obj["id"] as? String else {
            // Unparseable response; surface to all pending callers and bail.
            failAllPending(reason: "unparseable response from helper: \(line.prefix(200))")
            return
        }
        guard let cont = pending.removeValue(forKey: id) else {
            // Stale response; ignore.
            return
        }
        if let ok = obj["ok"] as? Bool, ok {
            let output = obj["output"] as? String ?? ""
            cont.resume(returning: output)
        } else {
            let err = obj["error"] as? String ?? "unknown error"
            cont.resume(throwing: FallbackError.pythonFailed(err))
        }
    }

    private func handleEOF(reason: String, stderrTail: String) {
        alive = false
        let detail = stderrTail.isEmpty ? reason : "\(reason); stderr: \(stderrTail)"
        failAllPending(reason: detail)
    }

    private func failAllPending(reason: String) {
        let snapshot = pending
        pending.removeAll()
        for (_, cont) in snapshot {
            cont.resume(throwing: FallbackError.helperCrashed(reason))
        }
    }

    func send(
        prompt: String,
        maxTokens: Int,
        imagePath: String?,
        audioPath: String?
    ) async throws -> String {
        guard alive else {
            throw FallbackError.helperCrashed("helper is not alive")
        }
        nextId &+= 1
        let id = "req-\(nextId)"
        var payload: [String: Any] = [
            "id": id,
            "prompt": prompt,
            "max_tokens": maxTokens,
            "image_path": imagePath as Any? ?? NSNull(),
            "audio_path": audioPath as Any? ?? NSNull(),
        ]
        // Replace Any? wrapping above with explicit nulls.
        payload["image_path"] = imagePath ?? (NSNull() as Any)
        payload["audio_path"] = audioPath ?? (NSNull() as Any)

        let data: Data
        do {
            data = try JSONSerialization.data(withJSONObject: payload, options: [])
        } catch {
            throw FallbackError.protocolError("failed to encode request: \(error.localizedDescription)")
        }

        return try await withCheckedThrowingContinuation { (cont: CheckedContinuation<String, Error>) in
            pending[id] = cont
            do {
                try stdinHandle.write(contentsOf: data)
                try stdinHandle.write(contentsOf: Data([0x0A]))
            } catch {
                pending.removeValue(forKey: id)
                alive = false
                cont.resume(throwing: FallbackError.helperCrashed(
                    "failed to write request: \(error.localizedDescription)"))
            }
        }
    }

    func terminate() {
        alive = false
        stdoutHandle.readabilityHandler = nil
        try? stdinHandle.close()
        if process.isRunning {
            process.terminate()
        }
        failAllPending(reason: "helper terminated")
    }

    private static func pythonExecutable() -> String? {
        // Mirror PythonFallback's resolution.
        if let env = ProcessInfo.processInfo.environment["KRILLM_PYTHON"],
           FileManager.default.fileExists(atPath: env) {
            return env
        }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let candidates = [
            "\(home)/.krillm/venv/bin/python3",
            "\(home)/.venv/bin/python3",
            "/opt/homebrew/bin/python3",
        ]
        for c in candidates {
            if FileManager.default.fileExists(atPath: c) { return c }
        }
        return "/usr/bin/env"
    }

    private static func pythonArguments() -> [String] {
        if let env = ProcessInfo.processInfo.environment["KRILLM_PYTHON"],
           FileManager.default.fileExists(atPath: env) {
            return []
        }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let candidates = [
            "\(home)/.krillm/venv/bin/python3",
            "\(home)/.venv/bin/python3",
            "/opt/homebrew/bin/python3",
        ]
        for c in candidates {
            if FileManager.default.fileExists(atPath: c) { return [] }
        }
        return ["python3"]
    }
}

// MARK: - LineSink

/// Thread-safe accumulator that splits an incoming byte stream into UTF-8
/// lines and forwards each completed line to a callback. Used by the helper
/// reader to feed stdout chunks (delivered on Dispatch threads by
/// `FileHandle.readabilityHandler`) to the actor.
final class LineSink: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = Data()
    private var done = false
    private let onLine: @Sendable (String) -> Void
    private let onEOF: @Sendable () -> Void

    init(onLine: @escaping @Sendable (String) -> Void,
         onEOF: @escaping @Sendable () -> Void) {
        self.onLine = onLine
        self.onEOF = onEOF
    }

    func feed(_ chunk: Data) {
        var lines: [String] = []
        lock.lock()
        if done { lock.unlock(); return }
        buffer.append(chunk)
        while let nl = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer.subdata(in: 0..<nl)
            buffer.removeSubrange(0...nl)
            if let s = String(data: lineData, encoding: .utf8),
               !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                lines.append(s)
            }
        }
        lock.unlock()
        for l in lines { onLine(l) }
    }

    func eof() {
        lock.lock()
        if done { lock.unlock(); return }
        done = true
        lock.unlock()
        onEOF()
    }
}

// MARK: - StderrBuffer

/// Thread-safe buffer for the helper's stderr. Detects the `READY` token
/// emitted by the sidecar after model load completes, and retains a small
/// tail for crash diagnostics.
final class StderrBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer: String = ""
    private var ready: Bool = false
    private var readyContinuations: [CheckedContinuation<Void, Error>] = []

    func append(_ chunk: Data) {
        guard let s = String(data: chunk, encoding: .utf8) else { return }
        lock.lock()
        buffer.append(s)
        if buffer.count > 16_384 {
            buffer = String(buffer.suffix(16_384))
        }
        let nowReady = !ready && buffer.range(of: "READY") != nil
        if nowReady {
            ready = true
        }
        let conts = nowReady ? readyContinuations : []
        if nowReady { readyContinuations.removeAll() }
        lock.unlock()
        for c in conts { c.resume() }
    }

    func tail() -> String {
        lock.lock()
        defer { lock.unlock() }
        return String(buffer.suffix(2000))
    }

    private func registerReadyContinuationOrResolve(
        _ cont: CheckedContinuation<Void, Error>
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if ready { return true }
        readyContinuations.append(cont)
        return false
    }

    private func isReady() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return ready
    }

    func waitForReady(timeout: TimeInterval) async throws {
        if isReady() { return }
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { [self] in
                try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                    if self.registerReadyContinuationOrResolve(cont) {
                        cont.resume()
                    }
                }
            }
            group.addTask { [self] in
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                throw FallbackError.helperCrashed(
                    "helper did not emit READY within \(Int(timeout))s; stderr: \(self.tail())")
            }
            _ = try await group.next()
            group.cancelAll()
        }
    }
}
