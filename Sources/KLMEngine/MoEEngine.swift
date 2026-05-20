import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Compatible-fallback runtime for mixture-of-experts text models
/// (Mixtral, Qwen3-MoE, Qwen2-MoE, OLMoE, DeepSeek-V3, etc.).
/// Spawns a long-lived Python sidecar (`tools/moe_bridge.py`
/// running under `~/.krillm/venv/bin/python3`) and forwards text
/// requests over stdin/stdout. mlx-lm handles router + expert-FFN
/// dispatch natively, so this bridge is intentionally
/// architecture-agnostic.
///
/// This is NOT a native Swift+MLX router. It exists so MoE models
/// are usable today as a compatible-fallback tier
/// (`SupportTier.compatibleFallback`); the registry advertises
/// the limitation. The native port (router weights, top-K expert
/// selection on Metal, expert FFN dispatch, memory policy) is a
/// follow-up to this PR.
///
/// Structurally identical to `Qwen25VLEngine` (same JSON
/// protocol, same poll(2)-bounded LineReader, same SIGINT
/// shutdown contract). The two engines are kept separate rather
/// than refactored into a shared base because their underlying
/// Python deps differ (`mlx_lm` vs `mlx_vlm`) and the request
/// shape differs (no image path here).
public final class MoEEngine: @unchecked Sendable {
    public static var defaultPython: String {
        ProcessInfo.processInfo.environment["KRILLM_MOE_PYTHON"]
            ?? Qwen25VLEngine.defaultPython
    }

    public static func defaultBridgeScript() -> String {
        if let override = ProcessInfo.processInfo.environment["KRILLM_MOE_BRIDGE"] {
            return override
        }
        let exe = URL(fileURLWithPath: CommandLine.arguments[0])
            .deletingLastPathComponent()
        let candidates = [
            exe.appendingPathComponent("moe_bridge.py"),
            exe.appendingPathComponent("../tools/moe_bridge.py"),
            URL(fileURLWithPath: "tools/moe_bridge.py"),
        ]
        for c in candidates where FileManager.default.fileExists(atPath: c.path) {
            return c.path
        }
        return "tools/moe_bridge.py"
    }

    private var process: Process?
    private var stdin: FileHandle?
    private var stdoutReader: LineReader?
    private var loadedDir: URL?
    private var nextRequestId: Int = 1
    private let lock = NSLock()
    /// Held for the entire write+read cycle on the half-duplex
    /// sidecar stdio so two concurrent generate calls cannot
    /// corrupt each other.
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

    /// Spawn the sidecar and wait for the `{"ready": true}` ack.
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
        proc.standardError = FileHandle.standardError

        try proc.run()
        let reader = LineReader(handle: outPipe.fileHandleForReading)
        // MoE checkpoints are large (Mixtral 8x7B is ~25GB even at
        // 4-bit); allow up to 5 minutes for the first load.
        guard let firstLine = try reader.readLine(timeout: 300) else {
            proc.terminate()
            throw VLMError.bridgeNotReady("MoE bridge produced no output")
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

    /// Forward a (messages, max_tokens) request. Serialized across
    /// concurrent callers.
    public func generate(
        messages: [[String: String]], maxTokens: Int = 256
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

        let req: [String: Any] = [
            "id": id, "messages": messages, "max_tokens": maxTokens,
        ]
        let data = try JSONSerialization.data(withJSONObject: req)
        try stdin.write(contentsOf: data + Data("\n".utf8))

        var fullText = ""
        var promptTokens = 0
        var completionTokens = 0
        while true {
            guard let line = try reader.readLine(timeout: 600) else {
                throw VLMError.bridgeCrashed("MoE bridge closed stdout mid-stream")
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
        let proc = withLock { () -> Process? in
            let p = process
            self.process = nil
            self.stdin?.closeFile()
            self.stdin = nil
            self.stdoutReader = nil
            self.loadedDir = nil
            return p
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
            throw VLMError.bridgeCrashed("MoE bridge emitted non-JSON: \(line)")
        }
        return obj
    }
}
