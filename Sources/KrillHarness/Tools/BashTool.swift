import Darwin
import Foundation
import KrillTooling

/// Run a shell command and return its combined stdout+stderr.
///
/// PR2 scope: there is no permission layer yet, so this runs behind a caller-
/// supplied allow flag (the `krill code` command gates it). A real
/// allow/deny/ask permission model arrives in a later PR. Output is capped and
/// the command is killed after `timeout` seconds so a runaway command cannot
/// hang the loop.
public struct BashTool: Tool {
    public let name = "bash"
    public let description =
        "Run a shell command on the local machine and return its combined stdout and stderr."
    public let parametersJSON = """
    {"type":"object","properties":{"command":{"type":"string",\
    "description":"The shell command to run, e.g. 'ls -la' or 'echo hi'."}},\
    "required":["command"]}
    """

    /// Seconds before the command is force-terminated.
    public let timeout: TimeInterval
    /// Maximum bytes of output returned to the model (older bytes kept).
    public let maxOutputBytes: Int

    public init(timeout: TimeInterval = 30, maxOutputBytes: Int = 16_384) {
        self.timeout = timeout
        self.maxOutputBytes = maxOutputBytes
    }

    public func run(argumentsJSON: String) async -> ToolResult {
        guard
            let data = argumentsJSON.data(using: .utf8),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let command = obj["command"] as? String,
            !command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return ToolResult(
                content: "Error: bash requires a non-empty 'command' string argument.",
                isError: true)
        }
        // Offload the blocking Process work to a background thread so it never
        // blocks the cooperative executor (and so the synchronous semaphore wait
        // is legal).
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(returning: execute(command))
            }
        }
    }

    private func execute(_ command: String) -> ToolResult {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/sh")
        proc.arguments = ["-c", command]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe

        do {
            try proc.run()
        } catch {
            return ToolResult(
                content: "Error: failed to launch command: \(error.localizedDescription)",
                isError: true)
        }

        // Collect output incrementally off a background queue so we never have
        // to block on pipe EOF (a SIGTERM-ignoring child or an orphaned
        // grandchild can hold the write end open indefinitely).
        let handle = pipe.fileHandleForReading
        let box = OutputBox()
        let eof = DispatchSemaphore(value: 0)
        handle.readabilityHandler = { h in
            let chunk = h.availableData
            if chunk.isEmpty {
                h.readabilityHandler = nil
                eof.signal()
            } else {
                box.append(chunk)
            }
        }

        // Watchdog: SIGTERM at the timeout, then escalate to SIGKILL so even a
        // TERM-ignoring shell is reaped and `waitUntilExit` returns.
        let timeoutQueue = DispatchQueue(label: "krill.bash.timeout")
        let timedOut = TimeoutFlag()
        timeoutQueue.asyncAfter(deadline: .now() + timeout) { [weak proc] in
            guard let proc, proc.isRunning else { return }
            timedOut.set()
            let pid = proc.processIdentifier
            proc.terminate()  // SIGTERM
            timeoutQueue.asyncAfter(deadline: .now() + 1) { [weak proc] in
                if let proc, proc.isRunning { kill(pid, SIGKILL) }
            }
        }

        proc.waitUntilExit()
        // Prefer a clean EOF (full output), but cap the wait: if an orphaned
        // grandchild still holds the pipe after the shell is gone, stop reading
        // rather than hang the loop.
        if eof.wait(timeout: .now() + 2) == .timedOut {
            handle.readabilityHandler = nil
        }
        let outData = box.data()

        // Lossy decode so invalid UTF-8 (or a byte-boundary cut) never silently
        // drops the whole output - it becomes U+FFFD instead.
        var output = String(decoding: outData, as: UTF8.self)
        if output.utf8.count > maxOutputBytes {
            var tailBytes = Array(output.utf8.suffix(maxOutputBytes))
            // Advance past any leading UTF-8 continuation bytes so the kept tail
            // starts on a character boundary and no glyph is mangled.
            while let first = tailBytes.first, first & 0xC0 == 0x80 { tailBytes.removeFirst() }
            output = "[output truncated to last \(maxOutputBytes) bytes]\n"
                + String(decoding: tailBytes, as: UTF8.self)
        }
        if output.isEmpty { output = "(no output)" }

        if timedOut.isSet {
            return ToolResult(
                content: "Error: command timed out after \(Int(timeout))s and was terminated.\n"
                    + output,
                isError: true)
        }
        let code = proc.terminationStatus
        if code != 0 {
            return ToolResult(content: "Exit code \(code).\n\(output)", isError: true)
        }
        return ToolResult(content: output, isError: false)
    }
}

/// Tiny lock-guarded boolean shared between the timeout watchdog and the
/// command-running flow.
private final class TimeoutFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func set() { lock.lock(); value = true; lock.unlock() }
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
}

/// Lock-guarded output accumulator, appended from the pipe's readability queue
/// and read back once the command is done.
private final class OutputBox: @unchecked Sendable {
    private let lock = NSLock()
    private var buf = Data()
    func append(_ d: Data) { lock.lock(); buf.append(d); lock.unlock() }
    func data() -> Data { lock.lock(); defer { lock.unlock() }; return buf }
}
