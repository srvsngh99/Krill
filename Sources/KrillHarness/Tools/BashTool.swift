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

        // Watchdog: terminate if it runs past the timeout. The flag is shared
        // between the timeout queue and this function, so guard it with a lock.
        let timeoutQueue = DispatchQueue(label: "krill.bash.timeout")
        let timedOut = TimeoutFlag()
        let deadline = DispatchTime.now() + timeout
        timeoutQueue.asyncAfter(deadline: deadline) { [weak proc] in
            if let proc, proc.isRunning {
                timedOut.set()
                proc.terminate()
            }
        }

        // Read to EOF BEFORE waiting, so a large output can't deadlock the pipe.
        let outData = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()

        var output = String(data: outData, encoding: .utf8) ?? ""
        if output.utf8.count > maxOutputBytes {
            let tail = Data(output.utf8.suffix(maxOutputBytes))
            output = "[output truncated to last \(maxOutputBytes) bytes]\n"
                + (String(data: tail, encoding: .utf8) ?? "")
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
