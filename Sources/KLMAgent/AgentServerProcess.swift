import Foundation

/// Operator-agent surface for "start the daemon" / "stop the daemon".
///
/// Kept behind a protocol because the right behavior depends on the
/// install profile (Homebrew, source build, LaunchAgent in a future
/// rung) and on whether the user wants the agent to actually spawn
/// processes vs. just print the recommended command.
///
/// The default implementation supplied by sub-PR B
/// (`InstructionsOnlyServerProcess`) does NOT spawn `krillm serve`;
/// it returns a copy-paste command the user can run themselves. This
/// keeps the operator agent's blast radius narrow (no detached
/// child processes the user did not authorize). Sub-PR C's CLI can
/// register a process-spawning variant when `--yes` is passed.
public protocol AgentServerProcess: Sendable {
    /// Returns a string the model can show the user. Throws only on
    /// catastrophic transport failure (the instructions-only default
    /// never throws).
    func start(model: String?) async throws -> String

    /// Same shape as `start` but for "stop the running daemon".
    func stop() async throws -> String
}

/// Default `AgentServerProcess` for sub-PR B: never spawns; only
/// surfaces the canonical commands the user can paste.
public struct InstructionsOnlyServerProcess: AgentServerProcess {
    private let port: Int
    private let binaryName: String

    public init(port: Int = 11435, binaryName: String = "krillm") {
        self.port = port
        self.binaryName = binaryName
    }

    public func start(model: String?) async throws -> String {
        let modelFlag = model.map { " --model \($0)" } ?? ""
        return "To start the daemon, run: " +
            "\(binaryName) serve --port \(port)\(modelFlag)"
    }

    public func stop() async throws -> String {
        return "To stop the daemon, run: \(binaryName) stop"
    }
}
