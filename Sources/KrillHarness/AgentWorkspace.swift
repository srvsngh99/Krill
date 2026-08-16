import Foundation

/// The working directory an agent run resolves paths against.
///
/// The CLI surfaces (`krill code`, the TUI) run one agent in the process that
/// launched them, so the process working directory IS the workspace and the
/// task-local stays nil. A hosting server runs many sessions in one process,
/// each rooted in its own repo - it binds `root` around the whole run
/// (`AgentWorkspace.$root.withValue(url) { await loop.run(...) }`) and every
/// tool under that task tree resolves against the session's workspace instead.
public enum AgentWorkspace {
    /// The current run's workspace root; nil means "use the process cwd".
    @TaskLocal public static var root: URL?

    /// The effective working directory path for the current task.
    public static var currentPath: String {
        root?.standardizedFileURL.path ?? FileManager.default.currentDirectoryPath
    }
}
