import Foundation

/// The ambient facts every agent turn should know without spending a tool
/// call: date/time, working directory, platform, and the serving model. One
/// compact line, prepended to the agent's system prompt by each surface —
/// local models otherwise burn a whole thinking budget guessing the date.
public enum AgentEnvironment {
    /// Deliberately differs from `ToolCalling.agenticToolDirective`: surfaces
    /// that synthesize their own system turn suppress the tooling fallback, so
    /// this version also carries Krill's training-cutoff/web-search guidance.
    public static let toolDirective =
        "Use tools only when needed. Once you have the tool results, "
        + "reply with the final answer and do not call any more tools - except that an ask_user "
        + "answer, or a granted request_execute, means you keep working instead of stopping. "
        + "Your training data has a cutoff: for facts that change over time "
        + "(office-holders, prices, versions, news), verify with web_search "
        + "instead of answering from memory."

    /// Always-on guidance: clarification is available in every permission
    /// posture and an answer continues the current run rather than ending it.
    public static let askUserDirective =
        "Use ask_user when a missing user choice materially affects the result. Ask one focused "
        + "question with concise options when useful. After the answer, continue the task with it; "
        + "a declined question means choose the safest assumption, state it, and do not ask again."

    /// Shared system-level planning steer for CLI and remote builders.
    public static let planSystemSteer =
        "You are in PLAN MODE (read-only). You may read and search files with the read-only tools, "
        + "but you must NOT write files, edit files, or run shell commands - those are denied. "
        + "Investigate as needed and use ask_user when a user choice affects the plan. When the plan "
        + "is ready, call request_execute with a concise summary; do not merely stop at a plan."

    /// Shared per-turn planning prefix for interactive/background TUI builders.
    public static let planTurnPrefix =
        "(Plan mode: read-only. Investigate with the read-only tools and propose a clear, step-by-step "
        + "plan. You may use ask_user for clarification. Call request_execute when ready to implement. "
        + "Do not edit files or run commands.)"

    public static let adaptivePlanTail =
        "This is ADAPTIVE mode: begin by planning read-only, then call request_execute when you decide "
        + "the plan is sufficient. It will enter guarded execution without asking the user."

    /// Pure prompt fragments for a posture. Builders append these to their own
    /// environment/project/user-system parts; `askUserDirective` is unconditional.
    public static func permissionDirectives(for mode: PermissionMode) -> [String] {
        var directives = [askUserDirective]
        if mode.initialEffective == .plan {
            directives.append(planSystemSteer)
            if mode == .adaptive { directives.append(adaptivePlanTail) }
        }
        return directives
    }

    /// The project brief: `Krill.md` at the working-directory root (the file
    /// `/init` generates — Krill's CLAUDE.md), loaded into every agent session
    /// so its build commands, architecture notes, and conventions are simply
    /// known. Capped so a runaway file cannot eat the context window; nil when
    /// absent or unreadable.
    public static func projectBrief(maxChars: Int = 12_000) -> String? {
        let url = URL(fileURLWithPath: AgentWorkspace.currentPath)
            .appendingPathComponent("Krill.md")
        guard let raw = try? String(contentsOf: url, encoding: .utf8),
              !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        let clipped = raw.count > maxChars
            ? String(raw.prefix(maxChars)) + "\n… (Krill.md truncated)"
            : raw
        return "Project brief (Krill.md):\n\(clipped)"
    }

    public static func contextLine(modelName: String? = nil) -> String {
        let now = Date()
        // Month spelled out: numeric MM/DD reads as DD/MM to many models
        // (gemma answered "April 8" for 08/04/2026).
        let date = now.formatted(.dateTime.weekday(.wide).month(.wide).day().year())
        let time = now.formatted(.dateTime.hour().minute())
        let zone = TimeZone.current.abbreviation() ?? TimeZone.current.identifier
        let os = ProcessInfo.processInfo.operatingSystemVersionString
        #if arch(arm64)
        let arch = "arm64"
        #else
        let arch = "x86_64"
        #endif
        var parts = [
            "Environment: \(date), \(time) \(zone)",
            "cwd \(AgentWorkspace.currentPath)",
            "macOS \(os) \(arch)",
        ]
        if let modelName, !modelName.isEmpty { parts.append("model \(modelName)") }
        return parts.joined(separator: " · ")
    }
}
