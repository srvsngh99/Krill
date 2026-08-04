import Foundation

/// The ambient facts every agent turn should know without spending a tool
/// call: date/time, working directory, platform, and the serving model. One
/// compact line, prepended to the agent's system prompt by each surface —
/// local models otherwise burn a whole thinking budget guessing the date.
public enum AgentEnvironment {
    /// Mirror of `ToolCalling.agenticToolDirective`: surfaces that synthesize a
    /// system turn (which suppresses the tooling layer's own fallback
    /// directive) append this so over-calling families still get the nudge.
    public static let toolDirective =
        "Use tools only when needed. Once you have the tool results, "
        + "reply with the final answer and do not call any more tools. "
        + "Your training data has a cutoff: for facts that change over time "
        + "(office-holders, prices, versions, news), verify with web_search "
        + "instead of answering from memory."

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
            "cwd \(FileManager.default.currentDirectoryPath)",
            "macOS \(os) \(arch)",
        ]
        if let modelName, !modelName.isEmpty { parts.append("model \(modelName)") }
        return parts.joined(separator: " · ")
    }
}
