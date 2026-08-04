import Foundation
import KrillTooling

/// Current date and time. The system prompt already carries a timestamp from
/// launch; this tool is for long sessions where "now" has drifted, or when the
/// model needs an exact machine-readable instant.
public struct NowTool: Tool {
    public let name = "now"
    public let isReadOnly = true
    public let description =
        "Get the current date and time (local timezone, ISO 8601, and Unix epoch)."
    public let parametersJSON = """
    {"type":"object","properties":{},"required":[]}
    """
    public init() {}

    public func run(argumentsJSON: String) async -> ToolResult {
        let now = Date()
        let iso = ISO8601DateFormatter()
        iso.timeZone = .current
        let zone = TimeZone.current.abbreviation() ?? TimeZone.current.identifier
        let human = now.formatted(
            .dateTime.weekday(.wide).year().month(.wide).day().hour().minute().second())
        return ToolResult(content: """
            \(human) \(zone)
            iso8601: \(iso.string(from: now))
            unix: \(Int(now.timeIntervalSince1970))
            """, isError: false)
    }
}
