import Foundation
import KrillTooling

/// Find files whose path (relative to the search root) matches a glob pattern.
/// Supports `*`, `?`, character classes, and `**` (crosses directories).
public struct GlobTool: Tool {
    public let name = "glob"
    public let description =
        "Find files matching a glob pattern (e.g. '**/*.swift', 'Sources/*.json'). Returns matching paths."
    public let parametersJSON = """
    {"type":"object","properties":{\
    "pattern":{"type":"string","description":"Glob pattern, e.g. '**/*.swift'."},\
    "path":{"type":"string","description":"Root directory to search (default: working directory)."}},\
    "required":["pattern"]}
    """
    public let maxResults: Int
    public init(maxResults: Int = 500) { self.maxResults = maxResults }

    public func run(argumentsJSON: String) async -> ToolResult {
        guard let obj = jsonObject(argumentsJSON), let pattern = obj["pattern"] as? String, !pattern.isEmpty else {
            return ToolResult(content: "Error: glob requires a 'pattern'.", isError: true)
        }
        let rootPath = (obj["path"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "."
        let root = FileToolSupport.resolve(rootPath)
        guard let regex = try? NSRegularExpression(pattern: FileToolSupport.globToRegex(pattern)) else {
            return ToolResult(content: "Error: invalid glob pattern.", isError: true)
        }
        var matches: [String] = []
        for file in FileToolSupport.walkFiles(root: root) {
            let rel = FileToolSupport.relativePath(file, to: root)
            let range = NSRange(rel.startIndex..<rel.endIndex, in: rel)
            if regex.firstMatch(in: rel, range: range) != nil {
                matches.append(rel)
                if matches.count >= maxResults { break }
            }
        }
        if matches.isEmpty {
            return ToolResult(content: "No files match \(pattern).", isError: false)
        }
        var out = matches.sorted().joined(separator: "\n")
        if matches.count >= maxResults { out += "\n... (truncated at \(maxResults) matches)" }
        return ToolResult(content: out, isError: false)
    }
}
