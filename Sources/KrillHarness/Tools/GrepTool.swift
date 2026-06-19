import Foundation
import KrillTooling

/// Search file contents for a regular expression, returning `path:line: text`
/// matches. Optionally restrict to files matching a glob.
public struct GrepTool: Tool {
    public let name = "grep"
    public let isReadOnly = true
    public let description =
        "Search file contents with a regular expression. Returns path:line:text matches. Optional glob filter."
    public let parametersJSON = """
    {"type":"object","properties":{\
    "pattern":{"type":"string","description":"Regular expression to search for."},\
    "path":{"type":"string","description":"Root directory to search (default: working directory)."},\
    "glob":{"type":"string","description":"Only search files matching this glob (e.g. '**/*.swift')."}},\
    "required":["pattern"]}
    """
    public let maxMatches: Int
    public init(maxMatches: Int = 200) { self.maxMatches = maxMatches }

    public func run(argumentsJSON: String) async -> ToolResult {
        guard let obj = jsonObject(argumentsJSON), let pattern = obj["pattern"] as? String, !pattern.isEmpty else {
            return ToolResult(content: "Error: grep requires a 'pattern'.", isError: true)
        }
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return ToolResult(content: "Error: invalid regular expression.", isError: true)
        }
        let rootPath = (obj["path"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "."
        let root = FileToolSupport.resolve(rootPath)

        var globRegex: NSRegularExpression?
        if let g = obj["glob"] as? String, !g.isEmpty {
            globRegex = try? NSRegularExpression(pattern: FileToolSupport.globToRegex(g))
        }

        var results: [String] = []
        var truncated = false
        outer: for file in FileToolSupport.walkFiles(root: root) {
            let rel = FileToolSupport.relativePath(file, to: root)
            if let gx = globRegex {
                let r = NSRange(rel.startIndex..<rel.endIndex, in: rel)
                if gx.firstMatch(in: rel, range: r) == nil { continue }
            }
            guard let data = try? Data(contentsOf: file), !isProbablyBinary(data) else { continue }
            let text = String(decoding: data, as: UTF8.self)
            var lineNo = 0
            for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
                lineNo += 1
                let s = String(line)
                let r = NSRange(s.startIndex..<s.endIndex, in: s)
                if regex.firstMatch(in: s, range: r) != nil {
                    let trimmed = s.count > 300 ? String(s.prefix(300)) + "..." : s
                    results.append("\(rel):\(lineNo): \(trimmed)")
                    if results.count >= maxMatches { truncated = true; break outer }
                }
            }
        }
        if results.isEmpty {
            return ToolResult(content: "No matches for /\(pattern)/.", isError: false)
        }
        var out = results.joined(separator: "\n")
        if truncated { out += "\n... (truncated at \(maxMatches) matches)" }
        return ToolResult(content: out, isError: false)
    }

    /// Heuristic: treat content with a NUL byte in the first chunk as binary.
    private func isProbablyBinary(_ data: Data) -> Bool {
        data.prefix(8000).contains(0)
    }
}
