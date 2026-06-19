import Foundation
import KrillTooling

/// Read a text file, returned with 1-based line numbers (so the model can cite
/// lines for edits). Optional `offset`/`limit` window large files.
public struct ReadTool: Tool {
    public let name = "read_file"
    public let isReadOnly = true
    public let description =
        "Read a text file and return its contents with line numbers. Use offset/limit for large files."
    public let parametersJSON = """
    {"type":"object","properties":{\
    "path":{"type":"string","description":"File path (relative to the working directory or absolute)."},\
    "offset":{"type":"integer","description":"1-based line to start at (optional)."},\
    "limit":{"type":"integer","description":"Maximum number of lines to return (optional)."}},\
    "required":["path"]}
    """

    /// Default max lines when no limit is given.
    public let defaultLimit: Int
    public init(defaultLimit: Int = 2000) { self.defaultLimit = defaultLimit }

    public func run(argumentsJSON: String) async -> ToolResult {
        guard let obj = jsonObject(argumentsJSON), let path = obj["path"] as? String, !path.isEmpty else {
            return ToolResult(content: "Error: read_file requires a 'path'.", isError: true)
        }
        let url = FileToolSupport.resolve(path)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            return ToolResult(content: "Error: no such file: \(FileToolSupport.display(url))", isError: true)
        }
        if isDir.boolValue {
            return ToolResult(content: "Error: \(FileToolSupport.display(url)) is a directory (use list_dir).", isError: true)
        }
        guard let data = try? Data(contentsOf: url) else {
            return ToolResult(content: "Error: could not read \(FileToolSupport.display(url)).", isError: true)
        }
        let text = String(decoding: data, as: UTF8.self)
        let allLines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let start = max(0, ((obj["offset"] as? Int) ?? 1) - 1)
        guard start < allLines.count else {
            return ToolResult(content: "(file has \(allLines.count) lines; offset \(start + 1) is past the end)", isError: false)
        }
        let limit = (obj["limit"] as? Int) ?? defaultLimit
        let end = min(allLines.count, start + max(1, limit))
        var rendered = ""
        for idx in start..<end {
            rendered += String(format: "%6d\t%@\n", idx + 1, allLines[idx])
        }
        if end < allLines.count {
            rendered += "... (\(allLines.count - end) more lines; use offset \(end + 1))\n"
        }
        return ToolResult(content: rendered.isEmpty ? "(empty file)" : rendered, isError: false)
    }
}
