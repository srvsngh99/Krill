import Foundation
import KrillTooling

/// Create or overwrite a file with the given contents (creating parent
/// directories as needed).
public struct WriteTool: Tool {
    public let name = "write_file"
    public let isFileEdit = true
    public let description =
        "Create a new file or overwrite an existing one with the given content. Creates parent directories."
    public let parametersJSON = """
    {"type":"object","properties":{\
    "path":{"type":"string","description":"File path to write (relative or absolute)."},\
    "content":{"type":"string","description":"The full file content."}},\
    "required":["path","content"]}
    """
    public init() {}

    public func run(argumentsJSON: String) async -> ToolResult {
        guard let obj = jsonObject(argumentsJSON), let path = obj["path"] as? String, !path.isEmpty else {
            return ToolResult(content: "Error: write_file requires a 'path'.", isError: true)
        }
        guard let content = obj["content"] as? String else {
            return ToolResult(content: "Error: write_file requires a 'content' string.", isError: true)
        }
        let url = FileToolSupport.resolve(path)
        var isDir: ObjCBool = false
        let existed = FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
        if existed && isDir.boolValue {
            return ToolResult(content: "Error: \(FileToolSupport.display(url)) is a directory.", isError: true)
        }
        // Read the prior content (if any) before overwriting, so the result can
        // report a diffstat against what was there.
        let oldContent = existed ? (try? String(contentsOf: url, encoding: .utf8)) ?? "" : ""
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(content.utf8).write(to: url)
        } catch {
            return ToolResult(content: "Error: could not write \(FileToolSupport.display(url)): \(error.localizedDescription)", isError: true)
        }
        let verb = existed ? "Overwrote" : "Created"
        let stat = FileToolSupport.diffstat(
            added: FileToolSupport.lineCount(content),
            removed: FileToolSupport.lineCount(oldContent))
        return ToolResult(
            content: "\(verb) \(FileToolSupport.display(url)) (\(stat), \(content.utf8.count) bytes).",
            isError: false)
    }
}
