import Foundation
import KrillTooling

/// List the entries of a directory (directories marked with a trailing slash).
public struct ListTool: Tool {
    public let name = "list_dir"
    public let description = "List the files and subdirectories of a directory."
    public let parametersJSON = """
    {"type":"object","properties":{\
    "path":{"type":"string","description":"Directory path (default: the working directory)."}},\
    "required":[]}
    """
    public init() {}

    public func run(argumentsJSON: String) async -> ToolResult {
        let obj = jsonObject(argumentsJSON) ?? [:]
        let path = (obj["path"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "."
        let url = FileToolSupport.resolve(path)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            return ToolResult(content: "Error: no such directory: \(FileToolSupport.display(url))", isError: true)
        }
        guard isDir.boolValue else {
            return ToolResult(content: "Error: \(FileToolSupport.display(url)) is a file (use read_file).", isError: true)
        }
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: url.path) else {
            return ToolResult(content: "Error: could not list \(FileToolSupport.display(url)).", isError: true)
        }
        if entries.isEmpty { return ToolResult(content: "(empty directory)", isError: false) }
        let rows = entries.sorted().map { name -> String in
            var sub: ObjCBool = false
            _ = FileManager.default.fileExists(atPath: url.appendingPathComponent(name).path, isDirectory: &sub)
            return sub.boolValue ? "\(name)/" : name
        }
        return ToolResult(content: rows.joined(separator: "\n"), isError: false)
    }
}
