import Foundation
import KrillTooling

/// Apply several exact-string edits to one file atomically: the edits run in
/// order against the in-memory text, and the file is written only if every edit
/// succeeds (all-or-nothing).
public struct MultiEditTool: Tool {
    public let name = "multi_edit"
    public let description =
        "Apply multiple exact-string edits to a single file in order, atomically (all succeed or none are written)."
    public let parametersJSON = """
    {"type":"object","properties":{\
    "path":{"type":"string","description":"File path to edit."},\
    "edits":{"type":"array","description":"Edits applied in order.","items":{"type":"object","properties":{\
    "old_string":{"type":"string"},"new_string":{"type":"string"},\
    "replace_all":{"type":"boolean"}},"required":["old_string","new_string"]}}},\
    "required":["path","edits"]}
    """
    public init() {}

    public func run(argumentsJSON: String) async -> ToolResult {
        guard let obj = jsonObject(argumentsJSON), let path = obj["path"] as? String, !path.isEmpty else {
            return ToolResult(content: "Error: multi_edit requires a 'path'.", isError: true)
        }
        guard let edits = obj["edits"] as? [[String: Any]], !edits.isEmpty else {
            return ToolResult(content: "Error: multi_edit requires a non-empty 'edits' array.", isError: true)
        }
        let url = FileToolSupport.resolve(path)
        guard let data = try? Data(contentsOf: url) else {
            return ToolResult(content: "Error: no such file: \(FileToolSupport.display(url))", isError: true)
        }

        var text = String(decoding: data, as: UTF8.self)
        var total = 0
        for (i, edit) in edits.enumerated() {
            guard let old = edit["old_string"] as? String, let new = edit["new_string"] as? String else {
                return ToolResult(content: "Error: edit \(i + 1) needs 'old_string' and 'new_string'.", isError: true)
            }
            let replaceAll = (edit["replace_all"] as? Bool) ?? false
            switch EditTool.apply(to: text, old: old, new: new, replaceAll: replaceAll) {
            case .error(let message):
                return ToolResult(content: "Error in edit \(i + 1) (no changes written): \(message)", isError: true)
            case .ok(let updated, let count):
                text = updated
                total += count
            }
        }
        do {
            try Data(text.utf8).write(to: url)
        } catch {
            return ToolResult(content: "Error: could not write \(FileToolSupport.display(url)): \(error.localizedDescription)", isError: true)
        }
        return ToolResult(
            content: "Applied \(edits.count) edit(s) to \(FileToolSupport.display(url)) (\(total) replacement(s)).",
            isError: false)
    }
}
