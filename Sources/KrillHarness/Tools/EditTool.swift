import Foundation
import KrillTooling

/// Exact-string replacement in a file. By default `old_string` must occur
/// exactly once (so an edit is unambiguous); set `replace_all` to replace every
/// occurrence.
public struct EditTool: Tool {
    public let name = "edit_file"
    public let isFileEdit = true
    public let description =
        "Replace an exact string in a file. old_string must match exactly (incl. whitespace) and be unique unless replace_all is true."
    public let parametersJSON = """
    {"type":"object","properties":{\
    "path":{"type":"string","description":"File path to edit."},\
    "old_string":{"type":"string","description":"Exact text to replace."},\
    "new_string":{"type":"string","description":"Replacement text."},\
    "replace_all":{"type":"boolean","description":"Replace all occurrences (default false)."}},\
    "required":["path","old_string","new_string"]}
    """
    public init() {}

    public func run(argumentsJSON: String) async -> ToolResult {
        guard let obj = jsonObject(argumentsJSON), let path = obj["path"] as? String, !path.isEmpty else {
            return ToolResult(content: "Error: edit_file requires a 'path'.", isError: true)
        }
        guard let oldString = obj["old_string"] as? String, let newString = obj["new_string"] as? String else {
            return ToolResult(content: "Error: edit_file requires 'old_string' and 'new_string'.", isError: true)
        }
        let replaceAll = (obj["replace_all"] as? Bool) ?? false
        let url = FileToolSupport.resolve(path)
        guard let data = try? Data(contentsOf: url) else {
            return ToolResult(content: "Error: no such file: \(FileToolSupport.display(url))", isError: true)
        }
        let text = String(decoding: data, as: UTF8.self)
        switch EditTool.apply(to: text, old: oldString, new: newString, replaceAll: replaceAll) {
        case .error(let message):
            return ToolResult(content: message, isError: true)
        case .ok(let updated, let count):
            do {
                try Data(updated.utf8).write(to: url)
            } catch {
                return ToolResult(content: "Error: could not write \(FileToolSupport.display(url)): \(error.localizedDescription)", isError: true)
            }
            return ToolResult(
                content: "Edited \(FileToolSupport.display(url)): \(count) replacement(s).\n"
                    + FileToolSupport.changeSummary(old: oldString, new: newString),
                isError: false)
        }
    }

    /// Outcome of a pure edit: either the updated text + replacement count, or
    /// an error message to surface to the model.
    enum Outcome: Equatable {
        case ok(text: String, count: Int)
        case error(String)
    }

    /// Pure edit logic (testable without the filesystem). `old`/`new` must
    /// differ; `old` must occur exactly once unless `replaceAll`.
    static func apply(to text: String, old: String, new: String, replaceAll: Bool) -> Outcome {
        if old == new {
            return .error("Error: old_string and new_string are identical.")
        }
        if old.isEmpty {
            return .error("Error: old_string must not be empty.")
        }
        let occurrences = text.components(separatedBy: old).count - 1
        if occurrences == 0 {
            return .error("Error: old_string not found in the file.")
        }
        if occurrences > 1 && !replaceAll {
            return .error("Error: old_string occurs \(occurrences) times; pass replace_all or add more context to make it unique.")
        }
        if replaceAll {
            return .ok(text: text.replacingOccurrences(of: old, with: new), count: occurrences)
        }
        if let range = text.range(of: old) {
            return .ok(text: text.replacingCharacters(in: range, with: new), count: 1)
        }
        return .error("Error: old_string not found in the file.")
    }
}
