import Foundation

/// Parse a tool's `argumentsJSON` into a dictionary, or nil if malformed.
func jsonObject(_ s: String) -> [String: Any]? {
    guard let data = s.data(using: .utf8),
          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    return obj
}

/// Shared helpers for the filesystem tools: path resolution relative to the
/// agent's working directory, a compact change summary for mutating tools, and
/// glob-to-regex translation.
enum FileToolSupport {
    /// Resolve a tool-supplied path. Absolute and `~` paths are honoured;
    /// relative paths are resolved against the agent's working directory
    /// (`AgentWorkspace` - the process cwd unless a hosting server bound a
    /// per-session workspace around the run).
    static func resolve(_ path: String) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") { return URL(fileURLWithPath: expanded).standardizedFileURL }
        let cwd = AgentWorkspace.currentPath
        return URL(fileURLWithPath: cwd).appendingPathComponent(expanded).standardizedFileURL
    }

    /// Path shown back to the model: relative to cwd when inside it, else absolute.
    static func display(_ url: URL) -> String {
        let cwd = URL(fileURLWithPath: AgentWorkspace.currentPath).standardizedFileURL.path
        let p = url.standardizedFileURL.path
        if p == cwd { return "." }
        let prefix = cwd.hasSuffix("/") ? cwd : cwd + "/"
        return p.hasPrefix(prefix) ? String(p.dropFirst(prefix.count)) : p
    }

    /// Number of lines in a string (an empty string is 0 lines).
    static func lineCount(_ s: String) -> Int {
        lines(in: s).count
    }

    /// Build line-based unified-diff hunks with true old/new line numbers.
    /// The complete result is intended for `ToolResult.display`, not the model
    /// transcript. `CollectionDifference` supplies a stable shortest edit script;
    /// this pass adds context and merges overlapping hunk ranges.
    static func unifiedDiff(old: String, new: String, context: Int = 3) -> [ToolDiffHunk] {
        let oldLines = lines(in: old)
        let newLines = lines(in: new)
        let difference = newLines.difference(from: oldLines)
        let removals = Set(difference.compactMap { change -> Int? in
            guard case .remove(let offset, _, _) = change else { return nil }
            return offset
        })
        let insertions = Set(difference.compactMap { change -> Int? in
            guard case .insert(let offset, _, _) = change else { return nil }
            return offset
        })

        var allLines: [ToolDiffLine] = []
        var oldBefore: [Int] = []
        var newBefore: [Int] = []
        var oldIndex = 0
        var newIndex = 0
        while oldIndex < oldLines.count || newIndex < newLines.count {
            oldBefore.append(oldIndex)
            newBefore.append(newIndex)
            if oldIndex < oldLines.count, removals.contains(oldIndex) {
                allLines.append(ToolDiffLine(
                    oldLine: oldIndex + 1, newLine: nil, kind: .deletion,
                    text: oldLines[oldIndex]))
                oldIndex += 1
            } else if newIndex < newLines.count, insertions.contains(newIndex) {
                allLines.append(ToolDiffLine(
                    oldLine: nil, newLine: newIndex + 1, kind: .addition,
                    text: newLines[newIndex]))
                newIndex += 1
            } else if oldIndex < oldLines.count, newIndex < newLines.count {
                allLines.append(ToolDiffLine(
                    oldLine: oldIndex + 1, newLine: newIndex + 1, kind: .context,
                    text: oldLines[oldIndex]))
                oldIndex += 1
                newIndex += 1
            } else if oldIndex < oldLines.count {
                // Defensive fallback for a malformed/incomplete edit script.
                allLines.append(ToolDiffLine(
                    oldLine: oldIndex + 1, newLine: nil, kind: .deletion,
                    text: oldLines[oldIndex]))
                oldIndex += 1
            } else {
                allLines.append(ToolDiffLine(
                    oldLine: nil, newLine: newIndex + 1, kind: .addition,
                    text: newLines[newIndex]))
                newIndex += 1
            }
        }

        let changed = allLines.indices.filter { allLines[$0].kind != .context }
        guard !changed.isEmpty else { return [] }
        let context = max(0, context)
        var ranges: [Range<Int>] = []
        for index in changed {
            let candidate = max(0, index - context) ..< min(allLines.count, index + context + 1)
            if let last = ranges.last, candidate.lowerBound <= last.upperBound {
                ranges[ranges.count - 1] = last.lowerBound ..< max(last.upperBound, candidate.upperBound)
            } else {
                ranges.append(candidate)
            }
        }

        return ranges.map { range in
            let hunkLines = Array(allLines[range])
            let oldCount = hunkLines.reduce(0) { $0 + ($1.oldLine == nil ? 0 : 1) }
            let newCount = hunkLines.reduce(0) { $0 + ($1.newLine == nil ? 0 : 1) }
            let oldCursor = oldBefore[range.lowerBound]
            let newCursor = newBefore[range.lowerBound]
            return ToolDiffHunk(
                oldStart: oldCount == 0 ? oldCursor : oldCursor + 1,
                oldCount: oldCount,
                newStart: newCount == 0 ? newCursor : newCursor + 1,
                newCount: newCount,
                lines: hunkLines)
        }
    }

    /// Compact diffstat for a mutating tool result, e.g. "+5 -2".
    static func diffstat(added: Int, removed: Int) -> String { "+\(added) -\(removed)" }

    static func diffstat(hunks: [ToolDiffHunk]) -> String {
        var added = 0
        var removed = 0
        for line in hunks.flatMap(\.lines) {
            if line.kind == .addition { added += 1 }
            if line.kind == .deletion { removed += 1 }
        }
        return diffstat(added: added, removed: removed)
    }

    /// A bounded preview of the first hunk for model-facing tool content. The
    /// complete hunks remain available through `ToolResult.display` for the UI.
    static func compactPreview(
        hunks: [ToolDiffHunk], maxLines: Int = 8, maxCharacters: Int = 600
    ) -> String {
        guard let hunk = hunks.first, maxLines > 0, maxCharacters > 0 else { return "" }
        var rows = ["@@ -\(hunk.oldStart),\(hunk.oldCount) +\(hunk.newStart),\(hunk.newCount) @@"]
        for line in hunk.lines.prefix(maxLines) {
            let marker: String
            switch line.kind {
            case .context: marker = " "
            case .addition: marker = "+"
            case .deletion: marker = "-"
            }
            rows.append(marker + line.text)
        }
        if hunk.lines.count > maxLines || hunks.count > 1 { rows.append("...") }
        let preview = rows.joined(separator: "\n")
        if preview.count <= maxCharacters { return preview }
        return String(preview.prefix(maxCharacters)) + "..."
    }

    private static func lines(in text: String) -> [String] {
        text.isEmpty ? [] : text.components(separatedBy: "\n")
    }

    /// Translate a glob (`*`, `?`, `**`, character classes) into an anchored
    /// regex matched against a path relative to the search root. `**` crosses
    /// directory separators; `*` does not.
    static func globToRegex(_ glob: String) -> String {
        var out = "^"
        let chars = Array(glob)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            switch c {
            case "*":
                if i + 1 < chars.count, chars[i + 1] == "*" {
                    if i + 2 < chars.count, chars[i + 2] == "/" {
                        // `**/` matches zero or more WHOLE path components, so it
                        // keeps a `/` boundary (`**/foo` must not match `barfoo`).
                        out += "(?:.*/)?"
                        i += 3
                    } else {
                        out += ".*"  // bare `**`
                        i += 2
                    }
                    continue
                }
                out += "[^/]*"
            case "?":
                out += "[^/]"
            case ".", "(", ")", "+", "|", "^", "$", "\\", "{", "}":
                out += "\\\(c)"
            case "[":
                out += "["  // pass character classes through
            case "]":
                out += "]"
            default:
                out.append(c)
            }
            i += 1
        }
        return out + "$"
    }

    /// Directory names skipped by recursive walks (glob/grep).
    static let ignoredDirs: Set<String> = [".git", ".build", "node_modules", ".swiftpm", "DerivedData"]

    /// Recursively collect regular files under `root`, skipping hidden files and
    /// `ignoredDirs`. Capped so a huge tree cannot exhaust memory.
    static func walkFiles(root: URL, maxFiles: Int = 20_000) -> [URL] {
        let fm = FileManager.default
        guard let en = fm.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
            options: [.skipsHiddenFiles],
            errorHandler: { _, _ in true }
        ) else { return [] }

        var files: [URL] = []
        for case let url as URL in en {
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey])
            if values?.isDirectory == true {
                if ignoredDirs.contains(url.lastPathComponent) { en.skipDescendants() }
                continue
            }
            if values?.isRegularFile == true {
                files.append(url)
                if files.count >= maxFiles { break }
            }
        }
        return files
    }

    /// Path of `url` relative to `root` (for glob matching and grep output).
    static func relativePath(_ url: URL, to root: URL) -> String {
        let base = root.standardizedFileURL.path
        let p = url.standardizedFileURL.path
        let prefix = base.hasSuffix("/") ? base : base + "/"
        return p.hasPrefix(prefix) ? String(p.dropFirst(prefix.count)) : p
    }
}
