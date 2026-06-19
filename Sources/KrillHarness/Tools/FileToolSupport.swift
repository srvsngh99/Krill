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
    /// relative paths are resolved against the process working directory.
    static func resolve(_ path: String) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") { return URL(fileURLWithPath: expanded).standardizedFileURL }
        let cwd = FileManager.default.currentDirectoryPath
        return URL(fileURLWithPath: cwd).appendingPathComponent(expanded).standardizedFileURL
    }

    /// Path shown back to the model: relative to cwd when inside it, else absolute.
    static func display(_ url: URL) -> String {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).standardizedFileURL.path
        let p = url.standardizedFileURL.path
        if p == cwd { return "." }
        let prefix = cwd.hasSuffix("/") ? cwd : cwd + "/"
        return p.hasPrefix(prefix) ? String(p.dropFirst(prefix.count)) : p
    }

    /// Compact before/after summary for an edit (the level of "diff rendering"
    /// a coding agent needs: what was replaced, not a full file patch).
    static func changeSummary(old: String, new: String, limit: Int = 240) -> String {
        func clip(_ s: String) -> String {
            let oneLine = s.replacingOccurrences(of: "\n", with: "\\n")
            return oneLine.count > limit ? String(oneLine.prefix(limit)) + "..." : oneLine
        }
        return "- \(clip(old))\n+ \(clip(new))"
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
                    out += ".*"
                    i += 2
                    if i < chars.count, chars[i] == "/" { i += 1 }  // `**/` also matches zero dirs
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
