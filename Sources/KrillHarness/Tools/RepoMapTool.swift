import Foundation
import KrillTooling

/// Native repo mapper: a compact, ranked tree of a project's source files with
/// their top-level symbols, built with nothing but FileManager and per-language
/// line regexes — no tree-sitter, no external binaries, in keeping with the
/// one-binary identity. `/init` uses it to write Krill.md's Repo map section,
/// and the agent can call it directly to orient in an unfamiliar tree.
public enum RepoMap {

    /// Directories that are dependency/build/VCS output, never source.
    static let skippedDirs: Set<String> = [
        ".git", ".build", ".swiftpm", "build", "dist", "out", "DerivedData",
        "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
        "Pods", "Carthage", "target", ".next", ".cache", "vendor", "coverage",
        ".idea", ".vscode", ".tox", ".eggs",
    ]

    /// One symbol-extraction rule: file extensions + top-level declaration
    /// patterns (anchored to column 0 — nested declarations are noise here).
    fileprivate struct Language {
        let extensions: Set<String>
        let patterns: [NSRegularExpression]
    }

    private static func regex(_ pattern: String) -> NSRegularExpression {
        // Patterns are compile-time constants; a failure is a programmer error.
        try! NSRegularExpression(pattern: pattern, options: [.anchorsMatchLines])
    }

    private static let languages: [Language] = [
        Language(
            extensions: ["swift"],
            patterns: [regex(#"^(?:@\w+ )*(?:public |open |package |internal |final |indirect )*(?:class|struct|enum|protocol|actor|extension|func|typealias|let|var) ([A-Za-z_][A-Za-z0-9_]*)"#)]),
        Language(
            extensions: ["py"],
            patterns: [regex(#"^(?:async )?(?:def|class) ([A-Za-z_][A-Za-z0-9_]*)"#)]),
        Language(
            extensions: ["js", "jsx", "mjs", "ts", "tsx"],
            patterns: [
                regex(#"^export (?:default )?(?:async )?(?:function|class) ([A-Za-z_$][A-Za-z0-9_$]*)"#),
                regex(#"^(?:async )?(?:function|class) ([A-Za-z_$][A-Za-z0-9_$]*)"#),
                regex(#"^export (?:const|let|var|type|interface|enum) ([A-Za-z_$][A-Za-z0-9_$]*)"#),
            ]),
        Language(
            extensions: ["go"],
            patterns: [regex(#"^(?:func|type) (?:\([^)]*\) )?([A-Za-z_][A-Za-z0-9_]*)"#)]),
        Language(
            extensions: ["rs"],
            patterns: [regex(#"^(?:pub(?:\([^)]*\))? )?(?:fn|struct|enum|trait|mod) ([A-Za-z_][A-Za-z0-9_]*)"#)]),
        Language(
            extensions: ["rb"],
            patterns: [regex(#"^(?:class|module|def) ([A-Za-z_][A-Za-z0-9_.]*)"#)]),
        Language(
            extensions: ["java", "kt", "kts"],
            patterns: [regex(#"^(?:@\w+ )*(?:public |private |internal |final |abstract |open |data |sealed )*(?:class|interface|enum|object|fun) ([A-Za-z_][A-Za-z0-9_]*)"#)]),
        Language(
            extensions: ["c", "h", "cpp", "hpp", "cc", "m", "mm", "metal"],
            patterns: [regex(#"^(?:typedef struct|struct|@interface|@implementation|void|int|float|double|kernel) ([A-Za-z_][A-Za-z0-9_]*)"#)]),
    ]

    /// Non-source files still worth a line in the map (build entry points).
    private static let notableNames: Set<String> = [
        "Makefile", "Dockerfile", "Package.swift", "CMakeLists.txt",
    ]

    private struct FileEntry {
        let relPath: String
        let depth: Int
        let symbols: [String]
        let modified: Date
    }

    /// Walk `root` and produce the map, at most ~`maxChars`. Deterministic:
    /// directories and files in sorted order; when over budget, files drop
    /// oldest-modified-first so what survives is what is being worked on.
    public static func generate(root: URL, maxChars: Int = 9_000) -> String {
        let fm = FileManager.default
        var entries: [FileEntry] = []
        var skippedDirCount = 0
        var otherFileCount = 0

        func walk(_ dir: URL, rel: String, depth: Int) {
            guard depth <= 8, entries.count < 600 else { return }
            let children = (try? fm.contentsOfDirectory(
                at: dir, includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey],
                options: [.skipsHiddenFiles])) ?? []
            for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                let name = child.lastPathComponent
                let childRel = rel.isEmpty ? name : rel + "/" + name
                var isDir: ObjCBool = false
                fm.fileExists(atPath: child.path, isDirectory: &isDir)
                if isDir.boolValue {
                    if skippedDirs.contains(name) { skippedDirCount += 1; continue }
                    walk(child, rel: childRel, depth: depth + 1)
                    continue
                }
                let ext = child.pathExtension.lowercased()
                let language = languages.first { $0.extensions.contains(ext) }
                let isNotable = notableNames.contains(name) || ext == "md"
                guard language != nil || isNotable else { otherFileCount += 1; continue }
                let modified = (try? child.resourceValues(forKeys: [.contentModificationDateKey])
                    .contentModificationDate) ?? Date.distantPast
                entries.append(FileEntry(
                    relPath: childRel, depth: depth,
                    symbols: language.map { symbols(of: child, language: $0) } ?? [],
                    modified: modified ?? Date.distantPast))
            }
        }
        walk(root, rel: "", depth: 0)

        // Over budget: drop oldest files until the render fits (structure-first
        // degradation — recent work survives, dormant corners fall off).
        var kept = entries
        var body = render(kept)
        if body.count > maxChars {
            let byAge = kept.sorted { $0.modified < $1.modified }
            var dropSet: Set<String> = []
            for candidate in byAge {
                guard body.count > maxChars else { break }
                dropSet.insert(candidate.relPath)
                kept = entries.filter { !dropSet.contains($0.relPath) }
                body = render(kept)
            }
        }

        var header = "Repo map of \(root.path) — \(kept.count) source files"
        if kept.count < entries.count { header += " (\(entries.count - kept.count) older files elided)" }
        if skippedDirCount > 0 { header += ", \(skippedDirCount) build/dependency dirs skipped" }
        if otherFileCount > 0 { header += ", \(otherFileCount) other files" }
        return header + ".\n" + body
    }

    /// Top-level symbols of one file, in declaration order, capped at 6.
    fileprivate static func symbols(of file: URL, language: Language) -> [String] {
        symbols(source: (try? String(contentsOf: file, encoding: .utf8)) ?? "", language: language)
    }

    private static func symbols(source: String, language: Language) -> [String] {
        // Cap the scanned prefix so one giant generated file cannot stall the walk.
        let text = String(source.prefix(200_000))
        let range = NSRange(text.startIndex..., in: text)
        var found: [(Int, String)] = []
        for pattern in language.patterns {
            pattern.enumerateMatches(in: text, range: range) { match, _, _ in
                guard let match, match.numberOfRanges > 1,
                      let nameRange = Range(match.range(at: 1), in: text) else { return }
                found.append((match.range.location, String(text[nameRange])))
            }
        }
        var seen: Set<String> = []
        return found.sorted { $0.0 < $1.0 }
            .map(\.1)
            .filter { seen.insert($0).inserted }
            .prefix(6)
            .map { $0 }
    }

    /// Render entries as an indented tree, emitting each directory line once.
    private static func render(_ entries: [FileEntry]) -> String {
        var lines: [String] = []
        var emittedDirs: Set<String> = []
        for entry in entries.sorted(by: { $0.relPath < $1.relPath }) {
            let components = entry.relPath.split(separator: "/").map(String.init)
            // Emit any not-yet-seen ancestor directories.
            for depth in 0 ..< max(0, components.count - 1) {
                let dirPath = components[0 ... depth].joined(separator: "/")
                if emittedDirs.insert(dirPath).inserted {
                    lines.append(String(repeating: "  ", count: depth) + components[depth] + "/")
                }
            }
            let indent = String(repeating: "  ", count: max(0, components.count - 1))
            let name = components.last ?? entry.relPath
            let symbolNote = entry.symbols.isEmpty ? "" : " — " + entry.symbols.joined(separator: ", ")
            lines.append(indent + name + symbolNote)
        }
        return lines.joined(separator: "\n")
    }

    // Test seam: expose symbol extraction for one language by extension.
    static func extractSymbols(source: String, fileExtension: String) -> [String] {
        guard let language = languages.first(where: { $0.extensions.contains(fileExtension) })
        else { return [] }
        return symbols(source: source, language: language)
    }
}

/// The `repo_map` tool: the mapper behind a read-only tool so the agent can
/// orient in a tree it has not seen (or refresh its picture after big changes).
public struct RepoMapTool: Tool {
    public let name = "repo_map"
    public let isReadOnly = true
    public let description =
        "Map the repository: a compact tree of source files with their top-level symbols "
        + "(classes, functions, types), skipping build output and dependencies. Use it to "
        + "orient in an unfamiliar codebase before reading specific files."
    public let parametersJSON = """
    {"type":"object","properties":{\
    "path":{"type":"string","description":"Directory to map (default: the working directory)."}},\
    "required":[]}
    """
    public init() {}

    public func run(argumentsJSON: String) async -> ToolResult {
        let obj = jsonObject(argumentsJSON) ?? [:]
        let path = (obj["path"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "."
        let url = FileToolSupport.resolve(path)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return ToolResult(content: "Error: no such directory: \(FileToolSupport.display(url))", isError: true)
        }
        return ToolResult(content: RepoMap.generate(root: url), isError: false)
    }
}
