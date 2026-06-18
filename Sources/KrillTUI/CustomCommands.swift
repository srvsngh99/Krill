import Foundation

/// A user-authored slash command, the Krill analogue of Claude Code's
/// `.claude/commands/*.md`. Each file under `~/.krill/commands/<name>.md`
/// becomes a `/<name>` command whose body is a prompt template expanded with
/// the text the user typed after the command.
///
/// Placeholders inside the template:
///   `$ARGUMENTS` / `$ARGS` / `$INPUT` -> the whole argument string
///   `$1` .. `$9`                      -> whitespace-split positional words
/// If a template references none of these and the user supplied arguments, the
/// arguments are appended on a new line (so a no-placeholder template still
/// receives the input).
public struct CustomCommand: Equatable, Sendable {
    /// Command word without the leading slash, lowercased (e.g. `review`).
    public let name: String
    /// One-line summary shown in the autosuggest popup and `/help`.
    public let description: String
    /// The raw prompt template (frontmatter already stripped).
    public let template: String

    public init(name: String, description: String, template: String) {
        self.name = name
        self.description = description
        self.template = template
    }

    /// Whole-argument tokens, longest first so `$ARGUMENTS` is matched before
    /// its `$ARGS` prefix.
    private static let wholeTokens = ["$ARGUMENTS", "$ARGS", "$INPUT"]

    /// Expand the template against the user-supplied argument string.
    ///
    /// A single forward pass over the template: at each `$` the longest known
    /// token wins (`$ARGUMENTS` before `$ARGS`; a lone `$1`..`$9` but never a
    /// multi-digit `$10`, which stays literal). Substituted *values* are never
    /// re-scanned, so a user argument that itself contains `$INPUT` is taken
    /// literally rather than re-expanded.
    public func expand(arguments: String) -> String {
        let args = arguments.trimmingCharacters(in: .whitespacesAndNewlines)
        let words = args.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
        let chars = Array(template)
        var out = ""
        out.reserveCapacity(template.count)
        var substituted = false
        var i = 0

        while i < chars.count {
            guard chars[i] == "$" else { out.append(chars[i]); i += 1; continue }

            // Whole-argument token (longest first).
            if let token = Self.wholeTokens.first(where: { Self.matches($0, chars, at: i) }) {
                out += args; i += token.count; substituted = true; continue
            }

            // Positional $1..$9, but only a lone single digit (a multi-digit run
            // like $10 is left verbatim).
            if i + 1 < chars.count, let d = chars[i + 1].wholeNumberValue, (1...9).contains(d),
               !(i + 2 < chars.count && chars[i + 2].isNumber) {
                out += d <= words.count ? words[d - 1] : ""
                i += 2; substituted = true; continue
            }

            out.append(chars[i]); i += 1   // literal `$`
        }

        // A template with no placeholder still receives the args (appended).
        if !substituted, !args.isEmpty { out += "\n\n" + args }
        return out
    }

    /// True if `token` (as characters) appears in `chars` starting at `i`.
    private static func matches(_ token: String, _ chars: [Character], at i: Int) -> Bool {
        let t = Array(token)
        guard i + t.count <= chars.count else { return false }
        for k in 0..<t.count where chars[i + k] != t[k] { return false }
        return true
    }
}

/// A loaded set of custom commands. `parse` (a single file's contents) and the
/// lookup are pure and unit-tested; `load(from:)` is the thin filesystem entry.
public struct CustomCommandStore: Sendable {
    public let commands: [CustomCommand]

    public init(commands: [CustomCommand]) {
        // Dedup by (lowercased) name so two files differing only in case do not
        // produce duplicate entries; the first one encountered wins.
        var seen = Set<String>()
        let unique = commands.filter { seen.insert($0.name).inserted }
        self.commands = unique.sorted { $0.name < $1.name }
    }

    public var isEmpty: Bool { commands.isEmpty }

    /// Look up a command by name, with or without a leading slash.
    public func command(named name: String) -> CustomCommand? {
        let key = (name.hasPrefix("/") ? String(name.dropFirst()) : name).lowercased()
        return commands.first { $0.name == key }
    }

    /// True for filenames that may become commands: non-empty, alphanumerics
    /// plus `-`/`_` (so a stray `README.md` or dotfile is skipped).
    public static func isValidName(_ s: String) -> Bool {
        !s.isEmpty && s.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" }
    }

    /// Parse one command file. An optional leading `--- ... ---` frontmatter
    /// block may set `description:`; the remainder is the template. When no
    /// description is given, the first non-empty body line is used.
    public static func parse(name: String, contents: String) -> CustomCommand {
        var description = ""
        var body = contents

        if contents.hasPrefix("---") {
            let lines = contents.components(separatedBy: "\n")
            if let end = lines.dropFirst().firstIndex(where: {
                $0.trimmingCharacters(in: .whitespaces) == "---"
            }) {
                for line in lines[1..<end] {
                    let kv = line.split(separator: ":", maxSplits: 1).map(String.init)
                    if kv.count == 2,
                       kv[0].trimmingCharacters(in: .whitespaces).lowercased() == "description" {
                        description = kv[1].trimmingCharacters(in: .whitespaces)
                    }
                }
                body = lines[(end + 1)...].joined(separator: "\n")
            }
        }

        let template = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if description.isEmpty {
            let first = template.components(separatedBy: "\n").first {
                !$0.trimmingCharacters(in: .whitespaces).isEmpty
            } ?? ""
            description = String(first.trimmingCharacters(in: .whitespaces).prefix(60))
            if description.isEmpty { description = "Custom command" }
        }
        return CustomCommand(name: name.lowercased(), description: description, template: template)
    }

    /// Load every `*.md` file in `directory` as a command. A missing directory
    /// or unreadable file yields an empty/partial store rather than an error.
    public static func load(from directory: URL) -> CustomCommandStore {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil) else {
            return CustomCommandStore(commands: [])
        }
        var cmds: [CustomCommand] = []
        for url in items where url.pathExtension.lowercased() == "md" {
            let name = url.deletingPathExtension().lastPathComponent
            guard isValidName(name),
                  let contents = try? String(contentsOf: url, encoding: .utf8) else { continue }
            cmds.append(parse(name: name, contents: contents))
        }
        return CustomCommandStore(commands: cmds)
    }
}
