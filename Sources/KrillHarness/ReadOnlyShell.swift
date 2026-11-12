import Foundation

/// Allowlist classifier for shell commands that cannot change anything.
/// BashTool runs `/bin/sh -c`, so anything not provably read-only is rejected.
enum ReadOnlyShell {
    private struct Rule {
        var clusters = ""
        var longs: [String] = []
        var exact: Set<String> = []
    }

    private static let rules: [String: Rule] = {
        var r = Dictionary(
            uniqueKeysWithValues: [
                "ls", "cat", "head", "tail", "wc", "grep", "pwd", "stat", "du", "df", "which",
                "cut", "tr", "diff", "cmp", "basename", "dirname", "realpath", "uname", "echo",
            ].map { ($0, Rule()) })
        r["rg"] = Rule(
            clusters: "z",
            longs: ["--pre", "--pre-glob", "--search-zip", "--hostname-bin"])
        r["find"] = Rule(exact: [
            "-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf",
            "-fls",
        ])
        r["file"] = Rule(clusters: "C", longs: ["--compile"])
        r["tree"] = Rule(clusters: "o", longs: ["--output"])
        r["sort"] = Rule(clusters: "o", longs: ["--output", "--compress-program"])
        return r
    }()

    private static let gitSubcommands: Set<String> = [
        "status", "log", "show", "diff", "rev-parse", "ls-files", "blame", "grep", "describe",
        "shortlog", "branch",
    ]
    private static let gitBranchListing: Set<String> = [
        "-a", "-r", "-v", "-vv", "--list", "--all", "--remotes", "--show-current",
    ]
    private static let gitLongs = [
        "--output", "--ext-diff", "--textconv", "--open-files-in-pager", "--exec-path",
    ]
    private static let forbiddenUnquoted = Set("$`\\;&<>(){}*?[".unicodeScalars)

    /// Nil when `command` is read-only; otherwise why it was rejected.
    static func violation(_ command: String) -> String? {
        let stages: [[String]]
        do { stages = try parse(command) } catch let e as Rejected { return e.why } catch { return "unparseable" }
        for words in stages {
            if let why = check(words) { return why }
        }
        return nil
    }

    private struct Rejected: Error { let why: String }

    private static func parse(_ command: String) throws -> [[String]] {
        if command.unicodeScalars.contains(where: { $0.value < 0x20 && $0 != "\t" }) {
            throw Rejected(why: "newlines and control characters are not allowed")
        }
        var stages: [[String]] = []
        var words: [String] = []
        var word = String.UnicodeScalarView()
        var inWord = false
        var quote: Unicode.Scalar?
        func flush() { if inWord { words.append(String(word)); word = .init(); inWord = false } }
        func endStage() throws {
            flush()
            if words.isEmpty { throw Rejected(why: "empty command or pipeline stage") }
            stages.append(words)
            words = []
        }
        // Scalars, not Characters: sh sees bytes, and a combining mark would hide a quote.
        for ch in command.unicodeScalars {
            if quote == "'" {
                if ch == "'" { quote = nil } else { word.append(ch) }
            } else if quote == "\"" {
                if ch == "\"" { quote = nil }
                else if "$`\\".unicodeScalars.contains(ch) { throw Rejected(why: "'\(ch)' is not allowed in double quotes") }
                else { word.append(ch) }
            } else if ch == "'" || ch == "\"" {
                quote = ch
                inWord = true
            } else if ch == " " || ch == "\t" {
                flush()
            } else if ch == "|" {
                try endStage()
            } else if forbiddenUnquoted.contains(ch) {
                throw Rejected(why: "'\(ch)' is not allowed outside single quotes")
            } else {
                word.append(ch)
                inWord = true
            }
        }
        if quote != nil { throw Rejected(why: "unbalanced quote") }
        try endStage()
        return stages
    }

    private static func check(_ words: [String]) -> String? {
        let name = words[0]
        let args = Array(words.dropFirst())
        switch name {
        case "git": return checkGit(args)
        case "date":
            for t in args where t.unicodeScalars.first != "-" && t.unicodeScalars.first != "+" {
                return "'date \(t)' could set the clock; only +FORMAT is allowed"
            }
            return flagViolation("date", args, Rule(clusters: "sf", longs: ["--set"]))
        default:
            guard let rule = rules[name] else { return "'\(name)' is not an allowed read-only command" }
            return flagViolation(name, args, rule)
        }
    }

    private static func checkGit(_ args: [String]) -> String? {
        guard let sub = args.first, gitSubcommands.contains(sub) else {
            return "'git \(args.first ?? "")' is not an allowed read-only git subcommand"
        }
        let rest = Array(args.dropFirst())
        if sub == "branch", let bad = rest.first(where: { !gitBranchListing.contains($0) }) {
            return "'git branch \(bad)' is not allowed; only listing (-a, -r, -v, --list) is"
        }
        return flagViolation("git \(sub)", rest, Rule(clusters: "O", longs: gitLongs))
    }

    private static func flagViolation(_ name: String, _ args: [String], _ rule: Rule) -> String? {
        for t in args {
            if rule.exact.contains(t) { return "'\(name) \(t)' can run commands or write files" }
            let scalars = Array(t.unicodeScalars)
            if scalars.starts(with: "--".unicodeScalars) {
                let flag = String(String.UnicodeScalarView(scalars.prefix { $0 != "=" }))
                if flag.unicodeScalars.count > 2, rule.longs.contains(where: { $0.hasPrefix(flag) }) {
                    return "'\(name) \(flag)' can run commands or write files"
                }
            } else if scalars.first == "-", scalars.dropFirst().contains(where: { rule.clusters.unicodeScalars.contains($0) }) {
                return "'\(name) \(t)' can run commands or write files"
            }
        }
        return nil
    }
}
