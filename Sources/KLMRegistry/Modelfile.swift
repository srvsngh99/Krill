import Foundation
import Logging

/// Parsed Ollama-compatible Modelfile (WS-C / T1-2).
///
/// Supported directives: `FROM`, `PARAMETER <k> <v>`, `SYSTEM`, `TEMPLATE`,
/// `LICENSE`, `MESSAGE <role> <content>`. `ADAPTER` (LoRA) is parsed and
/// recorded as a warning only - out of scope for v1 per the parity plan §7.
/// `SYSTEM`/`TEMPLATE`/`LICENSE` accept either a rest-of-line value or a
/// triple-quoted `"""…"""` multi-line block.
public struct Modelfile: Sendable, Equatable {
    public var from: String
    public var system: String?
    public var template: String?
    public var license: String?
    public var parameters: [String: String]
    public var messages: [[String: String]]
    public var adapterWarning: String?

    public func overrides() -> ModelOverrides {
        ModelOverrides(system: system, template: template, license: license,
                       parameters: parameters, messages: messages)
    }
}

public enum ModelfileError: Error, CustomStringConvertible {
    case missingFROM
    case empty

    public var description: String {
        switch self {
        case .missingFROM: return "Modelfile must contain a FROM directive"
        case .empty: return "Modelfile is empty"
        }
    }
}

public enum ModelfileParser {
    /// Tokenize-and-parse. Tolerant of comments (`#`), blank lines, and
    /// case-insensitive directives, matching Ollama's behavior.
    public static func parse(_ text: String) throws -> Modelfile {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ModelfileError.empty
        }
        var from: String?
        var system: String?
        var template: String?
        var license: String?
        var params: [String: String] = [:]
        var messages: [[String: String]] = []
        var adapterWarning: String?

        let scalars = Array(text.components(separatedBy: .newlines))
        var i = 0
        func readBlockOrRest(_ rest: String, startLine: Int) -> (String, Int) {
            let trimmed = rest.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("\"\"\"") {
                // Multi-line until closing """
                var body = String(trimmed.dropFirst(3))
                if let r = body.range(of: "\"\"\"") {  // single-line """x"""
                    return (String(body[..<r.lowerBound]), startLine)
                }
                var j = startLine + 1
                var collected = [body]
                while j < scalars.count {
                    if let r = scalars[j].range(of: "\"\"\"") {
                        collected.append(String(scalars[j][..<r.lowerBound]))
                        body = collected.joined(separator: "\n")
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        return (body, j)
                    }
                    collected.append(scalars[j])
                    j += 1
                }
                return (collected.joined(separator: "\n")
                    .trimmingCharacters(in: .whitespacesAndNewlines), j)
            }
            return (trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "\"")),
                    startLine)
        }

        while i < scalars.count {
            let raw = scalars[i]
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { i += 1; continue }

            let parts = line.split(separator: " ", maxSplits: 1).map(String.init)
            let directive = parts[0].uppercased()
            let rest = parts.count > 1 ? parts[1] : ""

            switch directive {
            case "FROM":
                from = rest.trimmingCharacters(in: .whitespaces)
            case "PARAMETER":
                let kv = rest.split(separator: " ", maxSplits: 1).map(String.init)
                if kv.count == 2 {
                    params[kv[0].lowercased()] = kv[1]
                        .trimmingCharacters(in: CharacterSet(charactersIn: "\" "))
                }
            case "SYSTEM":
                let (v, end) = readBlockOrRest(rest, startLine: i); system = v; i = end
            case "TEMPLATE":
                let (v, end) = readBlockOrRest(rest, startLine: i); template = v; i = end
            case "LICENSE":
                let (v, end) = readBlockOrRest(rest, startLine: i); license = v; i = end
            case "MESSAGE":
                let mk = rest.split(separator: " ", maxSplits: 1).map(String.init)
                if mk.count == 2 {
                    messages.append(["role": mk[0].lowercased(), "content": mk[1]])
                }
            case "ADAPTER":
                adapterWarning = "ADAPTER (LoRA) is not supported in v1 and was ignored"
            default:
                break  // unknown directive: ignore (Ollama-tolerant)
            }
            i += 1
        }

        guard let from, !from.isEmpty else { throw ModelfileError.missingFROM }
        return Modelfile(from: from, system: system, template: template,
                         license: license, parameters: params,
                         messages: messages, adapterWarning: adapterWarning)
    }
}

public enum ModelCreateError: Error, CustomStringConvertible {
    case baseNotInstalled(String)

    public var description: String {
        switch self {
        case .baseNotInstalled(let n):
            return "Base model '\(n)' is not installed. Pull it first: krillm pull \(n)"
        }
    }
}

extension Registry {
    /// Create a customized model from a parsed Modelfile. The base weights
    /// are *referenced* (the new blob dir is a symlink to the base) - no
    /// weight copy - with the Modelfile overrides recorded on the manifest.
    @discardableResult
    public func createModel(name: String, from modelfile: Modelfile) throws -> ModelManifest {
        let logger = Logger(label: "krillm.create")
        try Registry.requireValidName(name)
        let baseName = modelfile.from
        try Registry.requireValidName(baseName)
        guard let base = getModel(baseName) else {
            throw ModelCreateError.baseNotInstalled(baseName)
        }
        try ensureDirectories()

        let fm = FileManager.default
        let dst = modelPath(name)
        if fm.fileExists(atPath: dst.path) || isSymlink(dst) {
            try? fm.removeItem(at: dst)
        }
        // Reference base weights without copying: a real directory whose
        // entries are per-file symlinks into the base. (A directory-level
        // symlink is not traversed by the weight loader's directory
        // enumeration, so we link the files individually.)
        let baseDir = modelPath(baseName).resolvingSymlinksInPath()
        try fm.createDirectory(at: dst, withIntermediateDirectories: true)
        for entry in (try? fm.contentsOfDirectory(atPath: baseDir.path)) ?? [] {
            try? fm.createSymbolicLink(
                at: dst.appendingPathComponent(entry),
                withDestinationURL: baseDir.appendingPathComponent(entry))
        }

        if let w = modelfile.adapterWarning { logger.warning("\(w)") }

        let manifest = ModelManifest(
            name: name, family: base.family, params: base.params,
            quant: base.quant, source: "modelfile:\(baseName)",
            context: base.context, files: base.files,
            draftPair: base.draftPair, chatTemplate: base.chatTemplate,
            sizeBytes: base.sizeBytes, pulledAt: Date(),
            overrides: modelfile.overrides())
        try saveManifest(manifest)
        logger.info("Created \(name) from \(baseName)")
        return manifest
    }

    private func isSymlink(_ url: URL) -> Bool {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.type]
            as? FileAttributeType) == .typeSymbolicLink
    }
}
