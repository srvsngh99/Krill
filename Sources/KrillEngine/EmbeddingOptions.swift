import Foundation
import KrillCore

/// Optional per-request embedding controls shared by `/v1/embeddings`,
/// `/api/embed` and `/api/embeddings`.
///
/// - `dimensions` (OpenAI-style): Matryoshka truncation. EmbeddingGemma 2
///   accepts 768/512/256/128; the engine truncates then re-normalises.
/// - `task`: selects a sentence-transformers prompt prefix (for
///   EmbeddingGemma 2: `SearchQuery`, `Document`, `Classification`, ...).
/// - `instruction`: a literal prefix (pre-existing field, unchanged).
///
/// `task` and `instruction` are mutually exclusive. With neither, no prefix
/// is added (the model still works; the card says precision drops slightly).
public struct EmbeddingRequestOptions: Equatable, Sendable {
    public var dimensions: Int?
    public var task: String?
    public var instruction: String?

    public init(dimensions: Int? = nil, task: String? = nil, instruction: String? = nil) {
        self.dimensions = dimensions
        self.task = task
        self.instruction = instruction
    }

    public struct ParseError: Error, Equatable, CustomStringConvertible {
        public let message: String
        public var description: String { message }
    }

    /// Parse the optional fields out of a request body. Throws `ParseError`
    /// (a client error, HTTP 400) for malformed values.
    public static func parse(_ json: [String: Any]) throws -> EmbeddingRequestOptions {
        var o = EmbeddingRequestOptions()
        if let raw = json["dimensions"], !(raw is NSNull) {
            // JSONSerialization yields NSNumber; reject booleans and fractions.
            guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
                  n.doubleValue == n.doubleValue.rounded(), n.intValue > 0 else {
                throw ParseError(message: "'dimensions' must be a positive integer")
            }
            o.dimensions = n.intValue
        }
        if let raw = json["task"], !(raw is NSNull) {
            guard let s = raw as? String, !s.isEmpty else {
                throw ParseError(message: "'task' must be a non-empty string")
            }
            o.task = s
        }
        if let s = json["instruction"] as? String, !s.isEmpty { o.instruction = s }
        if o.task != nil, o.instruction != nil {
            throw ParseError(message: "'task' and 'instruction' are mutually exclusive")
        }
        return o
    }
}

/// Sentence-transformers prompt table (`config_sentence_transformers.json`
/// -> `prompts`). Lookup is exact first, then case-insensitive.
public struct EmbeddingPromptTable: Equatable, Sendable {
    public let prompts: [String: String]

    public init(prompts: [String: String]) { self.prompts = prompts }

    /// Load from a model directory; nil when the file or table is absent.
    public static func load(directory: URL) -> EmbeddingPromptTable? {
        let url = directory.appendingPathComponent("config_sentence_transformers.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let p = json["prompts"] as? [String: String], !p.isEmpty else { return nil }
        return EmbeddingPromptTable(prompts: p)
    }

    public func prefix(for task: String) -> String? {
        if let p = prompts[task] { return p }
        return prompts.first { $0.key.lowercased() == task.lowercased() }?.value
    }

    public var taskNames: [String] { prompts.keys.sorted() }
}
