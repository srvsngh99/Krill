import Foundation
import Tokenizers

/// Wrapper around HuggingFace swift-transformers tokenizer.
///
/// Loads tokenizer.json from a model directory and provides encode/decode
/// plus chat template formatting for Llama 3.
public final class KLMTokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizer
    public let eosTokenId: Int

    /// Load tokenizer from a model directory containing tokenizer.json.
    public init(from directory: URL) async throws {
        let tokenizerConfig = directory.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerConfig.path) else {
            throw TokenizerError.missingFile("tokenizer.json", directory)
        }

        // Load via swift-transformers AutoTokenizer
        self.tokenizer = try await AutoTokenizer.from(modelFolder: directory)

        // Resolve EOS token ID
        if let eosId = self.tokenizer.eosTokenId {
            self.eosTokenId = eosId
        } else {
            // Llama 3 default EOS
            self.eosTokenId = 128001
        }
    }

    /// The BOS token ID (beginning of sequence).
    public var bosTokenId: Int {
        tokenizer.bosTokenId ?? 128000
    }

    /// Encode a string to token IDs.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    /// Encode text that already includes special tokens (e.g., chat template output).
    /// Strips the auto-added BOS token to avoid duplication.
    public func encodeWithoutExtraBOS(_ text: String) -> [Int] {
        var tokens = tokenizer.encode(text: text)
        // If the tokenizer auto-added BOS and the text already starts with <|begin_of_text|>,
        // we get a double BOS. Strip the first one.
        if tokens.count >= 2 && tokens[0] == bosTokenId && tokens[1] == bosTokenId {
            tokens.removeFirst()
        }
        return tokens
    }

    /// Decode token IDs back to a string.
    public func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    /// Decode a single token ID to its string representation.
    public func decode(token: Int) -> String {
        tokenizer.decode(tokens: [token])
    }

    /// Apply chat template formatting for a conversation.
    ///
    /// Formats messages as the model expects them (e.g., Llama 3 instruct format).
    /// Uses the tokenizer's built-in applyChatTemplate (returns token IDs) then decodes,
    /// or falls back to manual Llama 3 format.
    public func applyChatTemplate(messages: [[String: String]]) -> String {
        // Try using the tokenizer's built-in chat template (returns token IDs)
        if let tokenIds = try? tokenizer.applyChatTemplate(messages: messages) {
            return tokenizer.decode(tokens: tokenIds)
        }

        // Fallback: manual Llama 3 instruct format
        var result = "<|begin_of_text|>"
        for msg in messages {
            let role = msg["role"] ?? "user"
            let content = msg["content"] ?? ""
            result += "<|start_header_id|>\(role)<|end_header_id|>\n\n\(content)<|eot_id|>"
        }
        result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return result
    }
}

// MARK: - Errors

public enum TokenizerError: Error, CustomStringConvertible {
    case missingFile(String, URL)

    public var description: String {
        switch self {
        case .missingFile(let name, let dir):
            return "Missing \(name) in \(dir.path)"
        }
    }
}
