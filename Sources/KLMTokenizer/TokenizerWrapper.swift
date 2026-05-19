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
    /// Supports model-specific templates:
    /// - Gemma 4: uses token IDs 105/106/107 for turn markers
    /// - Llama 3: uses <|begin_of_text|> style markers
    /// - Others: delegates to tokenizer's built-in template
    public func applyChatTemplate(messages: [[String: String]]) -> String {
        // Try using the tokenizer's built-in chat template (returns token IDs)
        if let tokenIds = try? tokenizer.applyChatTemplate(messages: messages) {
            return tokenizer.decode(tokens: tokenIds)
        }

        // Detect Gemma 4 by checking if token 105 decodes to a turn marker
        let token105 = tokenizer.decode(tokens: [105])
        if token105.contains("turn") || token105.contains("<|") {
            // Use direct token ID path (avoid decode→re-encode round-trip
            // which loses special tokens)
            return tokenizer.decode(tokens: formatGemma4TokenIds(messages: messages))
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

    /// Encode messages as Gemma 4 token IDs directly.
    ///
    /// Returns token IDs with correct special token IDs (105, 106, 107)
    /// that would be lost in a text decode→re-encode round-trip.
    ///
    /// Format: BOS + <|turn>user\ncontent<turn|>\n … <|turn>model\n
    ///
    /// Gemma 4 has only `user` and `model` turn roles. `assistant` maps to
    /// `model`; `system`/`tool`/`tools` fold into a `user` turn (emitting
    /// the literal role string for non-Gemma roles is a latent bug fixed
    /// here). Tool definitions (`tools`) and tool results (`tool`) are
    /// wrapped in the model's native tool special tokens so function
    /// calling matches the format Gemma 4 was fine-tuned on.
    public func formatGemma4TokenIds(messages: [[String: String]]) -> [Int] {
        // BOS=2, <|turn>=105, <turn|>=106, \n=107,
        // <|tool>=46, <tool|>=47, <|tool_response>=50, <tool_response|>=51
        // (added_tokens in this checkpoint's tokenizer.json; emitted as ids
        // because their text form does not round-trip through encode()).
        var tokens: [Int] = [2]  // BOS
        for msg in messages {
            let role = msg["role"] ?? "user"
            let content = msg["content"] ?? ""
            let turnRole = (role == "assistant") ? "model" : "user"
            tokens.append(105)  // <|turn>
            tokens += tokenizer.encode(text: turnRole)
            tokens.append(107)  // \n
            switch role {
            case "tools":  // tool definitions block
                tokens.append(46)
                tokens += tokenizer.encode(text: content)
                tokens.append(47)
            case "tool":   // tool result fed back
                tokens.append(50)
                tokens += tokenizer.encode(text: content)
                tokens.append(51)
            default:
                tokens += tokenizer.encode(text: content)
            }
            tokens.append(106)  // <turn|>
            tokens.append(107)  // \n
        }
        // Add model turn start
        tokens.append(105)  // <|turn>
        tokens += tokenizer.encode(text: "model")
        tokens.append(107)  // \n
        return tokens
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
