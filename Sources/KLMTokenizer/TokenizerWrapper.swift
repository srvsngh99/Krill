import Foundation
import Tokenizers

/// Wrapper around HuggingFace swift-transformers tokenizer.
///
/// Loads tokenizer.json from a model directory and provides encode/decode
/// plus chat template formatting for Llama 3.
public final class KLMTokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizer
    public let eosTokenId: Int
    /// Lowercased value of `tokenizer.json`'s `model.type` field
    /// (e.g. `"unigram"`, `"wordpiece"`, `"bpe"`). Captured at
    /// load time so callers can disambiguate behavior without
    /// guessing from the encoded special-token ids. Empty when
    /// the file is missing or unparseable.
    public let tokenizerModelKind: String

    /// Load tokenizer from a model directory containing tokenizer.json.
    public init(from directory: URL) async throws {
        let tokenizerConfig = directory.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerConfig.path) else {
            throw TokenizerError.missingFile("tokenizer.json", directory)
        }

        // Load via swift-transformers AutoTokenizer. Some
        // tokenizer_class values (e.g. XLMRobertaTokenizer used by
        // BGE Reranker v2-m3) are not in swift-transformers'
        // `knownTokenizers` registry; AutoTokenizer will throw
        // `unsupportedTokenizer`. For these, fall back to the
        // generic `PreTrainedTokenizer` which can drive the
        // tokenizer.json directly: the tokenizer.json file is the
        // universal HuggingFace format, so the BPE / SentencePiece
        // models and pre/post processors it declares are enough to
        // tokenize correctly without a model-specific wrapper.
        do {
            self.tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        } catch {
            // swift-transformers' TokenizerError is an enum without
            // a public Sendable error code; we cannot pattern-match
            // its associated value cleanly, so dispatch on the
            // error description. Any tokenizer that the
            // `knownTokenizers` registry rejects (e.g.
            // XLMRobertaTokenizer for BGE Reranker) falls into the
            // generic PreTrainedTokenizer path.
            let desc = String(describing: error)
            if desc.contains("unsupportedTokenizer")
                || desc.contains("missingTokenizerClassInConfig")
            {
                self.tokenizer = try await Self.loadAsPreTrainedTokenizer(
                    directory: directory)
            } else {
                throw error
            }
        }

        // Resolve EOS token ID
        if let eosId = self.tokenizer.eosTokenId {
            self.eosTokenId = eosId
        } else {
            // Llama 3 default EOS
            self.eosTokenId = 128001
        }

        // Capture tokenizer.json's `model.type` (e.g. "Unigram",
        // "WordPiece", "BPE") so callers can disambiguate
        // behavior without guessing from token IDs. Best-effort:
        // empty string when the file is missing or unreadable.
        self.tokenizerModelKind = Self.readModelKind(directory: directory)
    }

    private static func readModelKind(directory: URL) -> String {
        let url = directory.appendingPathComponent("tokenizer.json")
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = obj["model"] as? [String: Any],
              let type = model["type"] as? String else {
            return ""
        }
        return type.lowercased()
    }

    /// Stage a shadow directory containing tokenizer.json from the
    /// real model dir plus a tokenizer_config.json whose
    /// `tokenizer_class` field has been overridden to a value
    /// swift-transformers' `knownTokenizers` registry recognizes
    /// (`PreTrainedTokenizer`, backed by BPETokenizer). Loads
    /// AutoTokenizer from the shadow dir, which then reads the
    /// override and dispatches to the generic path.
    ///
    /// The override is safe for cross-encoder rerankers
    /// (XLMRobertaTokenizer) because the actual tokenization
    /// algorithm is fully described by the embedded
    /// `tokenizer.json` model + pre/post processors; the
    /// `tokenizer_class` field is just a dispatch hint, not a
    /// behavior specifier.
    private static func loadAsPreTrainedTokenizer(
        directory: URL
    ) async throws -> Tokenizer {
        let dataURL = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: dataURL.path) else {
            throw TokenizerError.missingFile("tokenizer.json", directory)
        }
        // Stage a temp directory we can drop in place of the real
        // model dir. AutoTokenizer reads tokenizer_config.json and
        // tokenizer.json from a single directory; we link
        // tokenizer.json (huge file) and write a modified
        // tokenizer_config.json next to it.
        let stagingDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: stagingDir, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: stagingDir)
        }

        // Symlink tokenizer.json so we do not duplicate a 17MB+ file.
        let stagedData = stagingDir.appendingPathComponent("tokenizer.json")
        try FileManager.default.createSymbolicLink(
            at: stagedData, withDestinationURL: dataURL)

        // Copy + override tokenizer_config.json. If the original is
        // missing entirely we write a minimal config with just the
        // overridden class.
        let realCfg = directory.appendingPathComponent("tokenizer_config.json")
        var configDict: [String: Any] = [:]
        if FileManager.default.fileExists(atPath: realCfg.path),
           let cfgData = try? Data(contentsOf: realCfg),
           let parsed = try? JSONSerialization.jsonObject(with: cfgData) as? [String: Any] {
            configDict = parsed
        }
        // Pick the override based on tokenizer.json's `model.type`.
        // swift-transformers' `knownTokenizers` registry maps:
        //   - `PreTrainedTokenizer` -> BPETokenizer
        //     (XLM-RoBERTa is NOT BPE, so this would crash with
        //     "requires merges").
        //   - `T5Tokenizer` -> UnigramTokenizer
        //     (XLM-RoBERTa, BGE Reranker, all SentencePiece Unigram
        //     models).
        //   - `BertTokenizer` -> BertTokenizer (WordPiece).
        let dataData = try Data(contentsOf: dataURL)
        var modelKind: String? = nil
        if let dataDict = try? JSONSerialization.jsonObject(
            with: dataData) as? [String: Any],
           let model = dataDict["model"] as? [String: Any] {
            modelKind = model["type"] as? String
        }
        let overrideClass: String
        switch modelKind {
        case "Unigram":
            overrideClass = "T5Tokenizer"
        case "WordPiece":
            overrideClass = "BertTokenizer"
        case "BPE":
            overrideClass = "PreTrainedTokenizer"
        default:
            overrideClass = "PreTrainedTokenizer"
        }
        configDict["tokenizer_class"] = overrideClass
        let outData = try JSONSerialization.data(
            withJSONObject: configDict, options: [])
        try outData.write(
            to: stagingDir.appendingPathComponent("tokenizer_config.json"))

        // Symlink the rest of the side files AutoTokenizer may
        // read (special_tokens_map.json, config.json - the latter
        // is consulted for fallback BOS/EOS/PAD ids when the
        // tokenizer config does not declare them).
        for sideFile in ["special_tokens_map.json", "config.json"] {
            let src = directory.appendingPathComponent(sideFile)
            if FileManager.default.fileExists(atPath: src.path) {
                try FileManager.default.createSymbolicLink(
                    at: stagingDir.appendingPathComponent(sideFile),
                    withDestinationURL: src)
            }
        }

        return try await AutoTokenizer.from(modelFolder: stagingDir)
    }

    /// The BOS token ID (beginning of sequence).
    public var bosTokenId: Int {
        tokenizer.bosTokenId ?? 128000
    }

    /// Encode a string to token IDs.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    /// Encode a (query, document) pair for a cross-encoder reranker.
    ///
    /// Cross-encoders consume the two strings as a single sequence with
    /// special-token separators between them. Two layouts cover all
    /// rerankers KrillLM currently ships:
    ///
    /// - XLMRoberta-class (BGE Reranker, etc.):
    ///   `<s> query </s></s> document </s>`
    /// - Bert-class:
    ///   `[CLS] query [SEP] document [SEP]`
    ///
    /// The tokenizer's `applyChatTemplate` does NOT apply here (rerankers
    /// have no chat template); the tokenizer's per-string `encode` adds
    /// its model-default leading/trailing specials. We use that as a
    /// building block: encode each side separately, strip the trailing
    /// special added to the LEFT side (it would land between the two
    /// halves), insert the pair separator, and let the RIGHT side carry
    /// the final special. The result matches the HuggingFace
    /// `tokenizer(query, document)` pair-encoding shape.
    public func encodePair(query: String, document: String) -> [Int] {
        // Per-side encodes carry the model's BOS and EOS specials.
        let qIds = tokenizer.encode(text: query)
        let dIds = tokenizer.encode(text: document)

        // Infer the bos/eos ids from what the tokenizer ACTUALLY
        // emitted. Asking the wrapper's `tokenizer.bosTokenId` is
        // unreliable for cross-encoder rerankers: the swift-
        // transformers AutoTokenizer dispatch sometimes loads them
        // through a path that does not expose BOS (e.g. T5
        // override for XLM-Roberta Unigram models), and our own
        // fallback would return Llama's BOS. Reading the first
        // and last tokens of an actual encode is the source of
        // truth.
        guard let bos = qIds.first, let eos = qIds.last else {
            return qIds + dIds
        }

        // Drop the trailing EOS on the query side (we will re-add
        // a tailored separator below). Keep the leading BOS.
        var qHead = qIds
        if qHead.last == eos {
            qHead.removeLast()
        }

        // Drop the leading BOS on the document side (it would
        // appear mid-sequence after the separator).
        var dTail = dIds
        if dTail.first == bos {
            dTail.removeFirst()
        }

        // Pair separator shape:
        //   - XLM-Roberta (Unigram tokenizer) -> `</s></s>` (two
        //     EOS tokens), matching HuggingFace
        //     `tokenizer(query, document)`.
        //   - Bert / DistilBert / sentence-bert cross-encoders
        //     (WordPiece tokenizer) -> single `[SEP]`.
        //   - BPE-class cross-encoders (rare today; e.g. some
        //     Cohere reranker variants) -> single EOS.
        //
        // `tokenizerModelKind` is captured at load time from
        // `tokenizer.json`'s `model.type` field, so we dispatch
        // on the actual tokenizer algorithm rather than guessing
        // from special-token id magnitudes.
        let separator: [Int]
        switch tokenizerModelKind {
        case "unigram":
            separator = [eos, eos]
        case "wordpiece", "bpe":
            separator = [eos]
        default:
            // Unknown / missing tokenizer.json: fall back to the
            // historical id-magnitude heuristic (XLM-R-class
            // specials are small ids; Bert-class are larger).
            separator = (bos < 10 && eos < 10) ? [eos, eos] : [eos]
        }

        return qHead + separator + dTail
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
