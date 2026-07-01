import Foundation
import Tokenizers
import Jinja

/// Wrapper around HuggingFace swift-transformers tokenizer.
///
/// Loads tokenizer.json from a model directory and provides encode/decode
/// plus chat template formatting for Llama 3.
public final class KrillTokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizer
    public let eosTokenId: Int
    /// Lowercased value of `tokenizer.json`'s `model.type` field
    /// (e.g. `"unigram"`, `"wordpiece"`, `"bpe"`). Captured at
    /// load time so callers can disambiguate behavior without
    /// guessing from the encoded special-token ids. Empty when
    /// the file is missing or unparseable.
    public let tokenizerModelKind: String
    /// Model vocabulary size, read from `config.json`'s `vocab_size`
    /// (falling back to the entry count in `tokenizer.json`'s model
    /// vocab). Used to size the grammar logit mask to the logits width.
    /// `nil` when neither file declares it.
    public let vocabSize: Int?
    /// Token IDs whose decoded text is suppressed (rendered empty) by
    /// `decodeForOutput` so structural Gemma-4 media markers cannot leak into
    /// the visible answer. Resolved once at load from `gemmaMediaMarkerLiterals`
    /// (empty for every tokenizer that lacks them as dedicated special tokens).
    public let outputSuppressedTokenIDs: Set<Int>
    /// Contents of `chat_template.jinja` from the model directory, if
    /// present. Newer HF checkpoints (Qwen 3 Coder, Qwen 3 Instruct-2507,
    /// Gemma 4) ship the chat template as a separate Jinja file instead
    /// of embedding it under `tokenizer_config.json["chat_template"]`;
    /// swift-transformers only reads the embedded form, so we load the
    /// file here once and pass it through to the library as a literal
    /// argument on every `applyChatTemplate(messages:)` call when the
    /// embedded path is empty. Nil for checkpoints that don't ship
    /// the file (the older embedded-template repos work unchanged).
    private let externalChatTemplate: String?
    /// The chat template embedded in `tokenizer_config.json` (`chat_template`),
    /// when it is a plain string. Captured so the engine can render the template
    /// itself with extra context like `enable_thinking` (which the upstream
    /// `applyChatTemplate` cannot pass through). `nil` when absent or a named-list
    /// form. `chatTemplateString` prefers the external `.jinja` file over this.
    private let embeddedChatTemplate: String?
    /// Special-token wrap applied manually for tokenizers whose
    /// `RobertaProcessing`/`BertProcessing` post-processor was stripped at load
    /// (swift-transformers fatal-errors parsing its sep/cls shape). `encode`
    /// wraps `[cls] .. [sep]`; `encodePair` uses `pairDoubleSep` to pick the
    /// cross-encoder separator (`</s></s>` for RoBERTa-class, `[SEP]` for
    /// Bert-class). `nil` for the common case where the library handles wrapping.
    private let specialWrap: (cls: Int, sep: Int, pairDoubleSep: Bool)?

    /// Load tokenizer from a model directory containing tokenizer.json.
    public init(from directory: URL) async throws {
        let tokenizerConfig = directory.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerConfig.path) else {
            throw TokenizerError.missingFile("tokenizer.json", directory)
        }

        // swift-transformers' RobertaProcessing/BertProcessing post-processor
        // parser fatal-errors (uncatchable) on the standard `["</s>", 2]` sep/cls
        // JSON shape - it expects a Swift tuple. For those tokenizers (RoBERTa,
        // XLM-R, MPNet, and the sentence-transformers built on them) load from a
        // temp copy with the post-processor stripped, and re-add the `[cls]..[sep]`
        // wrap in `encode`. Other tokenizers load from the original directory.
        let (loadDir, wrap) = Self.sanitizedTokenizerDirectory(directory)
        self.specialWrap = wrap

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
            self.tokenizer = try await AutoTokenizer.from(modelFolder: loadDir)
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
                    directory: loadDir)
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

        // Capture a separate `chat_template.jinja` if the checkpoint
        // ships one. Best-effort, no-op when absent.
        self.externalChatTemplate = Self.readExternalChatTemplate(directory: directory)
        self.embeddedChatTemplate = Self.readEmbeddedChatTemplate(directory: directory)

        // Model vocab size (for sizing the grammar logit mask). Best-effort.
        self.vocabSize = Self.readVocabSize(directory: directory)

        // Resolve the media-marker suppression set once (no-op on non-Gemma
        // tokenizers). Computed eagerly so the streaming decode path stays
        // lock-free under concurrent generation.
        self.outputSuppressedTokenIDs = Self.resolveOutputSuppressedTokenIDs(
            tokenizer: self.tokenizer)
    }

    /// Normalize a tokenizer.json for swift-transformers when it trips one of two
    /// known gaps, returning a temp directory with the rewritten file (plus the
    /// `(cls, sep)` ids to wrap manually in `encode` when a post-processor was
    /// stripped). Returns the original directory and `nil` when neither applies.
    /// Only the small tokenizer input files are copied; weights are untouched.
    ///
    /// Fix 1 (post-processor): `RobertaProcessing`/`BertProcessing` (whose
    /// `["</s>", 2]` sep/cls shape swift-transformers cannot parse) is stripped
    /// and re-applied as a `[cls]..[sep]` wrap in `encode`.
    ///
    /// Fix 2 (Metaspace prefix): new-style Metaspace configs declare
    /// `prepend_scheme` but omit `add_prefix_space` (e.g.
    /// nomic-embed-text-v2-moe). swift-transformers gates the whole `▁` prepend
    /// behind `add_prefix_space` (defaulting false when absent), so it never
    /// prepends and every word matches the wrong, non-word-initial vocab entries
    /// (garbage embeddings). Inject `add_prefix_space: true` so the existing
    /// prepend-scheme logic runs; the scheme value still governs behavior, so
    /// "never" stays a no-op. Tokenizers that already set the field (e.g.
    /// bge-reranker-v2-m3) are untouched, so the reranker is unaffected.
    private static func sanitizedTokenizerDirectory(
        _ directory: URL
    ) -> (URL, (cls: Int, sep: Int, pairDoubleSep: Bool)?) {
        let tjURL = directory.appendingPathComponent("tokenizer.json")
        guard let data = try? Data(contentsOf: tjURL),
              var obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return (directory, nil)
        }

        // Fix 1: strip a RobertaProcessing/BertProcessing post-processor.
        var wrap: (cls: Int, sep: Int, pairDoubleSep: Bool)?
        if let pp = obj["post_processor"] as? [String: Any],
           let type = pp["type"] as? String,
           type == "RobertaProcessing" || type == "BertProcessing" {
            func idOf(_ key: String) -> Int? {
                guard let pair = pp[key] as? [Any], pair.count == 2 else { return nil }
                return (pair[1] as? Int) ?? (pair[1] as? NSNumber)?.intValue
            }
            if let cls = idOf("cls"), let sep = idOf("sep") {
                // RoBERTa pairs use a doubled separator (`</s></s>`); Bert a single
                // `[SEP]`. Captured for `encodePair` since the post-processor drops.
                wrap = (cls, sep, type == "RobertaProcessing")
                obj.removeValue(forKey: "post_processor")
            }
        }

        // Fix 2: inject add_prefix_space on new-style Metaspace pretokenizers.
        var metaspaceFixed = false
        if var pre = obj["pre_tokenizer"] as? [String: Any] {
            metaspaceFixed = injectMetaspaceAddPrefixSpace(&pre)
            if metaspaceFixed { obj["pre_tokenizer"] = pre }
        }

        // Neither fix applied: load straight from the original directory.
        if wrap == nil && !metaspaceFixed {
            return (directory, nil)
        }

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-tok-\(UUID().uuidString)")
        do {
            try FileManager.default.createDirectory(
                at: tmp, withIntermediateDirectories: true)
            let sanitized = try JSONSerialization.data(withJSONObject: obj)
            try sanitized.write(to: tmp.appendingPathComponent("tokenizer.json"))
            // AutoTokenizer also reads these sibling files when present.
            for name in [
                "tokenizer_config.json", "special_tokens_map.json", "config.json",
                "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
                "tokenizer.model",
            ] {
                let src = directory.appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: src.path) {
                    try? FileManager.default.copyItem(
                        at: src, to: tmp.appendingPathComponent(name))
                }
            }
        } catch {
            return (directory, nil)
        }
        return (tmp, wrap)
    }

    /// Recursively set `add_prefix_space: true` on any Metaspace pretokenizer
    /// (including those nested in a `Sequence`) that declares `prepend_scheme`
    /// but omits `add_prefix_space`. Returns true when any node was changed. See
    /// `sanitizedTokenizerDirectory` Fix 2 for why. Internal for testability.
    static func injectMetaspaceAddPrefixSpace(_ node: inout [String: Any]) -> Bool {
        var changed = false
        let type = node["type"] as? String
        if type == "Metaspace", node["prepend_scheme"] != nil, node["add_prefix_space"] == nil {
            node["add_prefix_space"] = true
            changed = true
        }
        if type == "Sequence", var subs = node["pretokenizers"] as? [[String: Any]] {
            for i in subs.indices where injectMetaspaceAddPrefixSpace(&subs[i]) {
                changed = true
            }
            if changed { node["pretokenizers"] = subs }
        }
        return changed
    }

    private static func readVocabSize(directory: URL) -> Int? {
        // Preferred source: config.json's `vocab_size` — this matches the
        // model's lm_head output width (the logits the mask must align to).
        let cfg = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: cfg),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let v = obj["vocab_size"] as? Int, v > 0 {
            return v
        }
        // Fallback: the entry count of tokenizer.json's model vocab
        // (BPE/WordPiece use a `{token: id}` dict; Unigram uses a
        // `[[token, score], ...]` array).
        let tok = directory.appendingPathComponent("tokenizer.json")
        if let data = try? Data(contentsOf: tok),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let model = obj["model"] as? [String: Any] {
            if let vocab = model["vocab"] as? [String: Any] { return vocab.count }
            if let vocab = model["vocab"] as? [Any] { return vocab.count }
        }
        return nil
    }

    internal static func readExternalChatTemplate(directory: URL) -> String? {
        let url = directory.appendingPathComponent("chat_template.jinja")
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8),
              !text.isEmpty else {
            return nil
        }
        return text
    }

    /// Read the chat template embedded in `tokenizer_config.json` (`chat_template`),
    /// when it is a plain string (the common Qwen/Llama/etc. form). Returns nil for
    /// the named-list form or when absent. Used so the engine can render the
    /// template with `enable_thinking`.
    internal static func readEmbeddedChatTemplate(directory: URL) -> String? {
        let url = directory.appendingPathComponent("tokenizer_config.json")
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let template = obj["chat_template"] as? String,
              !template.isEmpty else {
            return nil
        }
        return template
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
            .appendingPathComponent("krill-tokenizer-\(UUID().uuidString)")
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
        let ids = tokenizer.encode(text: text)
        // Re-add the special wrap the library would have applied if its
        // post-processor had not been stripped at load (see `specialWrap`).
        if let w = specialWrap {
            return [w.cls] + ids + [w.sep]
        }
        return ids
    }

    /// Encode a (query, document) pair for a cross-encoder reranker.
    ///
    /// Cross-encoders consume the two strings as a single sequence with
    /// special-token separators between them. Two layouts cover all
    /// rerankers Krill currently ships:
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

        // When the post-processor was stripped at load (`specialWrap` set), the
        // per-side encodes have NO specials, so the first/last-token inference
        // below would be wrong. Build the pair layout explicitly from the known
        // cls/sep instead: `[cls] q <sep..> d [sep]`.
        if let w = specialWrap {
            let separator = w.pairDoubleSep ? [w.sep, w.sep] : [w.sep]
            return [w.cls] + qIds + separator + dIds + [w.sep]
        }

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

    /// Gemma-4 multimodal marker literals (image/audio soft tokens plus the
    /// begin/end markers — mirrors `InferenceEngine.gemma4*`). They are
    /// structural special tokens that demarcate a media run in the prompt; the
    /// model occasionally emits one (notably `<image|>` = `<end_of_image>`)
    /// during plain text decode, where the byte-level detokenizer renders the
    /// literal into the visible answer. They must never reach the user.
    private static let gemmaMediaMarkerLiterals = [
        "<|image|>", "<|audio|>", "<|image>", "<|audio>", "<image|>", "<audio|>",
    ]

    /// Resolve `outputSuppressedTokenIDs` from `gemmaMediaMarkerLiterals`: a
    /// literal contributes the id of the single token that, decoded on its own,
    /// reproduces the literal exactly. Matching on each token's standalone
    /// decode (rather than the whole encoding) makes this independent of any
    /// BOS/EOS/metaspace tokens the tokenizer wraps around the literal, so it
    /// neither misses the marker when a checkpoint appends specials nor fires on
    /// a non-Gemma tokenizer that splits the literal into prose sub-pieces
    /// (none of which decode back to the literal). Ambiguous cases (zero or more
    /// than one matching token) are skipped, keeping `decodeForOutput` a
    /// transparent pass-through everywhere it is not unambiguously a marker.
    private static func resolveOutputSuppressedTokenIDs(
        tokenizer: Tokenizer
    ) -> Set<Int> {
        var ids = Set<Int>()
        for literal in gemmaMediaMarkerLiterals {
            let matches = tokenizer.encode(text: literal)
                .filter { tokenizer.decode(tokens: [$0]) == literal }
            if matches.count == 1 {
                ids.insert(matches[0])
            }
        }
        return ids
    }

    /// Decode a single generated token for VISIBLE output. Identical to
    /// `decode(token:)` except that structural media-marker special tokens
    /// (see `outputSuppressedTokenIDs`) decode to the empty string instead of
    /// their literal form, so they cannot leak into the streamed answer.
    public func decodeForOutput(token: Int) -> String {
        if outputSuppressedTokenIDs.contains(token) { return "" }
        return tokenizer.decode(tokens: [token])
    }

    /// Apply chat template formatting for a conversation.
    ///
    /// Supports model-specific templates:
    /// - Gemma 4: uses token IDs 105/106/107 for turn markers
    /// - Llama 3: uses <|begin_of_text|> style markers
    /// - Others: delegates to tokenizer's built-in template
    /// Render `messages` directly to token IDs via the embedded or
    /// external chat template, skipping the lossy decode -> re-encode
    /// round-trip that the legacy `applyChatTemplate(messages:) ->
    /// String` path takes (which silently mangles ChatML / FIM /
    /// tool special tokens when their text form does not survive
    /// re-tokenization).
    ///
    /// Returns nil when no chat template is available; callers fall
    /// back to the string path, which still drives the Gemma 4 manual
    /// path (direct IDs) and the Llama 3 manual fallback.
    public func applyChatTemplateTokens(messages: [[String: String]]) -> [Int]? {
        if let tokenIds = try? tokenizer.applyChatTemplate(messages: messages) {
            return tokenIds
        }
        if let template = externalChatTemplate,
           let tokenIds = try? tokenizer.applyChatTemplate(
               messages: messages, chatTemplate: template) {
            return tokenIds
        }
        return nil
    }

    /// The effective chat template string (external `.jinja` file preferred over
    /// the `tokenizer_config.json` embedded one), or nil if neither is available.
    public var chatTemplateString: String? { externalChatTemplate ?? embeddedChatTemplate }

    /// Pure detection of whether a chat template exposes a reasoning ("thinking")
    /// channel the engine can turn on: the Gemma-4 channel markers, or any
    /// template that branches on `enable_thinking` (Qwen 3, other reasoning
    /// fine-tunes). Static + side-effect-free so it is unit-testable without a
    /// loaded tokenizer.
    public static func templateSupportsThinking(externalTemplate: String?,
                                                embeddedTemplate: String?) -> Bool {
        if let ext = externalTemplate, ext.contains("<|channel>") || ext.contains("<|turn>") {
            return true  // Gemma-4 channel template
        }
        let effective = externalTemplate ?? embeddedTemplate
        return effective?.contains("enable_thinking") ?? false
    }

    /// Tokens for a prompt with the `enable_thinking` template variable pinned to
    /// `on` (BOTH directions), or nil when this isn't applicable (the caller then
    /// handles it: the Gemma-4 channel template via `gemma4ChannelPrompt`,
    /// everything else via the normal path).
    ///
    /// Pinning matters because a model's template may DEFAULT thinking on (Qwen 3
    /// does): turning it off then requires rendering with `enable_thinking=false`
    /// explicitly - the normal path would inherit the on-by-default and never
    /// disable. We render a FRESH string and encode it with the special-token-aware
    /// `tokenizer.encode` (NOT by decoding existing ids), the same render+encode
    /// swift-transformers' own direct path does, so ChatML special tokens stay
    /// intact. The Gemma-4 channel template is EXCLUDED explicitly (not left to a
    /// Jinja parse failure) so that path deterministically uses `gemma4ChannelPrompt`.
    public func enableThinkingPrompt(messages: [[String: String]], on: Bool) -> [Int]? {
        guard !usesGemmaChannelTemplate,
              chatTemplateString?.contains("enable_thinking") == true,
              let rendered = renderTemplate(
                  messages: messages, extraContext: ["enable_thinking": on]) else {
            return nil
        }
        return encodeWithoutExtraBOS(rendered)
    }

    /// Render the effective chat template through Jinja with extra context merged
    /// over the standard (`messages`, `add_generation_prompt`, bos/eos) context.
    /// Returns nil when there is no template or it fails to parse/render (the
    /// caller then falls back to the normal prompt - never a hard failure).
    public func renderTemplate(messages: [[String: String]],
                               extraContext: [String: Any?]) -> String? {
        guard let template = chatTemplateString,
              let compiled = try? Template(template) else { return nil }
        var context: [String: Any?] = [
            "messages": messages,
            "add_generation_prompt": true,
            "bos_token": decode(token: bosTokenId),
            "eos_token": decode(token: eosTokenId),
        ]
        for (k, v) in extraContext { context[k] = v }
        return try? compiled.render(context)
    }

    /// True when the model ships the Gemma-4 "channel" chat template (the
    /// `<|turn>` / `<|channel>thought` reasoning format used by the coder
    /// fine-tune), as opposed to the stock Gemma-4 `<start_of_turn>` format. The
    /// Swift Jinja port cannot parse this template's macros, so the engine builds
    /// the prompt directly via ``gemma4ChannelPrompt(messages:enableThinking:)``
    /// for these models instead of the stock direct-id builder.
    public var usesGemmaChannelTemplate: Bool {
        guard let t = externalChatTemplate else { return false }
        return t.contains("<|channel>") || t.contains("<|turn>")
    }

    /// Build the Gemma-4 "channel" prompt for reasoning fine-tunes (e.g. the
    /// coder) whose `chat_template.jinja` gates chain-of-thought on
    /// `enable_thinking`. We reproduce BOTH renders directly as a string (the
    /// Swift Jinja port cannot parse this template) and re-encode - the marker
    /// strings (`<|turn>`, `<|think|>`, `<|channel>`, `<turn|>`) tokenize to their
    /// special-token ids. Both forms verified against the upstream template:
    ///   - `enableThinking`: a leading `<|turn>system\n<|think|>` block, then the
    ///     turns, then `<|turn>model\n` (the model reasons freely).
    ///   - else: the turns, then `<|turn>model\n<|channel>thought\n<channel|>` -
    ///     the empty thought channel the template appends so the model answers
    ///     WITHOUT reasoning (matching the SKU's training distribution; the stock
    ///     direct-id builder omits it).
    /// A system block is also emitted (without `<|think|>`) when a system message
    /// is present, mirroring the template's `messages[0] is system` branch.
    public func gemma4ChannelPrompt(messages: [[String: String]],
                                    enableThinking: Bool) -> String {
        var out = decode(token: bosTokenId)
        let systemContent = messages
            .filter { ($0["role"] ?? "") == "system" }
            .map { ($0["content"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        if enableThinking || !systemContent.isEmpty {
            out += "<|turn>system\n"
            if enableThinking { out += "<|think|>\n" }
            out += systemContent
            out += "<turn|>\n"
        }
        for m in messages where (m["role"] ?? "") != "system" {
            let role = (m["role"] ?? "user") == "assistant" ? "model" : (m["role"] ?? "user")
            out += "<|turn>\(role)\n\(m["content"] ?? "")<turn|>\n"
        }
        out += "<|turn>model\n"
        if !enableThinking { out += "<|channel>thought\n<channel|>" }
        return out
    }

    /// True when the model's effective chat template is ChatML-based (contains
    /// the `<|im_start|>` turn marker) — a Qwen / Qwen 3.5 (Ornith) / Hermes
    /// checkpoint. Used to pick the correct manual fallback when the Swift Jinja
    /// port cannot render the model's `chat_template.jinja`. Ornith-9B's template,
    /// for example, captures a macro's return value via
    /// `{% set x = render_content(...) %}`, which the port does not support; the
    /// render fails, and without this branch the generic Llama-3 fallback below
    /// emits `<|eot_id|>` turn terminators. Those are NOT this model's EOS
    /// (`<|im_end|>`), so the model echoes them as text and generation never
    /// stops — it runs on, fabricating turn after turn, until the token cap.
    public var usesChatMLTemplate: Bool {
        chatTemplateString?.contains("<|im_start|>") ?? false
    }

    /// Build a plain ChatML prompt string for a ChatML model whose
    /// `chat_template.jinja` could not be rendered by the Swift Jinja port.
    /// Mirrors the turn structure of the proven native ChatML builder
    /// (`formatQwen25VLTokenIds`): `<|im_start|>{role}\n{content}<|im_end|>\n`
    /// per message, then a trailing `<|im_start|>assistant\n` to cue generation.
    /// The marker strings tokenize to the checkpoint's real ChatML ids through
    /// the caller's `encodeWithoutExtraBOS`, so the assistant turn terminates on
    /// the model's true `<|im_end|>` EOS. ChatML roles are system / user /
    /// assistant / tool; anything else (e.g. a `tools` definitions block) folds
    /// into a user turn, exactly as the native builder does. Text-only by design:
    /// Krill's native qwen3_5 runtime serves text, and the checkpoint keeps its
    /// full vision-capable template untouched for other consumers (mlx_vlm).
    public func chatmlPrompt(messages: [[String: String]]) -> String {
        var out = ""
        for msg in messages {
            let rawRole = msg["role"] ?? "user"
            let content = msg["content"] ?? ""
            let role: String
            switch rawRole {
            case "system", "user", "assistant", "tool": role = rawRole
            default: role = "user"
            }
            out += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        }
        out += "<|im_start|>assistant\n"
        return out
    }

    public func applyChatTemplate(messages: [[String: String]]) -> String {
        // First preference: the tokenizer's embedded `chat_template`
        // (whatever shipped in `tokenizer_config.json`).
        if let tokenIds = try? tokenizer.applyChatTemplate(messages: messages) {
            return tokenizer.decode(tokens: tokenIds)
        }

        // Newer HF convention (mid-2025+, Qwen 3 Coder / Instruct-2507,
        // Gemma 4): the chat template ships as a separate
        // `chat_template.jinja` file that swift-transformers does NOT
        // read. Without this branch the next two fallbacks would kick
        // in (Gemma 4 manual path; Llama 3 manual path), which apply
        // the wrong special-token convention - on a Qwen checkpoint
        // that produces a prompt the model has never seen, and the
        // model degenerates into FIM tokens / repetition.
        if let template = externalChatTemplate,
           let tokenIds = try? tokenizer.applyChatTemplate(
               messages: messages, chatTemplate: template) {
            return tokenizer.decode(tokens: tokenIds)
        }

        // The model's own template exists but the Swift Jinja port could not
        // render it (both attempts above failed). If it is ChatML-based, emit a
        // manual ChatML prompt rather than falling through to the Gemma-4 /
        // Llama-3 manual formats below: those terminate turns with a marker that
        // is not this model's EOS, so generation never stops. (Ornith-9B /
        // qwen3_5 hits this — its template uses `{% set x = render_content(...) %}`
        // macro-capture the port can't evaluate; the old Llama-3 fallback emitted
        // `<|eot_id|>`, which the model echoed as text and looped on.)
        if usesChatMLTemplate {
            return chatmlPrompt(messages: messages)
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

    /// Encode messages as Qwen 2.5-VL token ids, manually rendering
    /// the ChatML sequence so an image span can be placed.
    ///
    /// Qwen 2.5-VL's stock chat template only emits a single
    /// `<|image_pad|>` and relies on a separate image processor to
    /// expand it; Krill's native runtime instead injects the full
    /// `<|vision_start|>` + `imagePadCount` × `<|image_pad|>` +
    /// `<|vision_end|>` run directly (the processor already produced
    /// `imagePadCount = gridH * gridW` merged vision tokens). The
    /// image attaches to the FIRST user turn. `imagePadCount == 0`
    /// renders a plain text-only ChatML prompt.
    ///
    /// Format (per turn): `<|im_start|>{role}\n{content}<|im_end|>\n`,
    /// then a trailing `<|im_start|>assistant\n` to cue generation.
    /// `<|im_start|>` / `<|im_end|>` are 151644 / 151645 in every
    /// Qwen 2 / 2.5 tokenizer; the vision ids come from the model
    /// config and are passed in.
    public func formatQwen25VLTokenIds(
        messages: [[String: String]],
        imagePadCount: Int,
        imageTokenId: Int,
        visionStartTokenId: Int,
        visionEndTokenId: Int
    ) -> [Int] {
        let imStart = 151_644  // <|im_start|>
        let imEnd = 151_645    // <|im_end|>
        let newline = tokenizer.encode(text: "\n")
        let firstUserIndex = messages.firstIndex { $0["role"] == "user" }

        var tokens: [Int] = []
        for (i, msg) in messages.enumerated() {
            let rawRole = msg["role"] ?? "user"
            let content = msg["content"] ?? ""
            // Qwen ChatML roles: system / user / assistant / tool.
            // Anything else (e.g. a "tools" definitions block) folds
            // into a user turn.
            let role: String
            switch rawRole {
            case "system", "user", "assistant", "tool": role = rawRole
            default: role = "user"
            }
            tokens.append(imStart)
            tokens += tokenizer.encode(text: role)
            tokens += newline
            if i == firstUserIndex && imagePadCount > 0 {
                tokens.append(visionStartTokenId)
                tokens += Array(repeating: imageTokenId, count: imagePadCount)
                tokens.append(visionEndTokenId)
            }
            tokens += tokenizer.encode(text: content)
            tokens.append(imEnd)
            tokens += newline
        }
        tokens.append(imStart)
        tokens += tokenizer.encode(text: "assistant")
        tokens += newline
        return tokens
    }

    /// Encode messages as Qwen3.5-VL (Ornith) token ids with an image span.
    /// Mirrors `formatQwen25VLTokenIds` but resolves `<|im_start|>` / `<|im_end|>`
    /// from the tokenizer (Ornith's ChatML ids are 248045 / 248046, NOT the
    /// Qwen2.5 151644 / 151645) and takes the vision ids from the model config.
    /// The image attaches to the FIRST user turn as
    /// `<|vision_start|>` + `imagePadCount` × `<|image_pad|>` + `<|vision_end|>`.
    /// `imagePadCount == 0` renders a plain text-only ChatML prompt (identical
    /// to `chatmlPrompt`, just as token ids).
    public func formatQwen35VLTokenIds(
        messages: [[String: String]],
        imagePadCount: Int,
        imageTokenId: Int,
        visionStartTokenId: Int,
        visionEndTokenId: Int
    ) -> [Int] {
        // Single special-token ids (encode() returns exactly one id per marker
        // for a ChatML tokenizer; verified 248045 / 248046 for Ornith).
        let imStart = tokenizer.encode(text: "<|im_start|>")
        let imEnd = tokenizer.encode(text: "<|im_end|>")
        let newline = tokenizer.encode(text: "\n")
        let firstUserIndex = messages.firstIndex { $0["role"] == "user" }

        var tokens: [Int] = []
        for (i, msg) in messages.enumerated() {
            let rawRole = msg["role"] ?? "user"
            let content = msg["content"] ?? ""
            let role: String
            switch rawRole {
            case "system", "user", "assistant", "tool": role = rawRole
            default: role = "user"
            }
            tokens += imStart
            tokens += tokenizer.encode(text: role)
            tokens += newline
            if i == firstUserIndex && imagePadCount > 0 {
                tokens.append(visionStartTokenId)
                tokens += Array(repeating: imageTokenId, count: imagePadCount)
                tokens.append(visionEndTokenId)
            }
            tokens += tokenizer.encode(text: content)
            tokens += imEnd
            tokens += newline
        }
        tokens += imStart
        tokens += tokenizer.encode(text: "assistant")
        tokens += newline
        return tokens
    }

    /// Encode messages as LLaVA-1.5 token ids, manually rendering the vicuna-v1
    /// conversation the model was fine-tuned on and placing the image-token run
    /// directly (mirrors `formatQwen25VLTokenIds`).
    ///
    /// LLaVA-1.5's processor inserts a single `<image>` placeholder and relies
    /// on a separate image processor to expand it to one token per CLIP patch;
    /// the native runtime instead injects the full `imagePadCount` × image-token
    /// run inline (the engine already produced `imagePadCount =
    /// (image_size / patch_size)^2` vision features), so
    /// `LlavaForCausalLM`'s forward finds exactly that many image positions to
    /// splice the projected CLIP features into. The image attaches to the FIRST
    /// user turn. `imagePadCount == 0` renders a plain text-only prompt.
    ///
    /// Vicuna-v1 format: a system preamble, then ` USER: {content}` /
    /// ` ASSISTANT: {content}` turns joined by spaces (assistant answers are
    /// terminated by `</s>`), ending with a trailing ` ASSISTANT:` cue. A
    /// leading `system` message overrides the default preamble. `imageTokenId`
    /// is the checkpoint's `image_token_index` (32000 for llava-1.5).
    public func formatLlavaTokenIds(
        messages: [[String: String]],
        imageTokenId: Int,
        imagePadCount: Int
    ) -> [Int] {
        let defaultSystem =
            "A chat between a curious human and an artificial intelligence assistant. "
            + "The assistant gives helpful, detailed, and polite answers to the human's questions."
        let bos = bosTokenId
        // The underlying tokenizer may prepend BOS to a segment; we own BOS
        // placement (one leading BOS), so strip it from each piece.
        func enc(_ s: String) -> [Int] {
            var t = tokenizer.encode(text: s)
            if t.first == bos { t.removeFirst() }
            return t
        }

        var systemText = defaultSystem
        var turns = messages
        if let first = turns.first, (first["role"] ?? "") == "system" {
            let s = first["content"] ?? ""
            if !s.isEmpty { systemText = s }
            turns.removeFirst()
        }
        // An image request with no user turn (e.g. system-only messages) must
        // still place its token run somewhere, or the engine forwards the
        // pixels with zero image positions and `LlavaForCausalLM`'s
        // `imagePositions.count == features` precondition aborts the process on
        // client input. Synthesize an empty user turn to carry the image.
        if imagePadCount > 0 && !turns.contains(where: { ($0["role"] ?? "") == "user" }) {
            turns.append(["role": "user", "content": ""])
        }
        let firstUserIndex = turns.firstIndex { ($0["role"] ?? "") == "user" }

        var tokens: [Int] = [bos]
        tokens += enc(systemText)
        for (i, msg) in turns.enumerated() {
            let role = msg["role"] ?? "user"
            let content = msg["content"] ?? ""
            let label = (role == "assistant") ? "ASSISTANT" : "USER"
            tokens += enc(" \(label): ")
            if i == firstUserIndex && imagePadCount > 0 {
                tokens += Array(repeating: imageTokenId, count: imagePadCount)
                tokens += enc("\n")
            }
            tokens += enc(content)
            // Vicuna terminates a completed assistant turn with </s>.
            if role == "assistant" { tokens.append(eosTokenId) }
        }
        tokens += enc(" ASSISTANT:")
        return tokens
    }

    /// Build the Llama-3.2-Vision (mllama) prompt token ids. Uses the Llama-3
    /// chat structure (`<|start_header_id|>` / `<|end_header_id|>` / `<|eot_id|>`)
    /// and places `imageCount` `<|image|>` tokens at the START of the first user
    /// turn (one per supplied image) -- the cross-attention markers that the
    /// driver's `cross_attention_mask` keys off. Unlike LLaVA's `<image>` run,
    /// these are not soft-token placeholders: the image enters via cross-attention,
    /// so one marker per image is enough. `<|image|>` and the header tokens are
    /// special tokens in the mllama vocab, so encoding their literals yields the
    /// single ids; we own the one leading BOS (strip any the tokenizer re-adds).
    public func formatLlamaVisionTokenIds(
        messages: [[String: String]], imageTokenId: Int, imageCount: Int
    ) -> [Int] {
        let bos = bosTokenId
        func enc(_ s: String) -> [Int] {
            var t = tokenizer.encode(text: s)
            if t.first == bos { t.removeFirst() }
            return t
        }

        var turns = messages
        var systemText: String? = nil
        if let first = turns.first, (first["role"] ?? "") == "system" {
            let s = first["content"] ?? ""
            if !s.isEmpty { systemText = s }
            turns.removeFirst()
        }
        // An image request with no user turn must still place its markers, or
        // the prompt carries no image positions and the cross mask is empty.
        if imageCount > 0 && !turns.contains(where: { ($0["role"] ?? "") == "user" }) {
            turns.append(["role": "user", "content": ""])
        }
        let firstUserIndex = turns.firstIndex { ($0["role"] ?? "") == "user" }

        var tokens: [Int] = [bos]
        if let systemText {
            tokens += enc("<|start_header_id|>system<|end_header_id|>\n\n\(systemText)<|eot_id|>")
        }
        for (i, msg) in turns.enumerated() {
            let role = msg["role"] ?? "user"
            let content = msg["content"] ?? ""
            tokens += enc("<|start_header_id|>\(role)<|end_header_id|>\n\n")
            if i == firstUserIndex && imageCount > 0 {
                tokens += Array(repeating: imageTokenId, count: imageCount)
            }
            tokens += enc("\(content)<|eot_id|>")
        }
        tokens += enc("<|start_header_id|>assistant<|end_header_id|>\n\n")
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
