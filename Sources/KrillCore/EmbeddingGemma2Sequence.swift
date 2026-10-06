import Foundation
import MLX
import KrillTokenizer

// MARK: - EmbeddingGemma 2: multimodal sequence builder
//
// How EmbeddingGemma 2 fuses modalities (verified against transformers
// 5.19 `processing_embedding_gemma2.py` / `modeling_embedding_gemma2.py` and
// against sentence-transformers' own `input_ids`, see the `eg2_mm` fixture):
//
//   * one flat token sequence `<bos> ... <eos>` per input; media sit inside it
//     as   <boi> <image>xN <eoi>   (image)
//          (<boi> <video>xN <eoi>) x frames   (video; one block per frame)
//          <boa> <audio>xN <eoa>   (audio)
//   * the tower output, projected by `embed_vision` / `embed_audio` into the
//     text space, REPLACES the embeddings at the `<image>/<video>/<audio>`
//     positions (`masked_scatter`). Soft tokens are NOT multiplied by
//     sqrt(hidden); real text tokens (markers included) are.
//   * PLE is computed after the merge, then the one text backbone runs, then
//     mean pool over every position -> 512 -> 768 -> L2.
//   * text parts that touch each other are ONE string for the tokenizer (the
//     reference chat template concatenates them), so "query: " + "red" is
//     tokenised jointly; only media break the string.
//
// Adding a modality means: a tower + preprocessor that produce
// `[softTokens, hidden]` features, a `EG2MediaBlock` describing its layout, and
// a request part type. Nothing in this file needs to change for that (the
// layout of audio and video blocks is already here).

/// Marker / placeholder token ids (`config.json`, top level).
public struct EG2ModalityTokens: Equatable, Sendable {
    public let boi: Int
    public let eoi: Int
    public let image: Int
    public let boa: Int
    public let eoa: Int
    public let audio: Int
    public let video: Int

    /// The ids of `google/embeddinggemma-2` (used if the config omits them).
    public static let checkpointDefaults = EG2ModalityTokens(
        boi: 255999, eoi: 258882, image: 258880,
        boa: 256000, eoa: 258883, audio: 258881, video: 258884)

    public init(boi: Int, eoi: Int, image: Int, boa: Int, eoa: Int, audio: Int, video: Int) {
        self.boi = boi; self.eoi = eoi; self.image = image
        self.boa = boa; self.eoa = eoa; self.audio = audio; self.video = video
    }

    /// Literal strings of the markers / placeholders; user text containing one
    /// of these would be tokenised to a real special id and corrupt the
    /// soft-token count, so the builder rejects it.
    public static let reservedLiterals = [
        "<|image>", "<image|>", "<|image|>", "<|audio>", "<audio|>", "<|audio|>", "<|video|>",
    ]
}

/// A media kind that is fused as soft tokens.
public enum EG2Modality: String, CaseIterable, Sendable {
    case image, audio, video

    /// (open marker, soft-token placeholder, close marker) ids.
    func ids(_ t: EG2ModalityTokens) -> (open: Int, soft: Int, close: Int) {
        switch self {
        case .image: return (t.boi, t.image, t.eoi)
        case .video: return (t.boi, t.video, t.eoi)
        case .audio: return (t.boa, t.audio, t.eoa)
        }
    }
}

/// Layout of one media item inside the sequence: `blocks` blocks (1 for an
/// image / audio clip, one per frame for a video), each with
/// `softTokensPerBlock` soft tokens between its markers.
public struct EG2MediaBlock: Equatable, Sendable {
    public let modality: EG2Modality
    public let softTokensPerBlock: Int
    public let blocks: Int
    public init(_ modality: EG2Modality, softTokensPerBlock: Int, blocks: Int = 1) {
        self.modality = modality
        self.softTokensPerBlock = softTokensPerBlock
        self.blocks = blocks
    }
    public var softTokens: Int { softTokensPerBlock * blocks }
}

/// One element of an input, in order.
public enum EG2Segment: Equatable, Sendable {
    case text(String)
    case media(EG2MediaBlock)
}

/// A soft-token run inside a built sequence.
public struct EG2SoftSpan: Equatable, Sendable {
    public let modality: EG2Modality
    public let start: Int
    public let count: Int
    public init(modality: EG2Modality, start: Int, count: Int) {
        self.modality = modality; self.start = start; self.count = count
    }
}

/// Result of `EG2SequenceBuilder.build`.
public struct EG2Sequence: Equatable, Sendable {
    /// Token ids incl. `<bos>` / `<eos>`; soft positions hold the placeholder id.
    public let ids: [Int32]
    /// Soft-token runs in order (one per media block).
    public let spans: [EG2SoftSpan]
    public var count: Int { ids.count }
    public init(ids: [Int32], spans: [EG2SoftSpan]) { self.ids = ids; self.spans = spans }
    /// Total soft tokens of `modality` (what the tower must supply).
    public func softTokenCount(_ m: EG2Modality) -> Int {
        spans.filter { $0.modality == m }.reduce(0) { $0 + $1.count }
    }
}

public enum EG2SequenceError: Error, CustomStringConvertible, Equatable {
    case reservedLiteral(String)
    case tooLong(tokens: Int, limit: Int)
    case softTokenMismatch(modality: String, expected: Int, got: Int)
    case emptyInput

    public var description: String {
        switch self {
        case .reservedLiteral(let l):
            return "text contains the reserved placeholder '\(l)'; send media as separate content parts"
        case .tooLong(let n, let limit):
            return "input is \(n) tokens (media included) which exceeds the \(limit)-token context"
        case .softTokenMismatch(let m, let e, let g):
            return "\(m) features have \(g) soft tokens but the prompt reserved \(e)"
        case .emptyInput:
            return "input has no content"
        }
    }
}

public struct EG2SequenceBuilder: Sendable {
    public let tokenizer: CodePointBPETokenizer
    public let tokens: EG2ModalityTokens
    public let maxTokens: Int

    public init(tokenizer: CodePointBPETokenizer, tokens: EG2ModalityTokens,
                maxTokens: Int = EmbeddingGemma2Config.maxContext) {
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.maxTokens = maxTokens
    }

    /// Build the id sequence. `prefix` (the task prompt, if any) is concatenated
    /// in front of the first text and only applied when the input has text
    /// (the model card: media are passed without a prefix; the reference
    /// prepends it as a system message, i.e. as a string prefix).
    ///
    /// Never truncates: cutting a media block would desynchronise placeholders
    /// and features, so an over-long input throws `.tooLong`.
    public func build(_ segments: [EG2Segment], prefix: String = "") throws -> EG2Sequence {
        guard !segments.isEmpty else { throw EG2SequenceError.emptyInput }
        // Merge touching text (the reference concatenates before tokenising).
        var merged: [EG2Segment] = []
        var hasText = false
        for seg in segments {
            switch seg {
            case .text(let t):
                for lit in EG2ModalityTokens.reservedLiterals where t.contains(lit) {
                    throw EG2SequenceError.reservedLiteral(lit)
                }
                hasText = true
                if case .text(let prev)? = merged.last {
                    merged[merged.count - 1] = .text(prev + t)
                } else {
                    merged.append(.text(t))
                }
            case .media:
                merged.append(seg)
            }
        }
        if hasText, !prefix.isEmpty {
            if case .text(let first)? = merged.first {
                merged[0] = .text(prefix + first)
            } else {
                merged.insert(.text(prefix), at: 0)
            }
        }

        var ids: [Int32] = []
        var spans: [EG2SoftSpan] = []
        if let b = tokenizer.bosId { ids.append(Int32(b)) }
        for seg in merged {
            switch seg {
            case .text(let t):
                ids += tokenizer.encode(t, addSpecialTokens: false).map { Int32($0) }
            case .media(let m):
                let id = m.modality.ids(tokens)
                for _ in 0 ..< m.blocks {
                    ids.append(Int32(id.open))
                    spans.append(EG2SoftSpan(modality: m.modality, start: ids.count, count: m.softTokensPerBlock))
                    ids += [Int32](repeating: Int32(id.soft), count: m.softTokensPerBlock)
                    ids.append(Int32(id.close))
                }
            }
        }
        if let e = tokenizer.eosId { ids.append(Int32(e)) }
        guard ids.count <= maxTokens else {
            throw EG2SequenceError.tooLong(tokens: ids.count, limit: maxTokens)
        }
        return EG2Sequence(ids: ids, spans: spans)
    }
}

// MARK: - Scatter

extension EmbeddingGemma2Model {
    /// Embed `sequence` for the backbone: text tokens (markers included) via the
    /// scaled table, soft positions replaced by `features[modality]` (one
    /// `[n, hidden]` array per media block, in order, already projected into the
    /// text space and cast to the compute dtype). Returns `[1, T, hidden]`.
    public func mergedEmbeddings(_ sequence: EG2Sequence,
                                 features: [EG2Modality: [MLXArray]]) throws -> MLXArray {
        var cursor = [EG2Modality: Int]()
        let T = sequence.count
        // Soft positions hold the placeholder id; its row is overwritten, but
        // the id may be out of vocab for odd configs, so look up PAD there.
        var lookup = sequence.ids
        for s in sequence.spans { for i in s.start ..< (s.start + s.count) { lookup[i] = 0 } }
        let text = embedText(MLXArray(lookup).reshaped(1, T))  // [1, T, H]
        if sequence.spans.isEmpty { return text }

        var pieces: [MLXArray] = []
        var pos = 0
        for span in sequence.spans {
            if span.start > pos { pieces.append(text[0..., pos ..< span.start, 0...]) }
            let k = cursor[span.modality, default: 0]
            guard let list = features[span.modality], k < list.count else {
                throw EG2SequenceError.softTokenMismatch(
                    modality: span.modality.rawValue, expected: span.count, got: 0)
            }
            let f = list[k]
            guard f.ndim == 2, f.dim(0) == span.count, f.dim(1) == config.hiddenSize else {
                throw EG2SequenceError.softTokenMismatch(
                    modality: span.modality.rawValue, expected: span.count, got: f.dim(0))
            }
            pieces.append(f.asType(computeDtype).reshaped(1, span.count, config.hiddenSize))
            cursor[span.modality] = k + 1
            pos = span.start + span.count
        }
        for (m, list) in features where (cursor[m] ?? 0) != list.count {
            throw EG2SequenceError.softTokenMismatch(
                modality: m.rawValue, expected: cursor[m] ?? 0, got: list.count)
        }
        if pos < T { pieces.append(text[0..., pos ..< T, 0...]) }
        return concatenated(pieces, axis: 1)
    }
}
