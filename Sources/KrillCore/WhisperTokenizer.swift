import Foundation

// MARK: - Whisper tokenizer (decode path)

/// GPT-2 byte-level BPE detokenizer for Whisper plus the fixed English-only
/// (`.en`) special-token ids. Transcription only needs the decode direction
/// (token ids -> text) and a handful of control ids, so this is a focused,
/// dependency-free reimplementation; the `vocab.json` is bundled by
/// `tools/convert_whisper.py`.
public final class WhisperTokenizer {
    /// `id -> byte-level token string` for the text vocabulary.
    private let idToToken: [String?]
    /// Inverse of the GPT-2 `bytes_to_unicode` map: unicode scalar -> raw byte.
    private let byteDecoder: [Character: UInt8]
    /// The control-token ids for this model (English vs multilingual layout).
    public let specials: Specials

    /// Whisper control-token ids. The English-only (`.en`) checkpoints and the
    /// multilingual checkpoints place these at different offsets (multilingual
    /// inserts 99 language tokens after `<|startoftranscript|>`), so the layout
    /// is selected at load time.
    public struct Specials: Sendable {
        public let endOfText: Int          // <|endoftext|>, also the stop token
        public let startOfTranscript: Int  // <|startoftranscript|>
        public let transcribe: Int         // <|transcribe|>
        public let noTimestamps: Int       // <|notimestamps|>
        public let timestampBegin: Int     // <|0.00|>
        /// First language token (`<|en|>`); nil for English-only models.
        public let languageBase: Int?
        /// Number of language tokens (`languageBase ..< languageBase+count`).
        public let languageCount: Int

        /// whisper-*.en layout.
        public static let english = Specials(
            endOfText: 50256, startOfTranscript: 50257, transcribe: 50358,
            noTimestamps: 50362, timestampBegin: 50363,
            languageBase: nil, languageCount: 0)

        /// Multilingual whisper-* layout (tiny/base/small/medium/large v1-v2).
        public static let multilingual = Specials(
            endOfText: 50257, startOfTranscript: 50258, transcribe: 50359,
            noTimestamps: 50363, timestampBegin: 50364,
            languageBase: 50259, languageCount: 99)
    }

    public enum TokenizerError: Error, CustomStringConvertible {
        case vocabNotFound(String)
        public var description: String {
            switch self {
            case .vocabNotFound(let p): return "Whisper vocab.json not found at \(p)"
            }
        }
    }

    /// Load from a converted model dir's `vocab.json` (`{token: id}`).
    /// `multilingual` selects the control-token layout.
    public init(vocabURL: URL, multilingual: Bool) throws {
        guard let data = try? Data(contentsOf: vocabURL) else {
            throw TokenizerError.vocabNotFound(vocabURL.path)
        }
        let map = try JSONDecoder().decode([String: Int].self, from: data)
        var maxId = 0
        for id in map.values where id > maxId { maxId = id }
        var table = [String?](repeating: nil, count: maxId + 1)
        for (tok, id) in map { table[id] = tok }
        idToToken = table
        byteDecoder = Self.makeByteDecoder()
        specials = multilingual ? .multilingual : .english
    }

    /// The decoder prompt for no-timestamp transcription. `language` (a
    /// language token id, multilingual only) is inserted after the start token.
    public func promptTokens(language: Int? = nil) -> [Int] {
        var p = [specials.startOfTranscript]
        if let lang = language { p.append(lang) }
        p.append(contentsOf: [specials.transcribe, specials.noTimestamps])
        return p
    }

    /// GPT-2 `bytes_to_unicode`, inverted to scalar -> byte.
    private static func makeByteDecoder() -> [Character: UInt8] {
        var bs = [Int]()
        bs.append(contentsOf: Int(Character("!").asciiValue!) ... Int(Character("~").asciiValue!))
        bs.append(contentsOf: 0xA1 ... 0xAC)
        bs.append(contentsOf: 0xAE ... 0xFF)
        var cs = bs
        var n = 0
        for b in 0 ..< 256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var dec = [Character: UInt8]()
        for i in 0 ..< bs.count {
            dec[Character(UnicodeScalar(cs[i])!)] = UInt8(bs[i])
        }
        return dec
    }

    /// Decode generated token ids to text. Stops at `<|endoftext|>` and drops
    /// any special/timestamp tokens; the surviving byte-level strings are
    /// mapped back to raw bytes and UTF-8 decoded.
    public func decode(_ ids: [Int]) -> String {
        var bytes = [UInt8]()
        for id in ids {
            if id == specials.endOfText { break }
            guard id >= 0, id < idToToken.count, id < specials.endOfText,
                  let tok = idToToken[id] else { continue }
            for ch in tok {
                if let b = byteDecoder[ch] { bytes.append(b) }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
