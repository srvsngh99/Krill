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

    // Special token ids (whisper-small.en / English-only layout).
    public static let endOfText = 50256          // <|endoftext|> (also stop)
    public static let startOfTranscript = 50257  // <|startoftranscript|>
    public static let transcribe = 50358         // <|transcribe|>
    public static let noTimestamps = 50362       // <|notimestamps|>
    public static let timestampBegin = 50363     // <|0.00|>

    /// The decoder prompt for English, no-timestamp transcription.
    public static let promptTokens = [startOfTranscript, transcribe, noTimestamps]

    public enum TokenizerError: Error, CustomStringConvertible {
        case vocabNotFound(String)
        public var description: String {
            switch self {
            case .vocabNotFound(let p): return "Whisper vocab.json not found at \(p)"
            }
        }
    }

    /// Load from a converted model dir's `vocab.json` (`{token: id}`).
    public init(vocabURL: URL) throws {
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
            if id == Self.endOfText { break }
            guard id >= 0, id < idToToken.count, id < Self.endOfText,
                  let tok = idToToken[id] else { continue }
            for ch in tok {
                if let b = byteDecoder[ch] { bytes.append(b) }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
