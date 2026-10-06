import Foundation

// MARK: - Embedding inputs: text and media content parts
//
// `/v1/embeddings` and `/api/embed` take `input` as a string, an array of
// strings (unchanged, for every model) or, for multimodal embedders
// (EmbeddingGemma 2), an array whose items are strings or CONTENT-PART items:
//
//   {"content": [ {"type":"text","text":"a red bicycle"},
//                 {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}} ]}
//
// Part order is token order; one item is one embedding (image + caption = one
// joint vector). No network fetches: media must be `data:` URLs or base64.
//
// Extending to a new modality (see docs/EMBEDDINGGEMMA2.md "Extension guide"):
//   1. add a case to `EmbeddingPart`;
//   2. add a `case` in `parsePart`;
//   3. handle the case in `EmbeddingEngine.embedMediaItem`.

/// One piece of an embedding input.
public enum EmbeddingPart: Equatable, Sendable {
    case text(String)
    /// Raw encoded image bytes (PNG/JPEG/...), already base64-decoded.
    case image(Data)
    /// Raw encoded audio bytes (wav/mp3/m4a/flac/...), base64-decoded, plus the
    /// client's format hint (`"wav"`, `"mp3"`, ... or "" when unknown).
    case audio(Data, format: String)
    /// Raw encoded video bytes (mp4 / mov / m4v), base64-decoded, plus the format hint.
    case video(Data, format: String)
}

/// One embedding input: an ordered list of parts. A plain string is a single
/// `.text` part.
public struct EmbeddingInput: Equatable, Sendable {
    public var parts: [EmbeddingPart]
    public init(parts: [EmbeddingPart]) { self.parts = parts }
    public init(text: String) { self.parts = [.text(text)] }

    /// True when every part is text (served by the unchanged text path).
    public var isTextOnly: Bool {
        parts.allSatisfy { if case .text = $0 { return true } else { return false } }
    }
    /// All text parts concatenated (only meaningful when `isTextOnly`).
    public var joinedText: String {
        parts.reduce(into: "") { if case .text(let t) = $1 { $0 += t } }
    }
    public var hasMedia: Bool { !isTextOnly }
}

public struct EmbeddingInputError: Error, Equatable, CustomStringConvertible {
    public let message: String
    public var description: String { message }
    public init(_ m: String) { message = m }
}

public enum EmbeddingInputParser {
    static let shapeHelp = "'input' must be a string, an array of strings, or an array of strings and "
        + "{\"content\":[{\"type\":\"text\"|\"image_url\"|\"input_image\"|\"input_audio\"|\"video_url\",...}]} items"

    /// Parse the `input` field of an embeddings request.
    public static func parse(_ raw: Any?) throws -> [EmbeddingInput] {
        if let s = raw as? String { return [EmbeddingInput(text: s)] }
        guard let arr = raw as? [Any], !arr.isEmpty else { throw EmbeddingInputError(shapeHelp) }
        var out: [EmbeddingInput] = []
        out.reserveCapacity(arr.count)
        for (i, el) in arr.enumerated() {
            if let s = el as? String {
                out.append(EmbeddingInput(text: s))
            } else if let d = el as? [String: Any] {
                out.append(try parseItem(d, index: i))
            } else if let parts = el as? [Any], parts.allSatisfy({ $0 is [String: Any] }), !parts.isEmpty {
                // A bare array of part objects is accepted as one item.
                out.append(try parseParts(parts, index: i))
            } else {
                throw EmbeddingInputError("input[\(i)]: \(shapeHelp)")
            }
        }
        return out
    }

    static func parseItem(_ d: [String: Any], index i: Int) throws -> EmbeddingInput {
        if let content = d["content"] {
            if let s = content as? String { return EmbeddingInput(text: s) }
            guard let parts = content as? [Any], !parts.isEmpty else {
                throw EmbeddingInputError("input[\(i)].content must be a non-empty array of parts")
            }
            return try parseParts(parts, index: i)
        }
        if d["type"] != nil { return try parseParts([d], index: i) }  // a single bare part
        throw EmbeddingInputError("input[\(i)]: object items need a 'content' array of parts")
    }

    static func parseParts(_ parts: [Any], index i: Int) throws -> EmbeddingInput {
        var out: [EmbeddingPart] = []
        for (j, p) in parts.enumerated() {
            guard let d = p as? [String: Any] else {
                throw EmbeddingInputError("input[\(i)].content[\(j)] must be an object")
            }
            out.append(try parsePart(d, at: "input[\(i)].content[\(j)]"))
        }
        return EmbeddingInput(parts: out)
    }

    static func parsePart(_ d: [String: Any], at path: String) throws -> EmbeddingPart {
        guard let type = d["type"] as? String else {
            throw EmbeddingInputError("\(path): missing 'type'")
        }
        switch type {
        case "text":
            guard let t = d["text"] as? String else {
                throw EmbeddingInputError("\(path): 'text' part needs a string 'text'")
            }
            return .text(t)
        case "image_url", "input_image":
            // `image_url` may be {"url": ...} or a bare string; `input_image` may
            // also carry raw base64 in `data`.
            var ref: String? = nil
            if let o = d["image_url"] as? [String: Any] { ref = o["url"] as? String }
            else if let s = d["image_url"] as? String { ref = s }
            else if type == "input_image", let s = d["data"] as? String { ref = s }
            guard let r = ref, !r.isEmpty else {
                throw EmbeddingInputError(
                    "\(path): '\(type)' part needs \(type == "image_url" ? "image_url.url" : "image_url or data") (a data: URL or base64)")
            }
            return .image(try decodeMedia(r, kind: "image", at: path))
        case "input_audio", "audio_url":
            // OpenAI shape: {"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}.
            // Also accepted: `data` may be a data: URL, `input_audio` may be a bare
            // base64 string, and `audio_url` carries {"url": "data:audio/...;base64,..."}.
            var ref: String? = nil
            var format = ""
            if let o = d["input_audio"] as? [String: Any] {
                ref = (o["data"] as? String) ?? (o["url"] as? String)
                format = (o["format"] as? String) ?? ""
            } else if let s = d["input_audio"] as? String { ref = s }
            else if let o = d["audio_url"] as? [String: Any] { ref = o["url"] as? String }
            else if let s = d["audio_url"] as? String { ref = s }
            else if let s = d["data"] as? String { ref = s }
            if format.isEmpty, let f = d["format"] as? String { format = f }
            guard let r = ref, !r.isEmpty else {
                throw EmbeddingInputError(
                    "\(path): '\(type)' part needs input_audio.data (base64 or a data: URL) and input_audio.format")
            }
            if format.isEmpty, r.lowercased().hasPrefix("data:audio/"),
               let semi = r.firstIndex(of: ";") {
                format = String(r[r.index(r.startIndex, offsetBy: 11) ..< semi])
            }
            return .audio(try decodeMedia(r, kind: "audio", at: path), format: format)
        case "video_url", "input_video", "video":
            // {"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}} (the
            // OpenAI-compatible shape; `video_url` may also be a bare string), or
            // {"type":"input_video","input_video":{"data":"<base64>","format":"mp4"}} /
            // {"type":"input_video","data":"<base64>"}.
            var ref: String? = nil
            var format = ""
            if let o = d["video_url"] as? [String: Any] { ref = o["url"] as? String }
            else if let s = d["video_url"] as? String { ref = s }
            else if let o = d["input_video"] as? [String: Any] {
                ref = (o["data"] as? String) ?? (o["url"] as? String)
                format = (o["format"] as? String) ?? ""
            } else if let s = d["input_video"] as? String { ref = s }
            else if let s = d["video"] as? String { ref = s }
            else if let s = d["data"] as? String { ref = s }
            if format.isEmpty, let f = d["format"] as? String { format = f }
            guard let r = ref, !r.isEmpty else {
                throw EmbeddingInputError(
                    "\(path): '\(type)' part needs video_url.url (a data: URL) or input_video.data (base64)")
            }
            if format.isEmpty, r.lowercased().hasPrefix("data:video/"), let semi = r.firstIndex(of: ";") {
                format = String(r[r.index(r.startIndex, offsetBy: 11) ..< semi])
            }
            return .video(try decodeMedia(r, kind: "video", at: path), format: format)
        default:
            throw EmbeddingInputError(
                "\(path): unknown part type '\(type)'; supported: text, image_url, input_image, input_audio, video_url")
        }
    }

    /// `data:<mime>;base64,<payload>` or bare base64. Remote / file references
    /// are refused: Krill does not fetch anything on behalf of a request.
    static func decodeMedia(_ ref: String, kind: String, at path: String) throws -> Data {
        let lower = ref.prefix(12).lowercased()
        if lower.hasPrefix("http:") || lower.hasPrefix("https:") || lower.hasPrefix("file:")
            || lower.hasPrefix("ftp:") || ref.hasPrefix("/") {
            throw EmbeddingInputError(
                "\(path): remote and file URLs are not fetched; send the \(kind) as a data: URL or base64")
        }
        var payload = Substring(ref)
        if ref.lowercased().hasPrefix("data:") {
            guard let comma = ref.firstIndex(of: ",") else {
                throw EmbeddingInputError("\(path): malformed data: URL (no comma)")
            }
            let header = ref[ref.startIndex ..< comma].lowercased()
            guard header.hasSuffix(";base64") else {
                throw EmbeddingInputError("\(path): data: URL must be base64-encoded")
            }
            let mime = header.dropFirst(5).split(separator: ";").first.map(String.init) ?? ""
            if !mime.isEmpty, !mime.hasPrefix(kind + "/") {
                throw EmbeddingInputError("\(path): data: URL mime type must be \(kind)/*, got \(mime)")
            }
            payload = ref[ref.index(after: comma)...]
        }
        guard let data = Data(base64Encoded: String(payload), options: .ignoreUnknownCharacters),
              !data.isEmpty else {
            throw EmbeddingInputError("\(path): invalid base64 \(kind) data")
        }
        return data
    }
}
