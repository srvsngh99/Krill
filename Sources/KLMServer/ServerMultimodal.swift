import Foundation

/// Decoded multimodal payload extracted from a server request.
///
/// `imagePath` and `audioPath` point to temporary files on disk that the
/// caller must delete (use ``cleanup()``) once inference completes.
internal struct DecodedMedia: Sendable {
    /// All decoded image temp-file paths, in request order. Single-image
    /// runtimes use the first; mllama (multi-image) uses them all.
    let imagePaths: [String]
    let audioPath: String?

    /// The first decoded image, for single-image runtimes.
    var imagePath: String? { imagePaths.first }

    /// Load the decoded image temp files into memory: the FIRST image (for
    /// single-image runtimes) and the FULL ordered list (for multi-image
    /// mllama). Every generate handler loads images through this one method, so
    /// a handler can't silently drop all but the first image of a multi-image
    /// request while the prompt still emits one `<|image|>` per image.
    func loadImages() -> (first: Data?, all: [Data]) {
        let all = imagePaths.compactMap { path -> Data? in
            try? Data(contentsOf: URL(fileURLWithPath: path))
        }
        return (all.first, all)
    }

    init(imagePaths: [String] = [], audioPath: String? = nil) {
        self.imagePaths = imagePaths
        self.audioPath = audioPath
    }

    /// Best-effort removal of temp files. Safe to call multiple times.
    func cleanup() {
        let fm = FileManager.default
        for p in imagePaths { try? fm.removeItem(atPath: p) }
        if let p = audioPath { try? fm.removeItem(atPath: p) }
    }
}

/// Errors raised while decoding base64 media payloads.
internal enum MediaDecodeError: Error, Equatable {
    /// The base64 string could not be decoded.
    case invalidBase64(field: String)
    /// More than one image was supplied — current models only accept one per turn.
    case tooManyImages
    /// Image / audio supplied but the loaded model cannot handle it.
    case mediaNotSupported(reason: String)
    /// Decoded payload exceeds the per-item size limit.
    case payloadTooLarge(field: String, bytes: Int, limit: Int)
    /// File could not be written to disk.
    case writeFailed(String)

    var httpStatus: Int {
        switch self {
        case .payloadTooLarge: return 413
        case .invalidBase64, .tooManyImages, .mediaNotSupported: return 400
        case .writeFailed: return 500
        }
    }

    var message: String {
        switch self {
        case .invalidBase64(let f):
            return "Field '\(f)' is not valid base64"
        case .tooManyImages:
            return "Only one image per request is supported"
        case .mediaNotSupported(let reason):
            return reason
        case .payloadTooLarge(let f, let bytes, let limit):
            return "Field '\(f)' is \(bytes) bytes; limit is \(limit) bytes"
        case .writeFailed(let detail):
            return "Failed to write media to temporary storage: \(detail)"
        }
    }
}

internal enum ServerMultimodal {
    /// Maximum decoded size, per item. Aligned with ServerLimits.maxBodySize so
    /// the per-item check and the global HTTP body limit cannot disagree.
    static let maxPayloadBytes = ServerLimits.maxBodySize

    /// Strip a `data:...;base64,` prefix if present, returning the bare base64 body.
    static func stripDataURLPrefix(_ s: String) -> String {
        // Form: `data:[<mediatype>][;base64],<data>`
        guard s.hasPrefix("data:") else { return s }
        guard let commaIdx = s.firstIndex(of: ",") else { return s }
        return String(s[s.index(after: commaIdx)...])
    }

    /// Detect the file extension from magic bytes. Defaults to `.png` if unrecognized.
    static func sniffImageExtension(_ data: Data) -> String {
        if data.count >= 8 {
            let pngMagic: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
            if Array(data.prefix(8)) == pngMagic { return "png" }
        }
        if data.count >= 3 {
            let bytes = Array(data.prefix(3))
            if bytes == [0xFF, 0xD8, 0xFF] { return "jpg" }
        }
        if data.count >= 6 {
            let prefix = Array(data.prefix(6))
            if prefix == [0x47, 0x49, 0x46, 0x38, 0x37, 0x61]
                || prefix == [0x47, 0x49, 0x46, 0x38, 0x39, 0x61] {
                return "gif"
            }
        }
        if data.count >= 12 {
            let bytes = Array(data.prefix(12))
            if bytes[0..<4] == [0x52, 0x49, 0x46, 0x46]
                && bytes[8..<12] == [0x57, 0x45, 0x42, 0x50] {
                return "webp"
            }
        }
        return "png"
    }

    /// Estimated decoded byte count for a base64 string. Cheaply rejects
    /// oversized payloads before allocating a Data buffer to decode them.
    static func estimateDecodedSize(_ base64: String) -> Int {
        let stripped = stripDataURLPrefix(base64)
        let len = stripped.count
        guard len > 0 else { return 0 }
        return (len * 3) / 4
    }

    /// Throw payloadTooLarge if any base64 item's estimated decoded size exceeds
    /// the per-item limit. Used by request validation BEFORE model-loaded checks
    /// so 413 fires regardless of server state.
    static func validatePayloadSizes(_ payload: ServerMediaPayload) throws {
        for (idx, b64) in payload.images.enumerated() {
            if estimateDecodedSize(b64) > maxPayloadBytes {
                throw MediaDecodeError.payloadTooLarge(
                    field: payload.images.count > 1 ? "images[\(idx)]" : "images",
                    bytes: estimateDecodedSize(b64), limit: maxPayloadBytes)
            }
        }
        if let audio = payload.audio, estimateDecodedSize(audio) > maxPayloadBytes {
            throw MediaDecodeError.payloadTooLarge(
                field: "audio", bytes: estimateDecodedSize(audio), limit: maxPayloadBytes)
        }
    }

    /// Decode a single base64 string, validate size, write to a temp file.
    static func decodeAndWrite(
        base64: String,
        field: String,
        preferredExtension: String? = nil,
        sniffImage: Bool = false
    ) throws -> String {
        let stripped = stripDataURLPrefix(base64)
        // Allow padding-relaxed base64 by appending `=` if needed.
        let padded: String = {
            let mod = stripped.count % 4
            return mod == 0 ? stripped : stripped + String(repeating: "=", count: 4 - mod)
        }()
        guard let data = Data(base64Encoded: padded, options: [.ignoreUnknownCharacters]) else {
            throw MediaDecodeError.invalidBase64(field: field)
        }
        if data.count > maxPayloadBytes {
            throw MediaDecodeError.payloadTooLarge(
                field: field, bytes: data.count, limit: maxPayloadBytes
            )
        }

        let ext = preferredExtension
            ?? (sniffImage ? sniffImageExtension(data) : "bin")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-\(field)-\(UUID().uuidString).\(ext)")
        do {
            try data.write(to: url)
        } catch {
            throw MediaDecodeError.writeFailed(error.localizedDescription)
        }
        return url.path
    }

    /// Coerce an `images` field into a `[String]` regardless of whether the
    /// caller passed a single string or an array.
    static func coerceStringArray(_ raw: Any?) -> [String]? {
        guard let raw else { return nil }
        if let s = raw as? String { return [s] }
        if let arr = raw as? [String] { return arr }
        if let arr = raw as? [Any] {
            var out: [String] = []
            for item in arr {
                guard let s = item as? String else { return nil }
                out.append(s)
            }
            return out
        }
        return nil
    }
}
