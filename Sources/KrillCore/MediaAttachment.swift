import Foundation

// MARK: - Interactive media attachment helpers
//
// Pure, dependency-free helpers shared by the CLI's interactive REPL so a user
// can attach an image / audio clip mid-conversation by typing a path, dragging
// a file into the terminal (which pastes its escaped path), or referencing it
// inline with `@path`. Kept in KrillCore so the path/kind/WAV logic is unit
// tested independently of the ArgumentParser wiring in KrillCLI.

/// The kind of media an attached file holds.
public enum MediaKind: String, Sendable, Equatable {
    case image
    case audio
}

public enum MediaAttachment {
    /// Detect whether `data` is an image or audio clip from its magic bytes,
    /// falling back to the file extension when the header is unrecognized.
    /// Returns `nil` when neither the bytes nor the extension identify a
    /// supported media kind (so the caller can treat the token as plain text).
    public static func detectKind(data: Data, pathExtension ext: String) -> MediaKind? {
        if isImageData(data) { return .image }
        if isAudioData(data) { return .audio }

        switch ext.lowercased() {
        case "png", "jpg", "jpeg", "gif", "webp", "bmp", "heic", "heif":
            return .image
        case "wav", "mp3", "flac", "ogg", "oga", "m4a", "aac", "aiff", "aif":
            return .audio
        default:
            return nil
        }
    }

    /// True iff `data` begins with a recognized image container's magic bytes
    /// (PNG, JPEG, GIF87a/89a, WebP). Mirrors the server's `sniffImageExtension`
    /// detection so CLI and HTTP paths agree on what counts as an image.
    public static func isImageData(_ data: Data) -> Bool {
        if data.count >= 8,
           Array(data.prefix(8)) == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
            return true // PNG
        }
        if data.count >= 3, Array(data.prefix(3)) == [0xFF, 0xD8, 0xFF] {
            return true // JPEG
        }
        if data.count >= 6 {
            let p = Array(data.prefix(6))
            if p == [0x47, 0x49, 0x46, 0x38, 0x37, 0x61]
                || p == [0x47, 0x49, 0x46, 0x38, 0x39, 0x61] {
                return true // GIF
            }
        }
        if data.count >= 12 {
            let b = Array(data.prefix(12))
            if Array(b[0..<4]) == [0x52, 0x49, 0x46, 0x46]
                && Array(b[8..<12]) == [0x57, 0x45, 0x42, 0x50] {
                return true // WebP (RIFF....WEBP)
            }
        }
        return false
    }

    /// True iff `data` begins with a recognized audio container's magic bytes
    /// (WAV, FLAC, OGG, MP3 with ID3 or frame-sync, MP4/M4A `ftyp`). The WebP
    /// RIFF check above runs first, so a RIFF/WAVE header is never misread as an
    /// image.
    public static func isAudioData(_ data: Data) -> Bool {
        let b = [UInt8](data.prefix(16))
        guard b.count >= 4 else { return false }
        // RIFF/WAVE
        if b.count >= 12,
           b[0] == 0x52, b[1] == 0x49, b[2] == 0x46, b[3] == 0x46,
           b[8] == 0x57, b[9] == 0x41, b[10] == 0x56, b[11] == 0x45 {
            return true
        }
        // FLAC ("fLaC"), OGG ("OggS")
        if b[0] == 0x66, b[1] == 0x4C, b[2] == 0x61, b[3] == 0x43 { return true }
        if b[0] == 0x4F, b[1] == 0x67, b[2] == 0x67, b[3] == 0x53 { return true }
        // MP3: ID3 tag or MPEG frame sync (11 bits set)
        if b[0] == 0x49, b[1] == 0x44, b[2] == 0x33 { return true }
        if b[0] == 0xFF, (b[1] & 0xE0) == 0xE0 { return true }
        // MP4 / M4A: "....ftyp"
        if b.count >= 8, b[4] == 0x66, b[5] == 0x74, b[6] == 0x79, b[7] == 0x70 {
            return true
        }
        return false
    }

    /// Normalize a path token as it arrives from an interactive terminal so the
    /// three attach styles all resolve to the same filesystem path:
    ///
    ///   * a dragged-in file (Terminal / iTerm paste the path with `\`-escaped
    ///     spaces and special chars),
    ///   * a quoted path (`"/a b/c.png"` or `'/a b/c.png'`),
    ///   * a `~`-relative path,
    ///   * an inline `@path` reference (leading `@` stripped by the caller).
    ///
    /// Returns the absolute, tilde-expanded path. No filesystem access - the
    /// caller checks existence.
    public static func normalizePath(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !s.isEmpty else { return s }

        // Strip one layer of surrounding matching quotes.
        if s.count >= 2 {
            let first = s.first!, last = s.last!
            if (first == "\"" && last == "\"") || (first == "'" && last == "'") {
                s = String(s.dropFirst().dropLast())
            }
        }

        // Un-escape shell-escaped characters (drag-and-drop emits `\ `, `\(`,
        // etc.). Only meaningful for unquoted paths; a stray backslash before a
        // normal char collapses to that char, which matches shell word-splitting.
        if s.contains("\\") {
            var out = ""
            var escaped = false
            for ch in s {
                if escaped { out.append(ch); escaped = false }
                else if ch == "\\" { escaped = true }
                else { out.append(ch) }
            }
            if escaped { out.append("\\") }
            s = out
        }

        // Expand a leading `~` / `~/` to the home directory.
        if s == "~" {
            s = NSHomeDirectory()
        } else if s.hasPrefix("~/") {
            s = NSHomeDirectory() + String(s.dropFirst(1))
        }

        return s
    }

    /// Best-effort pixel dimensions from an image header (PNG, GIF, JPEG).
    /// Returns nil when the format is unrecognized or the header is truncated.
    /// Used for attachment previews, so a miss is harmless.
    public static func imageDimensions(_ data: Data) -> (width: Int, height: Int)? {
        let b = [UInt8](data)
        // PNG: 8-byte signature, then IHDR with width/height as big-endian u32
        // at byte offsets 16 and 20.
        if b.count >= 24, Array(b.prefix(8)) == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
            let w = (Int(b[16]) << 24) | (Int(b[17]) << 16) | (Int(b[18]) << 8) | Int(b[19])
            let h = (Int(b[20]) << 24) | (Int(b[21]) << 16) | (Int(b[22]) << 8) | Int(b[23])
            if w > 0, h > 0 { return (w, h) }
        }
        // GIF: width/height are little-endian u16 at offsets 6 and 8.
        if b.count >= 10, b[0] == 0x47, b[1] == 0x49, b[2] == 0x46, b[3] == 0x38 {
            let w = Int(b[6]) | (Int(b[7]) << 8)
            let h = Int(b[8]) | (Int(b[9]) << 8)
            if w > 0, h > 0 { return (w, h) }
        }
        // JPEG: scan segments for a Start-Of-Frame marker (0xFFC0..0xFFCF,
        // excluding the non-SOF markers C4/C8/CC); height/width follow as
        // big-endian u16.
        if b.count >= 4, b[0] == 0xFF, b[1] == 0xD8 {
            var i = 2
            while i + 9 < b.count {
                guard b[i] == 0xFF else { i += 1; continue }
                let marker = b[i + 1]
                if marker >= 0xC0, marker <= 0xCF, marker != 0xC4, marker != 0xC8, marker != 0xCC {
                    let h = (Int(b[i + 5]) << 8) | Int(b[i + 6])
                    let w = (Int(b[i + 7]) << 8) | Int(b[i + 8])
                    if w > 0, h > 0 { return (w, h) }
                    return nil
                }
                // Skip this segment using its big-endian length field.
                let segLen = (Int(b[i + 2]) << 8) | Int(b[i + 3])
                if segLen < 2 { break }
                i += 2 + segLen
            }
        }
        return nil
    }

    /// Encode a mono Float (`[-1, 1]`) waveform as 16-bit PCM WAV bytes. Used by
    /// live microphone capture: writing WAV at the device's native sample rate
    /// lets the existing `AudioPreprocessor` decode + resample path run
    /// unchanged (it already downsamples to 16 kHz mono).
    public static func encodeWAV(samples: [Float], sampleRate: Int) -> Data {
        let channels = 1
        let bitsPerSample = 16
        let byteRate = sampleRate * channels * bitsPerSample / 8
        let blockAlign = channels * bitsPerSample / 8
        let dataBytes = samples.count * bitsPerSample / 8

        var d = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }

        d.append(contentsOf: Array("RIFF".utf8))
        u32(UInt32(36 + dataBytes))
        d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8))
        u32(16)                       // PCM fmt chunk size
        u16(1)                        // audio format = PCM
        u16(UInt16(channels))
        u32(UInt32(sampleRate))
        u32(UInt32(byteRate))
        u16(UInt16(blockAlign))
        u16(UInt16(bitsPerSample))
        d.append(contentsOf: Array("data".utf8))
        u32(UInt32(dataBytes))

        d.reserveCapacity(d.count + dataBytes)
        for s in samples {
            let clamped = max(-1.0, min(1.0, s))
            let i = Int16(clamping: Int((clamped * 32767.0).rounded()))
            u16(UInt16(bitPattern: i))
        }
        return d
    }
}
