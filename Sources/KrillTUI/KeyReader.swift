import Foundation

/// A decoded keypress from the terminal.
public enum Key: Equatable {
    case char(Character)
    case enter
    case tab
    case backspace
    case escape
    case up, down, left, right
    case home, end, pageUp, pageDown, delete
    case scrollUp, scrollDown
    case ctrlC, ctrlD, ctrlU, ctrlA, ctrlE, ctrlW, ctrlK, ctrlL, ctrlV, ctrlT
}

/// Decodes raw terminal input bytes into `Key` events. The decoder is pure (a
/// `[UInt8]` chunk maps to `[Key]`) so it is unit-tested without a terminal;
/// `KeyReader` wraps it for live reads. Terminals emit an escape sequence
/// (e.g. Up = `ESC [ A`) as one atomic write, so decoding a read chunk is
/// reliable in practice; a lone trailing `ESC` decodes to `.escape`.
public enum KeyDecoder {
    /// Decode a COMPLETE chunk (used by tests). Any trailing incomplete escape /
    /// UTF-8 sequence is dropped.
    public static func decode(_ bytes: [UInt8]) -> [Key] {
        decodeStreaming(bytes).keys
    }

    /// Decode a chunk that may end mid-sequence because a terminal write was
    /// split across reads (common when a fast trackpad scroll emits a burst of
    /// SGR mouse reports). Returns the decoded keys plus any trailing bytes that
    /// form an INCOMPLETE escape or UTF-8 sequence; the caller prepends those to
    /// the next read so a split sequence is never mis-decoded into stray text.
    public static func decodeStreaming(_ bytes: [UInt8]) -> (keys: [Key], remainder: [UInt8]) {
        var keys: [Key] = []
        var i = 0
        while i < bytes.count {
            let b = bytes[i]
            if b == 0x1b {
                switch escapeAt(bytes, from: i) {
                case .complete(let key, let consumed):
                    if let key { keys.append(key) }
                    i += consumed
                case .incomplete:
                    return (keys, Array(bytes[i...]))
                }
            } else if b < 0x80 {
                if let key = decodeControlOrAscii(b) { keys.append(key) }
                i += 1
            } else {
                // UTF-8 multibyte scalar; if truncated at the buffer end, buffer it.
                var len = 1
                if b & 0xE0 == 0xC0 { len = 2 }
                else if b & 0xF0 == 0xE0 { len = 3 }
                else if b & 0xF8 == 0xF0 { len = 4 }
                if i + len > bytes.count { return (keys, Array(bytes[i...])) }
                if let s = String(bytes: bytes[i..<i + len], encoding: .utf8), let c = s.first {
                    keys.append(.char(c))
                }
                i += len
            }
        }
        return (keys, [])
    }

    private enum Esc { case complete(key: Key?, consumed: Int); case incomplete }

    /// Dispatch an ESC-initiated sequence at `start`. Returns `.incomplete` when
    /// a sequence has begun but its terminator is not yet in `bytes` (a split
    /// read), so the caller can buffer and retry on the next read.
    private static func escapeAt(_ bytes: [UInt8], from start: Int) -> Esc {
        // Lone trailing ESC: treat as the Escape key (don't stall a real press).
        guard start + 1 < bytes.count else { return .complete(key: .escape, consumed: 1) }
        let intro = bytes[start + 1]
        // OSC ( ESC ] ... BEL | ST ) - terminal reports (e.g. an OSC 11 color
        // answer); consume and ignore so the report never leaks in as text.
        if intro == 0x5d {
            if let end = oscEnd(bytes, from: start) { return .complete(key: nil, consumed: end - start) }
            return .incomplete
        }
        // CSI ( ESC [ ) or SS3 ( ESC O ).
        if intro == 0x5b || intro == 0x4f {
            // SGR mouse ( ESC [ < ... M/m ) - the wheel; clicks/motion ignored.
            if intro == 0x5b, start + 2 < bytes.count, bytes[start + 2] == 0x3c {
                if let (key, consumed) = parseMouse(bytes, from: start) {
                    return .complete(key: key, consumed: consumed)
                }
                return .incomplete
            }
            return parseCSI(bytes, from: start)
        }
        // ESC + any other byte: treat the ESC itself as the Escape key.
        return .complete(key: .escape, consumed: 1)
    }

    /// Index just past an OSC terminator (BEL or ST = `ESC \`), or nil if the
    /// sequence is not yet terminated within `bytes`.
    private static func oscEnd(_ bytes: [UInt8], from start: Int) -> Int? {
        var j = start + 2
        while j < bytes.count {
            if bytes[j] == 0x07 { return j + 1 }
            if bytes[j] == 0x1b {
                guard j + 1 < bytes.count else { return nil }
                if bytes[j + 1] == 0x5c { return j + 2 }
            }
            j += 1
        }
        return nil
    }

    private static func decodeControlOrAscii(_ b: UInt8) -> Key? {
        switch b {
        case 0x0d, 0x0a: return .enter
        case 0x09: return .tab
        case 0x7f, 0x08: return .backspace
        case 0x01: return .ctrlA
        case 0x03: return .ctrlC
        case 0x04: return .ctrlD
        case 0x05: return .ctrlE
        case 0x0b: return .ctrlK
        case 0x0c: return .ctrlL
        case 0x15: return .ctrlU
        case 0x14: return .ctrlT
        case 0x16: return .ctrlV
        case 0x17: return .ctrlW
        default:
            return b >= 0x20 ? .char(Character(UnicodeScalar(b))) : nil
        }
    }

    /// Parse a non-mouse CSI/SS3 sequence at `start`. A CSI/SS3 sequence ends at
    /// the first final byte (0x40..0x7e); parameter (0x30..0x3f) and intermediate
    /// (0x20..0x2f) bytes may precede it. Returns `.incomplete` if no final byte
    /// is in `bytes` yet, else `.complete` with the mapped key (nil = a valid but
    /// unhandled sequence, e.g. a focus event, which is consumed and ignored).
    private static func parseCSI(_ bytes: [UInt8], from start: Int) -> Esc {
        var j = start + 2
        while j < bytes.count {
            let c = bytes[j]
            if c >= 0x40, c <= 0x7e {
                return .complete(key: mapCSI(bytes, start: start, finalIndex: j), consumed: j - start + 1)
            }
            if c >= 0x20, c <= 0x3f { j += 1; continue }   // parameter / intermediate
            return .complete(key: nil, consumed: j - start)  // control byte: malformed, drop
        }
        return .incomplete
    }

    /// Map a complete CSI/SS3 sequence to a key, or nil if it is recognized but
    /// not one we handle.
    private static func mapCSI(_ bytes: [UInt8], start: Int, finalIndex: Int) -> Key? {
        let final = bytes[finalIndex]
        // Single-final forms: ESC [ A  /  ESC O A  (arrows, home, end).
        if finalIndex == start + 2 {
            switch final {
            case 0x41: return .up
            case 0x42: return .down
            case 0x43: return .right
            case 0x44: return .left
            case 0x48: return .home
            case 0x46: return .end
            default: return nil
            }
        }
        // Numeric tilde forms: ESC [ <n> ~
        if bytes[start + 1] == 0x5b, final == 0x7e, finalIndex == start + 3 {
            switch bytes[start + 2] {
            case 0x31, 0x37: return .home
            case 0x34, 0x38: return .end
            case 0x33: return .delete
            case 0x35: return .pageUp
            case 0x36: return .pageDown
            default: return nil
            }
        }
        return nil
    }

    /// Parse an SGR mouse report `ESC [ < Cb ; Cx ; Cy (M|m)` starting at `start`.
    /// We only care about the wheel: button code 64 = wheel up, 65 = wheel down;
    /// every other mouse event (clicks, motion) is recognized-but-ignored so it
    /// does not leak into the input as stray characters.
    private static func parseMouse(_ bytes: [UInt8], from start: Int) -> (key: Key?, consumed: Int)? {
        var j = start + 3   // skip ESC [ <
        var cb = 0
        var hasDigit = false
        while j < bytes.count {
            let c = bytes[j]
            if c >= 0x30, c <= 0x39 {
                if !hasDigit { cb = 0 }       // only the first field (button) matters
                if cb < 1_000_000 { cb = cb * 10 + Int(c - 0x30) }
                hasDigit = true
                j += 1
            } else if c == 0x3b {              // ';' end of the button field
                break
            } else if c == 0x4d || c == 0x6d { // 'M'/'m' with no ';' (malformed); stop
                break
            } else {
                return nil
            }
        }
        // Skip to the terminating M/m.
        while j < bytes.count, bytes[j] != 0x4d, bytes[j] != 0x6d { j += 1 }
        guard j < bytes.count else { return nil }     // incomplete report
        let consumed = j - start + 1
        guard hasDigit else { return (nil, consumed) }
        if cb == 64 { return (.scrollUp, consumed) }
        if cb == 65 { return (.scrollDown, consumed) }
        return (nil, consumed)
    }
}

/// Live reader: puts the terminal-derived bytes through `KeyDecoder`, carrying a
/// trailing incomplete escape sequence across reads so a split (e.g. a scroll
/// burst chopped at the 4 KB read boundary) is reassembled rather than leaking
/// its bytes into the input as text.
public final class KeyReader {
    private var pending: [UInt8] = []
    public init() {}
    /// Blocking read of the next batch of keys from `fd` (default stdin). Returns
    /// `nil` on EOF (the syscall read 0 bytes / errored) and a (possibly EMPTY)
    /// array otherwise. An empty array means bytes WERE read but decoded to no
    /// actionable key - a mouse click, a focus event, a terminal color report.
    /// Callers MUST distinguish this from EOF: treating "decoded to no keys" as
    /// EOF makes the TUI quit on a stray mouse/terminal event.
    public func read(fd: Int32 = 0) -> [Key]? {
        var buf = [UInt8](repeating: 0, count: 4096)
        let n = Foundation.read(fd, &buf, buf.count)
        guard n > 0 else {
            // True EOF. Surface any buffered partial bytes once, then signal EOF.
            if pending.isEmpty { return nil }
            let leftover = KeyDecoder.decode(pending); pending = []
            return leftover
        }
        var chunk = pending
        chunk.append(contentsOf: buf[0..<n])
        pending = []
        let (keys, remainder) = KeyDecoder.decodeStreaming(chunk)
        // A well-formed split is at most a few bytes; if the remainder grows
        // implausibly large it is malformed, so flush it rather than buffer
        // forever (it would otherwise stall input).
        if remainder.count > 64 {
            return keys + KeyDecoder.decode(remainder)
        }
        pending = remainder
        return keys
    }
}
