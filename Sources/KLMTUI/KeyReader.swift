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
    case ctrlC, ctrlD, ctrlU, ctrlA, ctrlE, ctrlW, ctrlK, ctrlL
}

/// Decodes raw terminal input bytes into `Key` events. The decoder is pure (a
/// `[UInt8]` chunk maps to `[Key]`) so it is unit-tested without a terminal;
/// `KeyReader` wraps it for live reads. Terminals emit an escape sequence
/// (e.g. Up = `ESC [ A`) as one atomic write, so decoding a read chunk is
/// reliable in practice; a lone trailing `ESC` decodes to `.escape`.
public enum KeyDecoder {
    public static func decode(_ bytes: [UInt8]) -> [Key] {
        var keys: [Key] = []
        var i = 0
        while i < bytes.count {
            let b = bytes[i]
            if b == 0x1b {
                // SGR mouse report ( ESC [ < ... M/m ) - we use it for the wheel.
                if i + 2 < bytes.count, bytes[i + 1] == 0x5b, bytes[i + 2] == 0x3c {
                    if let (key, consumed) = parseMouse(bytes, from: i) {
                        if let key { keys.append(key) }
                        i += consumed
                        continue
                    }
                }
                // OSC ( ESC ] ... terminated by BEL or ST ) - e.g. a terminal
                // color report from an OSC 11 background query. Consume and
                // ignore so the report never leaks into the composer as text.
                if i + 1 < bytes.count, bytes[i + 1] == 0x5d {
                    i = skipOSC(bytes, from: i)
                    continue
                }
                // CSI ( ESC [ ) or SS3 ( ESC O ) sequence?
                if i + 1 < bytes.count, bytes[i + 1] == 0x5b || bytes[i + 1] == 0x4f {
                    if let (key, consumed) = parseEscape(bytes, from: i) {
                        if let key { keys.append(key) }
                        i += consumed
                        continue
                    }
                }
                keys.append(.escape)
                i += 1
            } else if b < 0x80 {
                if let key = decodeControlOrAscii(b) { keys.append(key) }
                i += 1
            } else {
                // UTF-8 multibyte scalar.
                var len = 1
                if b & 0xE0 == 0xC0 { len = 2 }
                else if b & 0xF0 == 0xE0 { len = 3 }
                else if b & 0xF8 == 0xF0 { len = 4 }
                let end = min(i + len, bytes.count)
                if let s = String(bytes: bytes[i..<end], encoding: .utf8), let c = s.first {
                    keys.append(.char(c))
                }
                i = end
            }
        }
        return keys
    }

    /// Consume an OSC sequence `ESC ] ... (BEL | ESC \)` starting at `start`
    /// (where bytes[start] == ESC, bytes[start+1] == ']'). Returns the index
    /// just past the terminator, or past the end if the terminator is not in
    /// this chunk.
    private static func skipOSC(_ bytes: [UInt8], from start: Int) -> Int {
        var j = start + 2
        while j < bytes.count {
            if bytes[j] == 0x07 { return j + 1 }                              // BEL
            if bytes[j] == 0x1b, j + 1 < bytes.count, bytes[j + 1] == 0x5c {  // ST: ESC \
                return j + 2
            }
            j += 1
        }
        return j
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
        case 0x17: return .ctrlW
        default:
            return b >= 0x20 ? .char(Character(UnicodeScalar(b))) : nil
        }
    }

    /// Parse a CSI/SS3 sequence starting at `start` (where bytes[start] == ESC).
    /// Returns the decoded key (nil if recognized-but-ignored) and the number of
    /// bytes consumed, or nil if the sequence is unrecognized/incomplete.
    private static func parseEscape(_ bytes: [UInt8], from start: Int) -> (key: Key?, consumed: Int)? {
        guard start + 2 < bytes.count else { return nil }
        let intro = bytes[start + 1]   // '[' or 'O'
        let third = bytes[start + 2]
        // Single-final-byte forms: ESC [ A  /  ESC O A
        switch third {
        case 0x41: return (.up, 3)
        case 0x42: return (.down, 3)
        case 0x43: return (.right, 3)
        case 0x44: return (.left, 3)
        case 0x48: return (.home, 3)
        case 0x46: return (.end, 3)
        default: break
        }
        // Numeric forms: ESC [ 5 ~  etc.
        if intro == 0x5b, third >= 0x30, third <= 0x39, start + 3 < bytes.count, bytes[start + 3] == 0x7e {
            switch third {
            case 0x31, 0x37: return (.home, 4)
            case 0x34, 0x38: return (.end, 4)
            case 0x33: return (.delete, 4)
            case 0x35: return (.pageUp, 4)
            case 0x36: return (.pageDown, 4)
            default: return (nil, 4)
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

/// Live reader: puts the terminal-derived bytes through `KeyDecoder`.
public struct KeyReader {
    public init() {}
    /// Blocking read of the next batch of keys from `fd` (default stdin). Returns
    /// `nil` on EOF (the syscall read 0 bytes / errored) and a (possibly EMPTY)
    /// array otherwise. An empty array means bytes WERE read but decoded to no
    /// actionable key - a mouse click, a focus event, a terminal color report.
    /// Callers MUST distinguish this from EOF: treating "decoded to no keys" as
    /// EOF makes the TUI quit on a stray mouse/terminal event.
    public func read(fd: Int32 = 0) -> [Key]? {
        var buf = [UInt8](repeating: 0, count: 64)
        let n = Foundation.read(fd, &buf, buf.count)
        guard n > 0 else { return nil }
        return KeyDecoder.decode(Array(buf[0..<n]))
    }
}
