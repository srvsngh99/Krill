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
                // CSI ( ESC [ ) or SS3 ( ESC O ) sequence?
                if i + 2 < bytes.count + 1, i + 1 < bytes.count, bytes[i + 1] == 0x5b || bytes[i + 1] == 0x4f {
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
}

/// Live reader: puts the terminal-derived bytes through `KeyDecoder`.
public struct KeyReader {
    public init() {}
    /// Blocking read of the next batch of keys from `fd` (default stdin). Returns
    /// an empty array on EOF.
    public func read(fd: Int32 = 0) -> [Key] {
        var buf = [UInt8](repeating: 0, count: 64)
        let n = Foundation.read(fd, &buf, buf.count)
        guard n > 0 else { return [] }
        return KeyDecoder.decode(Array(buf[0..<n]))
    }
}
