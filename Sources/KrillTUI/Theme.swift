import Foundation

/// Terminal background, used to pick a readable monochrome palette. The chat TUI
/// shaded speaker turns with absolute colors (bright-white user, gray model)
/// that only read on a DARK terminal; on a light background the user turn was
/// invisible. `Theme` resolves the background once at startup and hands back a
/// `Palette` so the styling adapts. All of this is pure (no terminal I/O) so it
/// is unit-tested; the actual env reads / OSC 11 query live in the CLI layer.
public enum Background: Sendable, Equatable {
    case dark, light, unknown
}

/// SGR codes for the three shade roles, or `nil` to mean "use the terminal's own
/// foreground unstyled" (which is always readable, whatever the background).
public struct Palette: Sendable, Equatable {
    public let userSGR: String?
    public let modelSGR: String?
    public let chromeSGR: String?
    public init(userSGR: String?, modelSGR: String?, chromeSGR: String?) {
        self.userSGR = userSGR
        self.modelSGR = modelSGR
        self.chromeSGR = chromeSGR
    }
}

public enum Theme {
    /// Resolve the background from an explicit override and the `COLORFGBG` env
    /// var, without touching the terminal. Returns `.unknown` when neither
    /// decides (the caller may then try an OSC 11 query, else fall back to the
    /// always-safe palette).
    ///
    /// - `override`: `"light"` / `"dark"` force the result; `"auto"`/`nil`/other
    ///   fall through.
    /// - `colorFGBG`: e.g. `"15;0"` (dark bg) or `"0;15"` (light bg); some
    ///   terminals emit a middle field (`"fg;default;bg"`) so the LAST field is
    ///   the background color index. rxvt convention: indices 7 and 9..15 are
    ///   light, everything else dark.
    public static func resolve(override: String?, colorFGBG: String?) -> Background {
        switch override?.lowercased() {
        case "light": return .light
        case "dark": return .dark
        default: break
        }
        if let last = colorFGBG?.split(separator: ";").last, let n = Int(last) {
            return (n == 7 || (9...15).contains(n)) ? .light : .dark
        }
        return .unknown
    }

    /// Map a perceived background luminance (0 dark .. 1 light), e.g. from an
    /// OSC 11 reply, to a background.
    public static func background(forLuminance lum: Double) -> Background {
        lum > 0.5 ? .light : .dark
    }

    /// The palette for a background. Dark keeps the original dramatic shading
    /// (bright-white user, gray model). Light swaps the user turn to bold default
    /// (dark-on-light, high contrast) and keeps a medium gray for the model.
    /// Unknown uses only relative attributes (bold / normal / dim) that derive
    /// from the terminal's own foreground, so it can never be unreadable.
    public static func palette(for background: Background) -> Palette {
        switch background {
        case .dark:    return Palette(userSGR: "97", modelSGR: "90", chromeSGR: "2")
        case .light:   return Palette(userSGR: "1",  modelSGR: "90", chromeSGR: "2")
        case .unknown: return Palette(userSGR: "1",  modelSGR: nil,  chromeSGR: "2")
        }
    }

    /// Parse the perceived luminance from an OSC 11 reply such as
    /// `ESC ]11;rgb:rrrr/gggg/bbbb BEL`. Channels may be 1..4 hex digits and are
    /// normalised to 0..1; returns `nil` when the string is not a valid reply.
    public static func luminance(fromOSC11 reply: String) -> Double? {
        guard let r = reply.range(of: "rgb:") else { return nil }
        let comps = reply[r.upperBound...].split(separator: "/").prefix(3)
        guard comps.count == 3 else { return nil }
        func channel(_ field: Substring) -> Double? {
            let hex = field.prefix { $0.isHexDigit }
            guard !hex.isEmpty, let v = Int(hex, radix: 16) else { return nil }
            let maxValue = Double((1 << (hex.count * 4)) - 1)
            return Double(v) / maxValue
        }
        guard let rr = channel(comps[0]), let gg = channel(comps[1]),
              let bb = channel(comps[2]) else { return nil }
        return 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
    }
}
