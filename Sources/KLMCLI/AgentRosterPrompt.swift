import Foundation
import KLMTUI
#if canImport(Darwin)
import Darwin
#endif

/// The branded roster shown by `krillm launch` with no agent argument. On an
/// interactive terminal it arrow-selects a coding agent to launch (the pure
/// ``AgentPicker`` drives selection; this layer renders + reads keys). On a
/// non-TTY (piped / redirected) it prints a static branded list and returns nil,
/// so scripts and `| cat` keep getting plain, scrollable output.
enum AgentRosterPrompt {

    /// Resolve a launch choice. Returns the chosen agent id, or nil when the
    /// user cancels (Esc/Ctrl-C/q) or stdout is not an interactive terminal (the
    /// static roster is printed in that case so the user still sees the list).
    static func choose(profiles: [AgentProfile]) -> String? {
        let entries = self.entries(from: profiles)
        guard Ansi.enabled, isatty(STDIN_FILENO) != 0, !entries.isEmpty else {
            printStatic(entries)
            return nil
        }
        return runInteractive(AgentPicker(entries: entries))
    }

    /// Build picker entries from the profiles, flagging each agent installed/not
    /// by a PATH lookup of its binary.
    static func entries(from profiles: [AgentProfile]) -> [AgentPicker.Entry] {
        profiles.map {
            AgentPicker.Entry(id: $0.id, displayName: $0.displayName,
                              summary: $0.summary, installed: isOnPath($0.binary))
        }
    }

    /// Print the static branded roster (used on a non-TTY and for the
    /// unknown-agent error path).
    static func printStatic(_ entries: [AgentPicker.Entry]) {
        printStaticRoster(entries)
    }

    // MARK: - Rendering

    private static let margin = "  "

    /// Branded masthead drawn above the roster on every frame.
    private static func head(width: Int) -> [String] {
        [
            margin + Ansi.bold(Brand.wordmark),
            Ansi.chrome(String(repeating: "\u{2500}", count: max(0, width))),
            margin + Ansi.chrome("Launch a coding agent wired to your local KrillLM server."),
            "",
        ]
    }

    /// One roster row. The selected row is bold with a chevron; installed agents
    /// read bright, not-yet-installed ones dim with a "needs install" tag.
    private static func row(_ e: AgentPicker.Entry, selected: Bool, idW: Int, nameW: Int, width: Int) -> String {
        func rpad(_ s: String, _ w: Int) -> String { s.padding(toLength: w, withPad: " ", startingAt: 0) }
        let chevron = selected ? "\u{25B8}" : " "
        let tail = e.installed ? e.summary : e.summary + "  (needs install)"
        let body = "\(chevron) \(rpad(e.id, idW))  \(rpad(e.displayName, nameW))  \(tail)"
        let line = margin + String(body.prefix(max(0, width - margin.count)))
        if selected { return Ansi.bold(line) }
        return e.installed ? Ansi.user(line) : Ansi.chrome(line)
    }

    private static func frame(_ p: AgentPicker, width: Int) -> [String] {
        let idW = max(6, p.entries.map { $0.id.count }.max() ?? 6)
        let nameW = max(10, p.entries.map { $0.displayName.count }.max() ?? 10)
        var out = head(width: width)
        for (i, e) in p.entries.enumerated() {
            out.append(row(e, selected: i == p.selected, idW: idW, nameW: nameW, width: width))
        }
        out.append("")
        out.append(margin + Ansi.chrome("Up/Down  \u{00B7}  Enter launch  \u{00B7}  Esc cancel"))
        return out
    }

    /// Static (non-interactive) roster: the same frame minus the live chevron,
    /// plus the usage line, printed once.
    private static func printStaticRoster(_ entries: [AgentPicker.Entry]) {
        let width = terminalWidth()
        for line in head(width: width) { print(line) }
        let idW = max(6, entries.map { $0.id.count }.max() ?? 6)
        let nameW = max(10, entries.map { $0.displayName.count }.max() ?? 10)
        for e in entries { print(row(e, selected: false, idW: idW, nameW: nameW, width: width)) }
        print("")
        print(margin + Ansi.chrome("Usage:  krillm launch <agent> [--model <name>] [--port <port>] [-- <agent args>]"))
    }

    // MARK: - Interactive driver

    /// Raw-mode arrow-select over the roster, redrawn in place. Stays on the main
    /// screen (no alt buffer) so the chosen line remains in scrollback. Keeps
    /// OPOST on, so `\n` still maps to `\r\n` and plain rendering works; only
    /// echo / canonical mode / key-signals are disabled.
    private static func runInteractive(_ picker: AgentPicker) -> String? {
        var p = picker
        var orig = termios()
        tcgetattr(STDIN_FILENO, &orig)
        var raw = orig
        raw.c_lflag &= ~tcflag_t(ECHO | ICANON | ISIG | IEXTEN)
        raw.c_cc.16 = 1   // VMIN  = 1
        raw.c_cc.17 = 0   // VTIME = 0
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw)
        print("\u{1B}[?25l", terminator: "")   // hide cursor
        defer {
            print("\u{1B}[?25h", terminator: "")   // show cursor
            tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig)
            fflush(stdout)
        }

        let width = terminalWidth()
        var lines = frame(p, width: width)
        draw(lines, moveUp: 0)
        let reader = KeyReader()

        while true {
            guard let keys = reader.read() else { return nil }   // EOF
            var chosen: String?? = nil   // outer nil = no decision; inner nil = cancel
            for key in keys {
                switch key {
                case .up, .scrollUp, .char("k"): p.selectPrevious()
                case .down, .scrollDown, .char("j"): p.selectNext()
                case .enter: chosen = .some(p.current?.id)
                case .escape, .ctrlC, .ctrlD, .char("q"): chosen = .some(nil)
                default: break
                }
            }
            if let decision = chosen { return decision }
            let next = frame(p, width: width)
            draw(next, moveUp: lines.count)
            lines = next
        }
    }

    /// Redraw the frame in place: move the cursor up `moveUp` rows (0 on the
    /// first paint), then clear-and-rewrite each line.
    private static func draw(_ lines: [String], moveUp: Int) {
        var s = ""
        if moveUp > 0 { s += "\u{1B}[\(moveUp)A" }
        for line in lines { s += "\u{1B}[2K" + line + "\n" }
        print(s, terminator: "")
        fflush(stdout)
    }

    // MARK: - Helpers

    private static func terminalWidth() -> Int {
        var ws = winsize()
        if ioctl(STDOUT_FILENO, UInt(TIOCGWINSZ), &ws) == 0, ws.ws_col > 0 { return Int(ws.ws_col) }
        return 80
    }

    /// True if `binary` resolves to an executable on the user's PATH (so we can
    /// flag agents that still need installing without trying to exec them).
    static func isOnPath(_ binary: String) -> Bool {
        if binary.contains("/") {
            return FileManager.default.isExecutableFile(atPath: binary)
        }
        guard let path = ProcessInfo.processInfo.environment["PATH"] else { return false }
        for dir in path.split(separator: ":") {
            let candidate = String(dir) + "/" + binary
            if FileManager.default.isExecutableFile(atPath: candidate) { return true }
        }
        return false
    }
}
