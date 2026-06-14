import Foundation

/// Owns the terminal's raw-mode lifecycle for the full-screen TUI: switches to
/// the alternate screen buffer, puts the tty in raw mode (no echo, no canonical
/// line editing, no terminal-generated signals so Ctrl-C arrives as a byte),
/// hides the cursor, and restores everything on `leave()`. macOS only.
final class RawTerminal {
    private var original = termios()
    private var entered = false

    /// True when stdin and stdout are both interactive terminals.
    static var isInteractive: Bool {
        isatty(STDIN_FILENO) != 0 && isatty(STDOUT_FILENO) != 0
    }

    func enter() {
        guard RawTerminal.isInteractive, !entered else { return }
        tcgetattr(STDIN_FILENO, &original)
        var raw = original
        raw.c_iflag &= ~tcflag_t(BRKINT | ICRNL | INPCK | ISTRIP | IXON)
        raw.c_oflag &= ~tcflag_t(OPOST)
        raw.c_cflag |= tcflag_t(CS8)
        // Disable echo, canonical mode, extended input, and key-generated
        // signals (ISIG) so Ctrl-C/Ctrl-Z reach us as bytes we handle.
        raw.c_lflag &= ~tcflag_t(ECHO | ICANON | IEXTEN | ISIG)
        raw.c_cc.16 = 1   // VMIN  = 1: block for at least one byte
        raw.c_cc.17 = 0   // VTIME = 0: no inter-byte timer
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw)
        Output.write("\u{1B}[?1049h\u{1B}[?25l\u{1B}[2J")  // alt screen, hide cursor, clear
        entered = true
    }

    func leave() {
        guard entered else { return }
        Output.write("\u{1B}[?25h\u{1B}[?1049l")  // show cursor, leave alt screen
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &original)
        entered = false
    }

    /// Current terminal size, falling back to 24x80 if the query fails.
    func size() -> (rows: Int, cols: Int) {
        var ws = winsize()
        if ioctl(STDOUT_FILENO, UInt(TIOCGWINSZ), &ws) == 0, ws.ws_row > 0, ws.ws_col > 0 {
            return (Int(ws.ws_row), Int(ws.ws_col))
        }
        return (24, 80)
    }

    /// Non-blocking poll: true if input is available to read within `ms`.
    func waitForInput(timeoutMs: Int32) -> Bool {
        var fds = pollfd(fd: STDIN_FILENO, events: Int16(POLLIN), revents: 0)
        return poll(&fds, 1, timeoutMs) > 0 && (fds.revents & Int16(POLLIN)) != 0
    }
}

/// Buffered stdout writer for the TUI (one syscall per frame keeps redraws from
/// flickering). Not thread-safe; the TUI writes from a single task.
enum Output {
    static func write(_ s: String) {
        var bytes = Array(s.utf8)
        var off = 0
        while off < bytes.count {
            let n = bytes[off...].withUnsafeBytes { Foundation.write(STDOUT_FILENO, $0.baseAddress, $0.count) }
            if n <= 0 { break }
            off += n
        }
    }
}
