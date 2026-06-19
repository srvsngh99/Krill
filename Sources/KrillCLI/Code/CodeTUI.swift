import Foundation
import KrillHarness
import KrillTUI

// File-private SIGWINCH plumbing for the code TUI. Separate from ChatTUI's
// (signal handlers are file-private); only one full-screen TUI runs at a time,
// so each installs its own handler when it starts.
private nonisolated(unsafe) var codeWinchFlag: sig_atomic_t = 0
private func codeWinchHandler(_ sig: Int32) { codeWinchFlag = 1 }

/// Thread-safe hand-off of `AgentEvent`s from the agent run Task to the render
/// loop. The run Task only ever touches this queue; all `CodeTUI` UI state lives
/// on the main task, so there is no shared mutable UI state across tasks.
private final class EventQueue: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [AgentEvent] = []
    private var done = false

    func push(_ e: AgentEvent) { lock.lock(); events.append(e); lock.unlock() }
    func markDone() { lock.lock(); done = true; lock.unlock() }

    func drain() -> [AgentEvent] {
        lock.lock(); defer { lock.unlock() }
        let out = events
        events.removeAll()
        return out
    }

    /// The run finished AND every event it produced has been drained.
    var isFinished: Bool {
        lock.lock(); defer { lock.unlock() }
        return done && events.isEmpty
    }
}

/// Full-screen, multi-turn TUI for `krill code`. Runs the initial task, then
/// hosts an input box for follow-up tasks that continue the same conversation
/// (via `AgentLoop`'s `priorMessages` seam). Each run is rendered live -
/// assistant prose, tool-call chips, diffs - in a scrollable transcript with a
/// working spinner, masthead/footer chrome, scroll, resize, and Ctrl-C cancel.
/// Reuses the same raw-terminal / key-decoder / geometry primitives as the chat
/// TUI.
///
/// One agent run at a time: while a run is in flight the input box is inert
/// (keys drive cancel/scroll only). Ask mode is not hosted here (its approval
/// prompt uses the classic line renderer), so the loop never blocks on input
/// mid-run.
final class CodeTUI {
    private enum Entry {
        case user(String)
        case assistant(String)
        case toolCall(name: String, args: String)
        case toolResult(content: String, isError: Bool)
        case finalAnswer(String)
        case note(String)
    }

    private let loop: AgentLoop
    private let initialTask: String
    private let system: String?
    private let modelName: String

    private let raw = RawTerminal()
    private let reader = KeyReader()

    private var model: [Entry] = []
    private var conversation: [[String: String]] = []   // carried across turns
    private var rows = 24
    private var cols = 80
    private var scrollOffset = 0          // lines scrolled up from the bottom
    private var chipShown = false         // a chip was emitted for the in-flight call
    private var status = "ready"

    // Composer state (the input box for follow-up tasks).
    private var input = ""
    private var cursor = 0
    private var inputHistory: [String] = []
    private var historyIndex = 0
    private var shouldQuit = false

    init(loop: AgentLoop, task: String, system: String?, modelName: String) {
        self.loop = loop
        self.initialTask = task
        self.system = system
        self.modelName = modelName
    }

    func run() async {
        raw.enter()
        resolveTheme()
        installWinch()
        updateSize()
        defer { raw.leave() }

        // Run the task the user launched with, then drop into the REPL.
        await runAgent(initialTask)
        render(working: false, spin: 0)

        while !shouldQuit {
            if codeWinchFlag != 0 { codeWinchFlag = 0; updateSize(); render(working: false, spin: 0) }
            guard raw.waitForInput(timeoutMs: 250) else { continue }
            guard let keys = reader.read() else { break }   // EOF
            var submit: String?
            for key in keys {
                if let text = handleInputKey(key) { submit = text }
                if shouldQuit { break }
            }
            render(working: false, spin: 0)
            if let task = submit, !shouldQuit {
                await runAgent(task)
                render(working: false, spin: 0)
            }
        }
    }

    // MARK: - One agent run (live render)

    private func runAgent(_ task: String) async {
        model.append(.user(task))
        status = "working"
        scrollOffset = 0
        chipShown = false

        let queue = EventQueue()
        let theLoop = loop, sys = system, prior = conversation
        let runTask = Task {
            let t = await theLoop.run(
                user: task, system: sys, priorMessages: prior,
                onEvent: { ev in queue.push(ev) })
            queue.markDone()
            return t
        }

        var spin = 0
        while true {
            if codeWinchFlag != 0 { codeWinchFlag = 0; updateSize() }
            for ev in queue.drain() { apply(ev) }
            let finished = queue.isFinished
            render(working: !finished, spin: spin)
            if finished { break }
            spin &+= 1
            if raw.waitForInput(timeoutMs: 120), let keys = reader.read() {
                for key in keys {
                    switch key {
                    case .ctrlC: runTask.cancel(); status = "cancelling"
                    case .pageUp, .scrollUp: scroll(by: pageStep())
                    case .pageDown, .scrollDown: scroll(by: -pageStep())
                    case .up: scroll(by: 1)
                    case .down: scroll(by: -1)
                    default: break
                    }
                }
            }
        }

        let transcript = await runTask.value
        conversation = transcript.messages   // continue from here next turn
        if transcript.wasCancelled { status = "cancelled" }
        else if transcript.hitIterationLimit { status = "iteration limit" }
        else { status = "ready" }
    }

    // MARK: - Events -> transcript

    private func apply(_ event: AgentEvent) {
        switch event {
        case .assistantTurn(let text):
            if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                model.append(.assistant(text))
            }
        case .toolStarted(let name, let args):
            model.append(.toolCall(name: name, args: args))
            chipShown = true
        case .toolFinished(let inv):
            // Denied / unknown tools never emit toolStarted, so add their chip
            // here so the observation is never orphaned.
            if !chipShown {
                model.append(.toolCall(name: inv.name, args: inv.argumentsJSON))
            }
            model.append(.toolResult(content: inv.result.content, isError: inv.result.isError))
            chipShown = false
        case .finalAnswer(let text):
            model.append(.finalAnswer(text))
        case .iterationLimitReached:
            model.append(.note("[stopped at the iteration limit without a final answer]"))
        case .cancelled:
            model.append(.note("[cancelled]"))
        }
        scrollOffset = 0   // new content jumps the view back to the bottom
    }

    // MARK: - Input handling (composer)

    private func handleInputKey(_ key: Key) -> String? {
        switch key {
        case .ctrlD:
            if input.isEmpty { shouldQuit = true }
            return nil
        case .ctrlC:
            if input.isEmpty { shouldQuit = true } else { input = ""; cursor = 0 }
            return nil
        case .ctrlU: input = ""; cursor = 0; return nil
        case .ctrlA, .home: cursor = 0; return nil
        case .ctrlE, .end: cursor = input.count; return nil
        case .pageUp: scroll(by: pageStep()); return nil
        case .pageDown: scroll(by: -pageStep()); return nil
        case .scrollUp: scroll(by: 3); return nil
        case .scrollDown: scroll(by: -3); return nil
        case .left: if cursor > 0 { cursor -= 1 }; return nil
        case .right: if cursor < input.count { cursor += 1 }; return nil
        case .up: recallHistory(delta: -1); return nil
        case .down: recallHistory(delta: 1); return nil
        case .backspace:
            if cursor > 0 {
                let idx = input.index(input.startIndex, offsetBy: cursor - 1)
                input.remove(at: idx); cursor -= 1
            }
            return nil
        case .delete:
            if cursor < input.count {
                let idx = input.index(input.startIndex, offsetBy: cursor)
                input.remove(at: idx)
            }
            return nil
        case .enter:
            let text = input.trimmingCharacters(in: .whitespaces)
            input = ""; cursor = 0; scrollOffset = 0
            guard !text.isEmpty else { return nil }
            inputHistory.append(text); historyIndex = inputHistory.count
            return text
        case .char(let c):
            let idx = input.index(input.startIndex, offsetBy: cursor)
            input.insert(c, at: idx); cursor += 1
            return nil
        default:
            return nil
        }
    }

    private func recallHistory(delta: Int) {
        guard !inputHistory.isEmpty else { return }
        let ni = max(0, min(inputHistory.count, historyIndex + delta))
        historyIndex = ni
        if ni == inputHistory.count { input = ""; cursor = 0 }
        else { input = inputHistory[ni]; cursor = input.count }
    }

    // MARK: - Render

    private func render(working: Bool, spin: Int) {
        let width = cols
        let paneTop = 3
        let box = inputBox(width: width, working: working)   // 3 lines
        let boxTop = max(paneTop + 1, rows - box.count)
        let footerRow = max(rows, boxTop + box.count)
        let convHeight = max(1, boxTop - paneTop)

        var frame = "\u{1B}[H"
        frame += positioned(1, Brand.header(width: width, model: modelName))
        frame += positioned(2, Brand.headerRule(width: width))

        let pane = transcriptLines(width: width)
        let maxStart = max(0, pane.count - convHeight)
        scrollOffset = min(max(0, scrollOffset), maxStart)
        let blankTop = Chrome.anchorBlankTop(paneCount: pane.count, convHeight: convHeight, centered: false)
        let start = max(0, maxStart - scrollOffset)
        for i in 0..<convHeight {
            let content: String
            if i < blankTop {
                content = ""
            } else {
                let idx = start + (i - blankTop)
                content = idx < pane.count ? pane[idx] : ""
            }
            frame += positioned(paneTop + i, content)
        }

        for (i, line) in box.enumerated() { frame += positioned(boxTop + i, line) }
        let (left, right) = footer(working: working, spin: spin)
        frame += positioned(footerRow, Brand.footer(width: width, left: left, right: right))
        frame += "\u{1B}[J"
        Output.write(frame)
    }

    private func inputBox(width: Int, working: Bool) -> [String] {
        let w = max(8, width)
        let h = "\u{2500}", v = "\u{2502}"
        let tl = "\u{256D}", tr = "\u{256E}", bl = "\u{2570}", br = "\u{256F}"
        let top = Ansi.chrome(Chrome.border(width: w, left: tl, fill: h, right: tr))
        let bottom = Ansi.chrome(Chrome.border(width: w, left: bl, fill: h, right: br))

        let fieldWidth = max(2, w - 4)
        let promptStr = "> "
        let textWidth = max(1, fieldWidth - promptStr.count)

        var body: String
        if working {
            let msg = "working\u{2026}"
            let clipped = String(msg.prefix(textWidth))
            body = Ansi.chrome(promptStr) + Ansi.dim(clipped)
                + String(repeating: " ", count: max(0, textWidth - clipped.count))
        } else {
            let chars = Array(input)
            if chars.isEmpty {
                let placeholder = "type a follow-up task   Ctrl-D to exit"
                let clipped = String(placeholder.prefix(max(0, textWidth - 1)))
                let pad = String(repeating: " ", count: max(0, textWidth - 1 - clipped.count))
                body = Ansi.bold(promptStr) + Ansi.inverse(" ") + Ansi.chrome(clipped) + pad
            } else {
                let (content, cursorCol) = Chrome.inputField(text: chars, cursor: cursor, textWidth: textWidth)
                var rendered = ""
                for (i, ch) in Array(content).enumerated() {
                    rendered += i == cursorCol ? Ansi.inverse(String(ch)) : String(ch)
                }
                body = Ansi.bold(promptStr) + rendered
            }
        }
        let field = Ansi.chrome(v) + " " + body + " " + Ansi.chrome(v)
        return [top, field, bottom]
    }

    private func footer(working: Bool, spin: Int) -> (String, String) {
        let dots = ["\u{2839}", "\u{2838}", "\u{2834}", "\u{2826}", "\u{2827}", "\u{2807}", "\u{280F}", "\u{2819}"]
        if working {
            return ("\(dots[spin % dots.count]) \(status)", "Ctrl-C cancel  PgUp/Dn scroll")
        }
        return (status, "Enter run  Ctrl-D exit  PgUp/Dn scroll")
    }

    private func transcriptLines(width: Int) -> [String] {
        var out: [String] = []
        func push(_ lines: [CodeLine]) { for l in lines { out.append(styled(l)) } }
        for entry in model {
            switch entry {
            case .user(let t):
                out.append("")
                push(CodeView.userTask(t, width: width))
            case .assistant(let t):
                for l in TUIMarkdown.render(t, width: width) { out.append(Ansi.model(l)) }
                out.append("")
            case .toolCall(let name, let args):
                push(CodeView.toolCall(name: name, argumentsJSON: args, width: width))
            case .toolResult(let content, let isError):
                push(CodeView.toolResult(content: content, isError: isError, width: width, maxLines: 14))
                out.append("")
            case .finalAnswer(let t):
                for l in TUIMarkdown.render(t, width: width) { out.append(Ansi.model(l)) }
                out.append("")
            case .note(let t):
                push(CodeView.note(t, width: width))
                out.append("")
            }
        }
        return out
    }

    private func styled(_ line: CodeLine) -> String {
        switch line.style {
        case .user: return Ansi.user(line.text)
        case .assistant: return Ansi.model(line.text)
        case .toolName: return Ansi.cyan(line.text)
        case .toolOk: return Ansi.dim(line.text)
        case .toolError: return Ansi.red(line.text)
        case .diffAdd: return Ansi.green(line.text)
        case .diffDel: return Ansi.red(line.text)
        case .note: return Ansi.yellow(line.text)
        case .dim: return Ansi.dim(line.text)
        }
    }

    private func positioned(_ row: Int, _ content: String) -> String {
        "\u{1B}[\(row);1H\u{1B}[2K" + content
    }

    // MARK: - Scroll / size / theme / signals

    private func scroll(by delta: Int) { scrollOffset = max(0, scrollOffset + delta) }

    private func pageStep() -> Int { max(1, (rows - 6) - 1) }

    private func updateSize() { let s = raw.size(); rows = s.rows; cols = s.cols }

    private func resolveTheme() {
        let env = ProcessInfo.processInfo.environment
        let override = env["KRILL_TUI_THEME"]
        var bg = Theme.resolve(override: override, colorFGBG: env["COLORFGBG"])
        if bg == .unknown, override == nil || override?.lowercased() == "auto" {
            if let lum = raw.queryBackgroundLuminance() { bg = Theme.background(forLuminance: lum) }
        }
        Ansi.theme = Theme.palette(for: bg)
    }

    private func installWinch() {
        var unblock = sigset_t(); sigemptyset(&unblock); sigaddset(&unblock, SIGWINCH)
        pthread_sigmask(SIG_UNBLOCK, &unblock, nil)
        signal(SIGWINCH, codeWinchHandler)
    }
}
