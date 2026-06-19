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

/// Full-screen renderer for a single `krill code` agent run. Drives the agent
/// loop on a child Task and renders its events live (assistant prose, tool-call
/// chips, observations/diffs) in a scrollable transcript, with a working spinner
/// during generation, Ctrl-C cancellation, and resize support. Reuses the same
/// raw-terminal / key-decoder / geometry primitives as the chat TUI.
///
/// Single-run and non-interactive-by-design: it does not host an `ask`-mode
/// approval prompt (that path uses the classic line renderer until an in-TUI
/// approval UI lands), so the loop here never blocks on user input mid-run.
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
    private let task: String
    private let system: String?
    private let modelName: String

    private let raw = RawTerminal()
    private let reader = KeyReader()

    private var model: [Entry] = []
    private var rows = 24
    private var cols = 80
    private var scrollOffset = 0          // lines scrolled up from the bottom
    private var chipShown = false         // a chip was emitted for the in-flight call
    private var status = "working"

    init(loop: AgentLoop, task: String, system: String?, modelName: String) {
        self.loop = loop
        self.task = task
        self.system = system
        self.modelName = modelName
    }

    func run() async {
        raw.enter()
        resolveTheme()
        installWinch()
        updateSize()
        defer { raw.leave() }

        model.append(.user(task))
        render(working: true, spin: 0)

        // Run the agent loop on a child Task; events arrive via the queue.
        let queue = EventQueue()
        let theLoop = loop, theTask = task, theSystem = system
        let runTask = Task {
            _ = await theLoop.run(
                user: theTask, system: theSystem,
                onEvent: { ev in queue.push(ev) })
            queue.markDone()
        }

        // Active phase: animate the spinner, drain events, allow scroll + cancel.
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
                    case .ctrlC:
                        runTask.cancel()
                        status = "cancelling"
                    case .pageUp, .scrollUp: scroll(by: pageStep())
                    case .pageDown, .scrollDown: scroll(by: -pageStep())
                    case .up: scroll(by: 1)
                    case .down: scroll(by: -1)
                    default: break
                    }
                }
            }
        }

        await idleLoop()
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
            status = "done"
        case .iterationLimitReached:
            model.append(.note("[stopped at the iteration limit without a final answer]"))
            status = "iteration limit"
        case .cancelled:
            model.append(.note("[cancelled]"))
            status = "cancelled"
        }
        scrollOffset = 0   // new content jumps the view back to the bottom
    }

    // MARK: - Idle (post-run) loop

    private func idleLoop() async {
        render(working: false, spin: 0)
        while true {
            if codeWinchFlag != 0 { codeWinchFlag = 0; updateSize(); render(working: false, spin: 0) }
            guard raw.waitForInput(timeoutMs: 250) else { continue }
            guard let keys = reader.read() else { break }   // EOF
            var quit = false
            for key in keys {
                switch key {
                case .ctrlC, .ctrlD, .escape, .char("q"): quit = true
                case .pageUp, .scrollUp: scroll(by: pageStep())
                case .pageDown, .scrollDown: scroll(by: -pageStep())
                case .up: scroll(by: 1)
                case .down: scroll(by: -1)
                default: break
                }
            }
            render(working: false, spin: 0)
            if quit { break }
        }
    }

    // MARK: - Render

    private func render(working: Bool, spin: Int) {
        let width = cols
        let convHeight = max(1, rows - 3)   // 1 masthead + 1 rule + 1 footer
        let paneTop = 3

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

        let (left, right) = footer(working: working, spin: spin)
        frame += positioned(rows, Brand.footer(width: width, left: left, right: right))
        frame += "\u{1B}[J"
        Output.write(frame)
    }

    private func footer(working: Bool, spin: Int) -> (String, String) {
        let dots = ["\u{2839}", "\u{2838}", "\u{2834}", "\u{2826}", "\u{2827}", "\u{2807}", "\u{280F}", "\u{2819}"]
        if working {
            return ("\(dots[spin % dots.count]) \(status)", "Ctrl-C cancel")
        }
        return (status, "q quit  \u{2191}/\u{2193} scroll")
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

    private func pageStep() -> Int { max(1, (rows - 3) - 1) }

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
