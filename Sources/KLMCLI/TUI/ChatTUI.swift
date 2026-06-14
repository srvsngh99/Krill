import Foundation
import KLMCore
import KLMEngine
import KLMRegistry
import KLMSampler
import KLMServer
import KLMTUI

// Terminal resize (SIGWINCH) sets this flag; the run loop polls it and re-reads
// the size. A signal handler may only touch a sig_atomic_t global.
private nonisolated(unsafe) var tuiWinchFlag: sig_atomic_t = 0
private func tuiWinchHandler(_ sig: Int32) { tuiWinchFlag = 1 }

/// Full-screen opencode-style chat TUI for KrillLM, in the Sourav AI Labs
/// monochrome identity: a branded masthead, a scrollable conversation pane, a
/// bottom input box with a slash-command autosuggest popup (Up/Down to cycle),
/// and a status footer. Falls back to the line REPL when not on a TTY.
final class ChatTUI {
    // Model / session
    private var engine: InferenceEngine
    private var modelName: String
    private var system: String?
    private let params: SamplingParams
    private let maxTokens: Int
    private let registry: Registry

    // Conversation
    private struct Msg { enum Role { case user, assistant, note }; let role: Role; var text: String }
    private var view: [Msg] = []
    private var modelTurns: [(role: String, content: String)] = []

    // Attachments
    private struct Att { let kind: MediaKind; let data: Data; let name: String; let dims: (Int, Int)? }
    private var pendingImages: [Att] = []
    private var pendingAudio: Att?

    // Input + UI state
    private var input = ""
    private var cursor = 0           // index into `input`
    private var menu = SlashMenu()
    private var inputHistory: [String] = []
    private var historyIndex = 0
    private var scrollOffset = 0     // lines scrolled up from bottom
    private var rows = 24, cols = 80
    private var lastStatus = ""
    private var contextWindow = 0          // model's max context (tokens), 0 = unknown
    private var shouldQuit = false

    // Working-directory label for the footer (e.g. "KrillLM:main"). Computed once.
    private lazy var cwdLabel: String = {
        let dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).lastPathComponent
        guard let head = try? String(contentsOfFile: ".git/HEAD", encoding: .utf8),
              let r = head.range(of: "refs/heads/") else { return dir }
        let branch = head[r.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
        return branch.isEmpty ? dir : "\(dir):\(branch)"
    }()

    private let raw = RawTerminal()
    private let reader = KeyReader()

    init(engine: InferenceEngine, modelName: String, system: String?,
         params: SamplingParams, maxTokens: Int, registry: Registry,
         initialImage: Data?, initialAudio: Data?) {
        self.engine = engine
        self.modelName = modelName
        self.system = system
        self.params = params
        self.maxTokens = maxTokens
        self.registry = registry
        self.contextWindow = AliasMap.resolve(modelName)?.context ?? 0
        if let initialImage { pendingImages.append(makeAtt(.image, initialImage, "image")) }
        if let initialAudio { pendingAudio = makeAtt(.audio, initialAudio, "audio") }
    }

    // MARK: - Run loop

    func run() async {
        raw.enter()
        installWinch()
        updateSize()
        defer { raw.leave() }

        if !pendingImages.isEmpty || pendingAudio != nil {
            view.append(Msg(role: .note, text: attachSummary()))
        }
        render()

        while !shouldQuit {
            if tuiWinchFlag != 0 { tuiWinchFlag = 0; updateSize(); render() }
            guard raw.waitForInput(timeoutMs: 250) else { continue }
            let keys = reader.read()
            if keys.isEmpty { break }   // EOF
            var submit: String?
            for key in keys {
                if let text = handleKey(key) { submit = text }
                if shouldQuit { break }
            }
            render()
            if let text = submit, !shouldQuit { await processSubmit(text) }
        }
    }

    // MARK: - Key handling

    /// Mutate UI state for a key; return non-nil text to submit on Enter.
    private func handleKey(_ key: Key) -> String? {
        switch key {
        case .ctrlD:
            if input.isEmpty { shouldQuit = true }
            return nil
        case .ctrlC:
            if input.isEmpty { shouldQuit = true } else { input = ""; cursor = 0; menu.close() }
            return nil
        case .ctrlU:
            input = ""; cursor = 0; menu.update(for: input); return nil
        case .ctrlL:
            return nil   // render() runs after each batch
        case .ctrlA, .home: cursor = 0; return nil
        case .ctrlE, .end: cursor = input.count; return nil
        case .pageUp: scrollOffset += max(1, paneHeight() - 1); return nil
        case .pageDown: scrollOffset = max(0, scrollOffset - max(1, paneHeight() - 1)); return nil
        case .escape: menu.close(); return nil
        case .left: if cursor > 0 { cursor -= 1 }; return nil
        case .right: if cursor < input.count { cursor += 1 }; return nil
        case .up:
            if menu.isActive { menu.selectPrevious() } else { recallHistory(delta: -1) }
            return nil
        case .down:
            if menu.isActive { menu.selectNext() } else { recallHistory(delta: 1) }
            return nil
        case .tab:
            if let item = menu.current { setInput(item.name + " "); menu.close() }
            return nil
        case .backspace:
            if cursor > 0 {
                let idx = input.index(input.startIndex, offsetBy: cursor - 1)
                input.remove(at: idx); cursor -= 1; menu.update(for: input)
            }
            return nil
        case .delete:
            if cursor < input.count {
                let idx = input.index(input.startIndex, offsetBy: cursor)
                input.remove(at: idx); menu.update(for: input)
            }
            return nil
        case .enter:
            // With the popup open, Enter runs the highlighted command directly.
            if let item = menu.current, input != item.name {
                let cmd = item.name
                input = ""; cursor = 0; menu.close()
                return cmd
            }
            let text = input
            input = ""; cursor = 0; menu.close(); scrollOffset = 0
            return text.isEmpty ? nil : text
        case .char(let c):
            let idx = input.index(input.startIndex, offsetBy: cursor)
            input.insert(c, at: idx); cursor += 1; menu.update(for: input)
            return nil
        default:
            return nil
        }
    }

    private func setInput(_ s: String) { input = s; cursor = s.count; menu.update(for: input) }

    private func recallHistory(delta: Int) {
        guard !inputHistory.isEmpty else { return }
        historyIndex = max(0, min(inputHistory.count, historyIndex + delta))
        setInput(historyIndex < inputHistory.count ? inputHistory[historyIndex] : "")
        menu.close()
    }

    // MARK: - Submit / commands

    private func processSubmit(_ text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        inputHistory.append(text); historyIndex = inputHistory.count

        if trimmed.hasPrefix("/"), isCommand(trimmed) {
            await handleCommand(trimmed)
            render()
            return
        }
        // Inline @path and bare-path attachments.
        let (cleaned, atts) = extractInline(trimmed)
        pendingImages.append(contentsOf: atts.filter { $0.kind == .image })
        if let a = atts.last(where: { $0.kind == .audio }) { pendingAudio = a }
        let prompt = cleaned.trimmingCharacters(in: .whitespaces)
        if prompt.isEmpty {
            if !atts.isEmpty { view.append(Msg(role: .note, text: attachSummary())); render() }
            return
        }
        await generate(prompt: prompt)
    }

    /// Aliases accepted by handleCommand but kept out of the autosuggest list to
    /// avoid cluttering it.
    private static let commandAliases: Set<String> = ["/img", "/exit", "/q"]

    private func isCommand(_ s: String) -> Bool {
        let first = String(s.split(separator: " ").first ?? "")
        return SlashMenu.all.contains { $0.name == first } || Self.commandAliases.contains(first)
    }

    private func handleCommand(_ s: String) async {
        let parts = s.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
        let cmd = String(parts.first ?? "")
        let arg = parts.count > 1 ? String(parts[1]).trimmingCharacters(in: .whitespaces) : ""
        switch cmd {
        case "/quit", "/exit", "/q": shouldQuit = true
        case "/help": view.append(Msg(role: .note, text: helpText()))
        case "/clear": pendingImages.removeAll(); pendingAudio = nil; note("Cleared pending attachments.")
        case "/reset":
            modelTurns.removeAll(); view.removeAll(); pendingImages.removeAll(); pendingAudio = nil
            note("Conversation reset.")
        case "/history":
            note(modelTurns.isEmpty ? "No conversation yet."
                 : modelTurns.map { "\($0.role): \($0.content)" }.joined(separator: "\n"))
        case "/system":
            if arg.isEmpty { note(system.map { "System: \($0)" } ?? "No system prompt. Usage: /system <text>") }
            else { system = arg; note("System prompt updated.") }
        case "/model":
            if arg.isEmpty { note("Current model: \(modelName)") } else { await switchModel(arg) }
        case "/save": saveTranscript(arg)
        case "/attach": note(attachSummary())
        case "/remove": removeAttachment(arg)
        case "/image", "/img", "/audio":
            if arg.isEmpty { note("Usage: \(cmd) <path>") }
            else if attach(path: arg) { note(attachSummary()) }
        case "/mic": await recordMic()
        default: note("Unknown command \(cmd).")
        }
    }

    private func note(_ s: String) { view.append(Msg(role: .note, text: s)) }

    // MARK: - Generation

    private func generate(prompt: String) async {
        view.append(Msg(role: .user, text: prompt))
        modelTurns.append((role: "user", content: prompt))
        let usedImgs = pendingImages.count
        let usedAud = pendingAudio != nil

        var messages: [[String: String]] = []
        if let system, !system.isEmpty { messages.append(["role": "system", "content": system]) }
        for t in modelTurns { messages.append(["role": t.role, "content": t.content]) }
        let imgs = pendingImages.map(\.data)

        view.append(Msg(role: .assistant, text: ""))
        let aIdx = view.count - 1
        scrollOffset = 0
        lastStatus = "thinking..."
        render()

        let gen = engine.generate(
            messages: messages, params: params, maxTokens: maxTokens,
            imageData: imgs.first, audioData: pendingAudio?.data, imagesData: imgs)
        var stream: AsyncStream<TokenEvent>? = gen.stream
        let filter = StreamingReasoningFilter()
        var assistant = ""
        var cancelled = false

        for await event in stream! {
            if event.isEnd { break }
            let visible = filter.consume(event.text)
            if !visible.isEmpty {
                assistant += visible
                view[aIdx].text = assistant
                render()
            }
            // Watch for Ctrl-C (raw mode delivers it as a byte) and resize.
            if raw.waitForInput(timeoutMs: 0) {
                let keys = reader.read()
                if keys.contains(.ctrlC) { cancelled = true; break }
                if keys.contains(.pageUp) { scrollOffset += paneHeight() - 1; render() }
                if keys.contains(.pageDown) { scrollOffset = max(0, scrollOffset - (paneHeight() - 1)); render() }
            }
            if tuiWinchFlag != 0 { tuiWinchFlag = 0; updateSize() }
        }
        stream = nil   // drop -> onTermination cancels the engine if we broke early
        let tail = filter.finish()
        if !tail.isEmpty { assistant += tail }
        view[aIdx].text = assistant

        if cancelled {
            view.append(Msg(role: .note, text: "(cancelled)"))
        } else {
            modelTurns.append((role: "assistant", content: assistant))
            if let st = gen.stats() { lastStatus = statusText(st, images: usedImgs, audio: usedAud) }
        }
        pendingImages.removeAll(); pendingAudio = nil
        render()
    }

    private func statusText(_ st: GenerationStats, images: Int, audio: Bool) -> String {
        var parts = [modelName]
        if images > 0 { parts.append("\(images) img") }
        if audio { parts.append("audio") }
        parts.append(String(format: "%.0f tok/s", st.decodeTokensPerSecond))
        let ctx = st.promptTokens + st.generatedTokens
        if contextWindow > 0 {
            let pct = min(100, Int((Double(ctx) / Double(contextWindow)) * 100.0))
            parts.append("ctx \(ctx) (\(pct)%)")
        } else {
            parts.append("ctx \(ctx)")
        }
        return parts.joined(separator: " \u{00B7} ")
    }

    private func switchModel(_ name: String) async {
        let dir = registry.hasModel(name) ? registry.modelPath(name) : URL(fileURLWithPath: name)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            note("Model not found: \(name). Install with: krillm pull \(name)"); return
        }
        note("Loading \(name)...")
        render()
        let newEngine = InferenceEngine(modelDirectory: dir)
        do {
            try await newEngine.load()
            engine = newEngine; modelName = name
            contextWindow = AliasMap.resolve(name)?.context ?? 0
            note("Switched to \(name). Conversation kept.")
        } catch { note("Failed to load \(name): \(error)") }
    }

    private func recordMic() async {
        guard engine.canUseNativeAudio else { note("This model cannot process audio; /mic is unavailable."); return }
        guard await MicrophoneRecorder.requestAccess() else {
            note(MicrophoneCaptureError.permissionDenied.description); return
        }
        let rec = MicrophoneRecorder()
        do { try rec.start() } catch { note("\(error)"); return }
        note("Recording... press Enter to stop."); render()
        while true {
            guard raw.waitForInput(timeoutMs: 100) else { continue }
            if reader.read().contains(where: { $0 == .enter }) { break }
        }
        do {
            let wav = try rec.stop()
            pendingAudio = makeAtt(.audio, wav, "microphone")
            note(String(format: "Captured %.1fs of audio.", rec.capturedSeconds))
            note(attachSummary())
        } catch { note("\(error)") }
    }

    // MARK: - Rendering

    // Rows consumed by chrome: 2 masthead (line + rule) + 3 input box + 1 footer.
    private let paneTop = 3
    private func paneHeight() -> Int { max(1, rows - 6) }

    private func render() {
        let width = cols
        var frame = "\u{1B}[H"   // cursor home

        // Rows 1-2: light masthead (wordmark line + dim rule).
        frame += positioned(1, Brand.header(width: width, model: modelName))
        frame += positioned(2, Brand.headerRule(width: width))

        // Layout from the bottom up: the 3-row input box, then the footer below
        // it, with the conversation pane filling everything between the masthead
        // and the box. `boxTop` is floored below the masthead so the box never
        // overwrites it or emits an invalid CSI on tiny terminals; the footer is
        // pushed below the box (off-screen, harmlessly, if the terminal is short).
        let box = inputBox(width: width)             // 3 lines: top / field / bottom
        let boxTop = max(paneTop + 1, rows - box.count)
        let footerRow = max(rows, boxTop + box.count)
        let pane = paneLines(width: width)
        let menuLines = menu.isActive ? renderMenu(width: width) : []
        let availRows = max(0, boxTop - paneTop)     // rows paneTop .. boxTop-1
        let convHeight = max(0, availRows - menuLines.count)

        // Bottom-anchor the conversation against the input box (new messages
        // appear where you type); the splash stays vertically centered.
        let blankTop = Chrome.anchorBlankTop(paneCount: pane.count, convHeight: convHeight, centered: view.isEmpty)
        let maxStart = max(0, pane.count - convHeight)
        scrollOffset = min(scrollOffset, maxStart)   // clamp: cannot scroll past the top
        let start = max(0, maxStart - scrollOffset)
        for i in 0..<convHeight {
            let content: String
            if i < blankTop {
                content = ""
            } else {
                let lineIdx = start + (i - blankTop)
                content = lineIdx < pane.count ? pane[lineIdx] : ""
            }
            frame += positioned(paneTop + i, content)
        }
        // Slash popup sits just above the input box.
        for (i, line) in menuLines.enumerated() {
            frame += positioned(paneTop + convHeight + i, line)
        }
        // Input box, then footer.
        for (i, line) in box.enumerated() {
            frame += positioned(boxTop + i, line)
        }
        let status = lastStatus.isEmpty ? "ready" : lastStatus
        let right = "\(cwdLabel) \u{00B7} \(KrillLMVersionTag)"
        frame += positioned(footerRow, Brand.footer(width: width, left: status, right: right))
        frame += "\u{1B}[J"   // clear anything below
        Output.write(frame)
    }

    private func positioned(_ row: Int, _ content: String) -> String {
        "\u{1B}[\(row);1H\u{1B}[2K" + content
    }

    /// Conversation pane lines. Turns are distinguished by SHADE, not by role
    /// labels: the user's own words read bright white; the model's reply reads
    /// dim gray so the user's turn clearly stands out. A faint rule opens each
    /// new exchange (before every user turn but the first) so the back-and-forth
    /// groups visually without any "you"/"krilllm" name tags. Vertical placement
    /// (bottom-anchor vs. centered splash) is handled by `render`.
    private func paneLines(width: Int) -> [String] {
        guard !view.isEmpty else { return Brand.splash(width: width) }
        let margin = "  "
        let w = max(10, width - 4)                       // 2-space margin both sides
        let rule = Ansi.dim(margin + String(repeating: "\u{2500}", count: min(w, 48)))
        var lines: [String] = []
        var sawTurn = false
        for msg in view {
            switch msg.role {
            case .user:
                if sawTurn { lines.append(rule); lines.append("") }
                sawTurn = true
                for l in Layout.wrap(msg.text, width: w) { lines.append(margin + Ansi.white(l)) }
            case .assistant:
                sawTurn = true
                lines.append("")
                for l in TUIMarkdown.render(msg.text, width: w) { lines.append(margin + Ansi.dimStyled(l)) }
            case .note:
                for l in Layout.wrap(msg.text, width: w) { lines.append(margin + Ansi.dim(l)) }
            }
            lines.append("")
        }
        return lines
    }

    private func renderMenu(width: Int) -> [String] {
        let maxVisible = 8
        let total = menu.matches.count
        // Window the visible items around the selection so cycling past the
        // window keeps the highlighted command on screen.
        var winStart = 0
        if total > maxVisible {
            winStart = min(max(0, menu.selected - maxVisible / 2), total - maxVisible)
        }
        let winEnd = min(winStart + maxVisible, total)
        var out: [String] = []
        if winStart > 0 { out.append(Ansi.dim("    ...")) }
        for i in winStart..<winEnd {
            let item = menu.matches[i]
            let marker = i == menu.selected ? ">" : " "
            let name = item.name.padding(toLength: 9, withPad: " ", startingAt: 0)
            let label = "  \(marker) \(name)  \(item.summary)"
            let clipped = String(label.prefix(width))
            out.append(i == menu.selected ? Ansi.inverse(clipped) : Ansi.dim(clipped))
        }
        if winEnd < total { out.append(Ansi.dim("    ...")) }
        return out
    }

    /// The bottom input as a rounded, padded 3-row box (top border, field,
    /// bottom border). The field shows a "> " prompt, the typed text with a fake
    /// inverse-video block cursor (the real cursor stays hidden), and a dim
    /// placeholder when empty. Long input scrolls horizontally so the frame is
    /// never broken.
    private func inputBox(width: Int) -> [String] {
        let w = max(8, width)
        let h = "\u{2500}", v = "\u{2502}"     // light horizontal / vertical
        let tl = "\u{256D}", tr = "\u{256E}"   // rounded corners
        let bl = "\u{2570}", br = "\u{256F}"
        let top = Ansi.dim(Chrome.border(width: w, left: tl, fill: h, right: tr))
        let bottom = Ansi.dim(Chrome.border(width: w, left: bl, fill: h, right: br))

        let fieldWidth = max(2, w - 4)         // inner span minus one pad space each side
        let promptStr = "> "
        let textWidth = max(1, fieldWidth - promptStr.count)
        let chars = Array(input)

        var body: String
        if chars.isEmpty {
            // Block cursor then a dim placeholder, clipped/padded to the field.
            let placeholder = "type a message   /help for commands"
            let clipped = String(placeholder.prefix(max(0, textWidth - 1)))
            let pad = String(repeating: " ", count: max(0, textWidth - 1 - clipped.count))
            body = Ansi.bold(promptStr) + Ansi.inverse(" ") + Ansi.dim(clipped) + pad
        } else {
            // Pure geometry windows the text and locates the cursor; we only
            // apply the inverse-video block cursor to that one cell.
            let (content, cursorCol) = Chrome.inputField(text: chars, cursor: cursor, textWidth: textWidth)
            let cs = Array(content)
            var rendered = ""
            for (i, ch) in cs.enumerated() {
                rendered += i == cursorCol ? Ansi.inverse(String(ch)) : String(ch)
            }
            body = Ansi.bold(promptStr) + rendered
        }
        let field = Ansi.dim(v) + " " + body + " " + Ansi.dim(v)
        return [top, field, bottom]
    }

    // MARK: - Size / signals

    private func updateSize() { let s = raw.size(); rows = s.rows; cols = s.cols }

    private func installWinch() {
        var unblock = sigset_t(); sigemptyset(&unblock); sigaddset(&unblock, SIGWINCH)
        pthread_sigmask(SIG_UNBLOCK, &unblock, nil)
        signal(SIGWINCH, tuiWinchHandler)
    }

    // MARK: - Attachments / media

    private func makeAtt(_ kind: MediaKind, _ data: Data, _ name: String) -> Att {
        Att(kind: kind, data: data, name: name, dims: kind == .image ? MediaAttachment.imageDimensions(data) : nil)
    }

    private func attachSummary() -> String {
        let total = pendingImages.count + (pendingAudio != nil ? 1 : 0)
        guard total > 0 else { return "No attachments pending." }
        var lines = ["Pending attachments (sent with your next message):"]
        var i = 1
        for img in pendingImages {
            let dim = img.dims.map { " \($0.0)x\($0.1)" } ?? ""
            lines.append("  [\(i)] image  \(img.name)\(dim)"); i += 1
        }
        if let a = pendingAudio { lines.append("  [\(i)] audio  \(a.name)") }
        lines.append("  /remove <n> to drop one, /clear to drop all.")
        return lines.joined(separator: "\n")
    }

    private func removeAttachment(_ arg: String) {
        guard let n = Int(arg) else { note("Usage: /remove <number>"); return }
        let total = pendingImages.count + (pendingAudio != nil ? 1 : 0)
        guard n >= 1, n <= total else { note("No attachment \(n)."); return }
        if n <= pendingImages.count { let r = pendingImages.remove(at: n - 1); note("Removed \(r.name).") }
        else { note("Removed \(pendingAudio?.name ?? "audio")."); pendingAudio = nil }
        note(attachSummary())
    }

    private enum Load { case notFound, notMedia, unsupported(MediaKind), ok(MediaKind, Data) }

    private func loadMedia(_ token: String) -> Load {
        let path = MediaAttachment.normalizePath(token)
        guard !path.isEmpty else { return .notFound }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), !isDir.boolValue,
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return .notFound }
        let ext = (path as NSString).pathExtension
        guard let kind = MediaAttachment.detectKind(data: data, pathExtension: ext) else { return .notMedia }
        switch kind {
        case .image where !engine.supportsNativeImage: return .unsupported(.image)
        case .audio where !engine.canUseNativeAudio: return .unsupported(.audio)
        default: return .ok(kind, data)
        }
    }

    private func attach(path: String) -> Bool {
        switch loadMedia(path) {
        case .notFound: note("File not found: \(path)"); return false
        case .notMedia: note("Not a recognized image or audio file: \(path)"); return false
        case .unsupported(let k): note("This model cannot process \(k.rawValue) input."); return false
        case .ok(let kind, let data):
            let name = (MediaAttachment.normalizePath(path) as NSString).lastPathComponent
            if kind == .image { pendingImages.append(makeAtt(.image, data, name)) }
            else { pendingAudio = makeAtt(.audio, data, name) }
            return true
        }
    }

    private func extractInline(_ line: String) -> (String, [Att]) {
        var cleaned = "", atts: [Att] = []
        let chars = Array(line); var i = 0; var atBoundary = true
        while i < chars.count {
            let ch = chars[i]
            if ch == "@", atBoundary {
                var j = i + 1, tok = ""
                while j < chars.count {
                    let c = chars[j]
                    if c == "\\", j + 1 < chars.count { tok.append(c); tok.append(chars[j + 1]); j += 2; continue }
                    if c == " " || c == "\t" { break }
                    tok.append(c); j += 1
                }
                if !tok.isEmpty, case .ok(let kind, let data) = loadMedia(tok) {
                    atts.append(makeAtt(kind, data, (MediaAttachment.normalizePath(tok) as NSString).lastPathComponent))
                    i = j; atBoundary = false; continue
                }
            }
            cleaned.append(ch); atBoundary = (ch == " " || ch == "\t"); i += 1
        }
        return (cleaned, atts)
    }

    // MARK: - Help / transcript

    private func helpText() -> String {
        """
        Keys: Up/Down history (or cycle slash menu) \u{00B7} Tab accept \u{00B7} Enter send
              PgUp/PgDn scroll \u{00B7} Ctrl-C cancel reply / clear \u{00B7} Ctrl-D quit
        Commands: /image /audio /mic /attach /remove /clear /system /model
                  /history /save /reset /help /quit
        Attach by dragging a file in or typing @path in your message.
        """
    }

    private func saveTranscript(_ arg: String) {
        let path = arg.isEmpty ? "krillm-transcript.txt" : MediaAttachment.normalizePath(arg)
        var text = ""
        if let system, !system.isEmpty { text += "system: \(system)\n\n" }
        for t in modelTurns { text += "\(t.role): \(t.content)\n\n" }
        do { try text.write(toFile: path, atomically: true, encoding: .utf8); note("Saved to \(path)") }
        catch { note("Could not save: \(error)") }
    }
}
