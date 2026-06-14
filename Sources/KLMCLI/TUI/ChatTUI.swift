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
    private var shouldQuit = false

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

    private func isCommand(_ s: String) -> Bool {
        let first = String(s.split(separator: " ").first ?? "")
        return SlashMenu.all.contains { $0.name == first }
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
        parts.append("ctx \(st.promptTokens + st.generatedTokens)")
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

    private func paneHeight() -> Int { max(1, rows - 3) }   // header + input + footer

    private func render() {
        let width = cols
        var frame = "\u{1B}[H"   // cursor home

        // Row 1: masthead.
        frame += positioned(1, Brand.header(width: width, model: modelName))

        // Conversation pane (rows 2 .. rows-2).
        let pane = paneLines(width: width)
        let ph = paneHeight()
        let menuLines = menu.isActive ? renderMenu(width: width) : []
        let convHeight = max(0, ph - menuLines.count)
        let maxStart = max(0, pane.count - convHeight)
        let start = max(0, maxStart - scrollOffset)
        for i in 0..<convHeight {
            let row = 2 + i
            let lineIdx = start + i
            let content = lineIdx < pane.count ? pane[lineIdx] : ""
            frame += positioned(row, content)
        }
        // Slash popup sits just above the input line.
        for (i, line) in menuLines.enumerated() {
            frame += positioned(2 + convHeight + i, line)
        }

        // Input line (row rows-1) and footer (row rows).
        frame += positioned(rows - 1, inputLine(width: width))
        frame += positioned(rows, Brand.footer(width: width, status: lastStatus.isEmpty ? "ready" : lastStatus))
        frame += "\u{1B}[J"   // clear anything below
        Output.write(frame)
    }

    private func positioned(_ row: Int, _ content: String) -> String {
        "\u{1B}[\(row);1H\u{1B}[2K" + content
    }

    private func paneLines(width: Int) -> [String] {
        guard !view.isEmpty else { return verticallyCentered(Brand.splash(width: width), in: paneHeight()) }
        let w = max(10, width - 2)
        var lines: [String] = []
        for msg in view {
            switch msg.role {
            case .user:
                lines.append(Ansi.green("you"))
                for l in Layout.wrap(msg.text, width: w) { lines.append("  " + l) }
            case .assistant:
                lines.append(Ansi.cyan(Brand.product.lowercased()))
                for l in TUIMarkdown.render(msg.text, width: w) { lines.append("  " + l) }
            case .note:
                for l in Layout.wrap(msg.text, width: width) { lines.append(Ansi.dim(l)) }
            }
            lines.append("")
        }
        return lines
    }

    private func verticallyCentered(_ block: [String], in height: Int) -> [String] {
        guard block.count < height else { return block }
        let top = (height - block.count) / 2
        return Array(repeating: "", count: top) + block
    }

    private func renderMenu(width: Int) -> [String] {
        var out: [String] = []
        let shown = Array(menu.matches.prefix(8))
        for (i, item) in shown.enumerated() {
            let marker = i == menu.selected ? ">" : " "
            let label = String(format: "  %@ %-9@ %@", marker, item.name, item.summary)
            let clipped = String(label.prefix(width))
            out.append(i == menu.selected ? Ansi.inverse(clipped) : Ansi.dim(clipped))
        }
        return out
    }

    private func inputLine(width: Int) -> String {
        let prompt = Ansi.bold("> ")
        // Fake block cursor (we keep the real cursor hidden).
        let chars = Array(input)
        var rendered = ""
        for (i, c) in chars.enumerated() {
            rendered += i == cursor ? Ansi.inverse(String(c)) : String(c)
        }
        if cursor >= chars.count { rendered += Ansi.inverse(" ") }
        return prompt + rendered
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
