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
    private struct Msg { enum Role { case user, assistant, note, pre }; let role: Role; var text: String }
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
    private let customCommands: CustomCommandStore
    private var picker: ModelPicker?          // active modal model picker, if any
    private var pendingModelLoad: String?     // model the picker chose; loaded by the run loop
    private var pendingVoiceCapture = false   // Space-to-talk armed; run loop records
    // What hold-Space does with the recorded clip. `.dictate` (default)
    // transcribes it into the composer for review - on-device via Apple Speech
    // when available, else the multimodal model's best effort. `.send` sends it
    // as an audio turn the model answers. Toggle with /voice.
    private enum VoiceMode { case dictate, send }
    private var voiceMode: VoiceMode = .dictate
    // Which speech-to-text engine dictation uses. `.apple` (default) is Apple's
    // on-device recognizer (no download). `.whisper` is KrillLM's native MLX
    // Whisper runtime (best accuracy; downloads an English model on first use).
    private enum VoiceEngine { case apple, whisper }
    private var voiceEngine: VoiceEngine = .apple
    private let speech = SpeechRecognizer()
    // Lazily loaded native Whisper runtime + its SKU (English dictation).
    private var whisper: WhisperRuntime?
    private var whisperSKU = WhisperModelManager.defaultSKU
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

    private let themeOverride: String?

    init(engine: InferenceEngine, modelName: String, system: String?,
         params: SamplingParams, maxTokens: Int, registry: Registry,
         initialImage: Data?, initialAudio: Data?, theme: String? = nil) {
        self.engine = engine
        self.modelName = modelName
        self.system = system
        self.params = params
        self.maxTokens = maxTokens
        self.registry = registry
        self.themeOverride = theme
        self.contextWindow = AliasMap.resolve(modelName)?.context ?? 0
        self.customCommands = CustomCommandStore.load(from: Self.commandsDir)
        menu.extra = customCommands.commands.map {
            SlashMenu.Item(name: "/\($0.name)", summary: $0.description)
        }
        if let initialImage { pendingImages.append(makeAtt(.image, initialImage, "image")) }
        if let initialAudio { pendingAudio = makeAtt(.audio, initialAudio, "audio") }
    }

    /// Where user-authored slash commands live: `~/.krillm/commands/<name>.md`.
    private static var commandsDir: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".krillm").appendingPathComponent("commands")
    }

    /// Pick the shade palette for the terminal background so turns stay readable
    /// on light AND dark terminals. Order: explicit `--theme` / `KRILL_TUI_THEME`
    /// override, then `COLORFGBG`, then a best-effort OSC 11 query, else the
    /// always-safe palette. Runs once, after raw mode is on (OSC 11 needs it).
    private func resolveTheme() {
        let env = ProcessInfo.processInfo.environment
        let override = themeOverride ?? env["KRILL_TUI_THEME"]
        var bg = Theme.resolve(override: override, colorFGBG: env["COLORFGBG"])
        if bg == .unknown, override == nil || override?.lowercased() == "auto" {
            if let lum = raw.queryBackgroundLuminance() { bg = Theme.background(forLuminance: lum) }
        }
        Ansi.theme = Theme.palette(for: bg)
    }

    // MARK: - Run loop

    func run() async {
        raw.enter()
        resolveTheme()
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
            guard let keys = reader.read() else { break }   // nil == EOF
            // An empty (non-nil) batch is a stray mouse/terminal event - ignore it.
            var submit: String?
            for key in keys {
                if let text = handleKey(key) { submit = text }
                if shouldQuit { break }
            }
            render()
            if let text = submit, !shouldQuit { await processSubmit(text) }
            // The picker chose a model: load (or download then load) it now.
            if let name = pendingModelLoad, !shouldQuit {
                pendingModelLoad = nil
                await switchOrDownload(name)
                render()
            }
            // Push-to-talk: record while Space is held, send on release.
            if pendingVoiceCapture, !shouldQuit {
                pendingVoiceCapture = false
                await holdToTalk()
                render()
            }
        }
    }

    // MARK: - Key handling

    /// Mutate UI state for a key; return non-nil text to submit on Enter.
    private func handleKey(_ key: Key) -> String? {
        // The model picker is modal: while open it owns the keyboard.
        if picker != nil { handlePickerKey(key); return nil }
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
        case .scrollUp: scrollOffset += 3; return nil
        case .scrollDown: scrollOffset = max(0, scrollOffset - 3); return nil
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
            // On a voice-capable model, Space on an empty composer is push-to-talk
            // (hold to talk, release to send) rather than a typed space.
            if c == " " && input.isEmpty && engine.canUseNativeAudio {
                pendingVoiceCapture = true
                return nil
            }
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

    // MARK: - Model picker

    /// Key handling while the modal model picker is open: Up/Down to move, Enter
    /// to choose (the run loop then loads or downloads it), Esc/Ctrl-C to cancel.
    private func handlePickerKey(_ key: Key) {
        switch key {
        case .up, .scrollUp: picker?.selectPrevious()
        case .down, .scrollDown: picker?.selectNext()
        case .enter:
            if let chosen = picker?.current?.name { pendingModelLoad = chosen }
            picker = nil
        case .escape, .ctrlC: picker = nil
        default: break
        }
    }

    /// Build the picker entries: every chat-capable built-in alias plus any
    /// catalog models, each flagged as downloaded. Embedding/reranker models are
    /// excluded - you cannot chat with them. Downloaded models sort first.
    private func modelPickerEntries() -> [ModelPicker.Entry] {
        var seen = Set<String>()
        var entries: [ModelPicker.Entry] = []
        func add(_ name: String, _ params: String, _ quant: String, _ family: ModelFamily) {
            guard !seen.contains(name) else { return }
            guard ModelCapabilities.capabilities(for: family).contains(.textGeneration) else { return }
            seen.insert(name)
            let downloaded = registry.hasModel(name)
            // Real on-disk size when installed (the manifest's sizeBytes is often
            // 0, so sum the files); otherwise an estimate from the parameter count
            // and quantization so the download cost is visible up front.
            let size: String
            let onDisk = downloaded ? directorySize(registry.modelPath(name)) : 0
            if onDisk > 0 {
                size = formatSize(onDisk)
            } else if let est = estimatedSizeBytes(params: params, quant: quant) {
                size = "~" + formatSize(est)
            } else {
                size = ""
            }
            let detail = size.isEmpty ? "\(params) \u{00B7} \(quant)"
                                      : "\(params) \u{00B7} \(quant) \u{00B7} \(size)"
            entries.append(.init(name: name, detail: detail, downloaded: downloaded))
        }
        for (_, m) in AliasMap.allAliases { add(m.name, m.params, m.quant, m.family) }
        if let catalog = ModelCatalogStore(baseDir: registry.baseDir).load() {
            for e in catalog.models { add(e.alias, e.params, e.quant, e.family) }
        }
        return entries.sorted { a, b in
            a.downloaded != b.downloaded ? a.downloaded : a.name < b.name
        }
    }

    /// Total size of the regular files under `url` (the model's on-disk weight).
    private func directorySize(_ url: URL) -> Int64 {
        let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey]
        guard let en = FileManager.default.enumerator(at: url, includingPropertiesForKeys: keys) else { return 0 }
        var total: Int64 = 0
        for case let f as URL in en {
            if let v = try? f.resourceValues(forKeys: Set(keys)), v.isRegularFile == true {
                total += Int64(v.fileSize ?? 0)
            }
        }
        return total
    }

    /// Human-readable byte size (decimal units, matching how disk size is shown).
    private func formatSize(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1_000_000_000
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        return String(format: "%.0f MB", Double(bytes) / 1_000_000)
    }

    /// Rough download size from the parameter count and quantization, for models
    /// not yet on disk (so the picker can show "~6 GB" before you pull). Returns
    /// nil if the parameter string has no number to work with.
    private func estimatedSizeBytes(params: String, quant: String) -> Int64? {
        let p = params.lowercased()
        let nums = p.split { !"0123456789.".contains($0) }.compactMap { Double($0) }
        guard !nums.isEmpty else { return nil }
        // "8x7B" multiplies (mixture of experts); a range like "1B-7B" takes the max.
        let billions = p.contains("x") ? nums.reduce(1, *) : (nums.max() ?? nums[0])
        let q = quant.lowercased()
        let bytesPerParam: Double
        if q.contains("32") { bytesPerParam = 4.2 }
        else if q.contains("16") { bytesPerParam = 2.1 }
        else if q.contains("8bit") { bytesPerParam = 1.1 }
        else { bytesPerParam = 0.6 }      // 4-bit family (incl. nvfp4/mxfp4)
        return Int64(billions * 1_000_000_000 * bytesPerParam)
    }

    /// Switch to `name` if it is already downloaded, otherwise pull it (with a
    /// live progress line) and then switch.
    private func switchOrDownload(_ name: String) async {
        if registry.hasModel(name) { await switchModel(name); return }
        let store = ModelCatalogStore(baseDir: registry.baseDir)
        guard let resolved = AliasMap.resolve(name, catalog: store) else {
            note("Unknown model: \(name)"); return
        }
        note("Downloading \(name) ...")
        render()
        // The progress closure is @Sendable, so capture only value types (no
        // self) and write the progress line straight to the footer row.
        let footerRow = rows
        let width = cols
        let label = name
        let puller = Puller(registry: registry)
        do {
            _ = try await puller.pull(resolved) { done, total, file in
                guard file != "done" else { return }
                let pct = total > 0 ? Int(Double(done) / Double(total) * 100.0) : 0
                let line = "  Downloading \(label): \(pct)%  \(file)"
                Output.write("\u{1B}[\(footerRow);1H\u{1B}[2K" + Ansi.chrome(String(line.prefix(max(0, width)))))
            }
            note("Downloaded \(name).")
            await switchModel(name)
        } catch {
            note("Download failed for \(name): \(error)")
        }
    }

    /// Render the modal picker list: downloaded models read bright (switch
    /// instantly), not-yet-downloaded ones read dim with a "will download" hint;
    /// the highlighted row is inverse-video. A filled dot marks downloaded, a
    /// hollow dot not-downloaded.
    private func renderPicker(_ p: ModelPicker, width: Int, height: Int) -> [String] {
        let margin = "  "
        var out: [String] = [
            margin + Ansi.bold("Select a model"),
            margin + Ansi.chrome("downloaded switch instantly \u{00B7} faded ones download first"),
            "",
        ]
        let maxVisible = max(3, height - 5)
        let total = p.entries.count
        var winStart = 0
        if total > maxVisible { winStart = min(max(0, p.selected - maxVisible / 2), total - maxVisible) }
        let winEnd = min(winStart + maxVisible, total)
        let nameW = min(20, p.entries.map { $0.name.count }.max() ?? 12)
        if winStart > 0 { out.append(margin + Ansi.chrome("  ...")) }
        for i in winStart..<winEnd {
            let e = p.entries[i]
            let dot = e.downloaded ? "\u{25CF}" : "\u{25CB}"
            let name = e.name.padding(toLength: nameW, withPad: " ", startingAt: 0)
            let hint = e.downloaded ? "" : "   will download"
            let body = "\(dot) \(name)  \(e.detail)\(hint)"
            let line = margin + "  " + String(body.prefix(max(0, width - 4)))
            if i == p.selected { out.append(Ansi.inverse(line)) }
            else if e.downloaded { out.append(Ansi.user(line)) }
            else { out.append(Ansi.chrome(line)) }
        }
        if winEnd < total { out.append(margin + Ansi.chrome("  ...")) }
        out.append("")
        out.append(margin + Ansi.chrome("Up/Down select \u{00B7} Enter switch \u{00B7} Esc cancel"))
        return out
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
    private static let commandAliases: Set<String> = ["/img", "/exit", "/q", "/reset"]

    private func isCommand(_ s: String) -> Bool {
        let first = String(s.split(separator: " ").first ?? "")
        return SlashMenu.all.contains { $0.name == first }
            || Self.commandAliases.contains(first)
            || customCommands.command(named: first) != nil
    }

    private func handleCommand(_ s: String) async {
        let parts = s.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
        let cmd = String(parts.first ?? "")
        let arg = parts.count > 1 ? String(parts[1]).trimmingCharacters(in: .whitespaces) : ""
        switch cmd {
        case "/quit", "/exit", "/q": shouldQuit = true
        case "/help": view.append(Msg(role: .pre, text: helpText()))
        case "/drop": pendingImages.removeAll(); pendingAudio = nil; note("Dropped pending attachments.")
        case "/clear", "/reset":   // /reset kept as an alias for clearing the chat
            modelTurns.removeAll(); view.removeAll(); pendingImages.removeAll(); pendingAudio = nil
            note("Conversation cleared.")
        case "/history":
            note(modelTurns.isEmpty ? "No conversation yet."
                 : modelTurns.map { "\($0.role): \($0.content)" }.joined(separator: "\n"))
        case "/compact": await compactConversation()
        case "/system":
            if arg.isEmpty { note(system.map { "System: \($0)" } ?? "No system prompt. Usage: /system <text>") }
            else { system = arg; note("System prompt updated.") }
        case "/model":
            if arg.isEmpty { picker = ModelPicker(entries: modelPickerEntries(), current: modelName) }
            else { await switchOrDownload(arg) }
        case "/save": saveTranscript(arg)
        case "/attach": note(attachSummary())
        case "/remove": removeAttachment(arg)
        case "/image", "/img", "/audio":
            if arg.isEmpty { note("Usage: \(cmd) <path>") }
            else if attach(path: arg) { note(attachSummary()) }
        case "/mic": await recordMic()
        case "/voice":
            let parts = arg.lowercased().split(separator: " ").map(String.init)
            switch parts.first ?? "" {
            case "send": voiceMode = .send; note(voiceModeNote())
            case "dictate": voiceMode = .dictate; note(voiceModeNote())
            case "engine":
                switch parts.count > 1 ? parts[1] : "" {
                case "apple": voiceEngine = .apple
                    note("Voice engine: Apple on-device speech-to-text (no download).")
                case "whisper": voiceEngine = .whisper
                    let mb = WhisperModelManager.sku(whisperSKU)?.approxMB ?? 290
                    let installed = WhisperModelManager.isInstalled(whisperSKU)
                    note("Voice engine: native MLX Whisper (\(whisperSKU), English). "
                        + (installed ? "Model installed." : "Downloads ~\(mb)MB on first dictation."))
                default: view.append(Msg(role: .pre, text: voiceEngineInfo()))
                }
            case "": voiceMode = (voiceMode == .send) ? .dictate : .send; note(voiceModeNote())
            default: note("Usage: /voice send|dictate  |  /voice engine apple|whisper")
            }
        default:
            if let custom = customCommands.command(named: cmd) {
                await generate(prompt: custom.expand(arguments: arg))
            } else {
                note("Unknown command \(cmd).")
            }
        }
    }

    private func note(_ s: String) { view.append(Msg(role: .note, text: s)) }

    private func voiceModeNote() -> String {
        if voiceMode == .send { return "Voice: send as audio - the model answers your speech." }
        let engine = voiceEngine == .apple && SpeechRecognizer.isAvailable
            ? "on-device speech-to-text" : "best-effort (no on-device engine here)"
        return "Voice: dictate to the composer (\(engine)). /voice engine to choose."
    }

    /// A small preformatted card showing the dictation engine choice and the
    /// tradeoffs, so `/voice engine` lets the user pick with eyes open.
    private func voiceEngineInfo() -> String {
        func mark(_ e: VoiceEngine) -> String { voiceEngine == e ? ">" : " " }
        let mb = WhisperModelManager.sku(whisperSKU)?.approxMB ?? 290
        let have = WhisperModelManager.isInstalled(whisperSKU) ? " (installed)" : ""
        return """
        Dictation engine            /voice engine apple | whisper
        \(mark(.apple)) apple      Apple on-device speech. No download, instant, macOS-only.
        \(mark(.whisper)) whisper    Native MLX Whisper (\(whisperSKU)). Higher accuracy, fully
                     local; downloads ~\(mb)MB on first use\(have). English.
        """
    }

    // MARK: - Generation

    private func generate(prompt: String) async {
        // An empty prompt with media (e.g. a sent voice clip) shows a placeholder
        // bubble instead of a blank line.
        let shown = !prompt.isEmpty ? prompt
            : (pendingAudio != nil ? "[voice message]"
               : (!pendingImages.isEmpty ? "[media]" : prompt))
        view.append(Msg(role: .user, text: shown))
        modelTurns.append((role: "user", content: shown))
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
                let keys = reader.read() ?? []
                if keys.contains(.ctrlC) { cancelled = true; break }
                if keys.contains(.pageUp) { scrollOffset += paneHeight() - 1; render() }
                if keys.contains(.pageDown) { scrollOffset = max(0, scrollOffset - (paneHeight() - 1)); render() }
                if keys.contains(.scrollUp) { scrollOffset += 3; render() }
                if keys.contains(.scrollDown) { scrollOffset = max(0, scrollOffset - 3); render() }
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
            parts.append("ctx \(ctx) / \(formatContext(contextWindow)) (\(pct)%)")
        } else {
            parts.append("ctx \(ctx)")
        }
        return parts.joined(separator: " \u{00B7} ")
    }

    /// Compact context-window size, e.g. 131072 -> "128K", 8192 -> "8K".
    private func formatContext(_ n: Int) -> String {
        guard n >= 1024 else { return "\(n)" }
        let k = Double(n) / 1024
        return k == k.rounded() ? "\(Int(k))K" : String(format: "%.0fK", k)
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

    /// Push-to-talk: record while Space is held, send the clip when released.
    /// Terminals report no key-release, so we use the OS key-repeat stream as the
    /// "still held" signal and treat a short gap with no Space as the release.
    private func holdToTalk() async {
        guard engine.canUseNativeAudio else { return }
        guard await MicrophoneRecorder.requestAccess() else {
            note(MicrophoneCaptureError.permissionDenied.description); return
        }
        let rec = MicrophoneRecorder()
        do { try rec.start() } catch { note("\(error)"); return }
        lastStatus = "Listening... (release Space to send, Esc to cancel)"
        render()

        let releaseTicks = 8       // ~800ms with no Space repeat => released
        var idle = 0
        var cancelled = false
        var done = false
        while !done {
            if raw.waitForInput(timeoutMs: 100) {
                for k in reader.read() ?? [] {
                    if k == .char(" ") { idle = 0 }                 // still held (auto-repeat)
                    else if k == .enter { done = true }             // explicit send
                    else if k == .escape || k == .ctrlC { cancelled = true; done = true }
                }
            } else {
                idle += 1
                if idle >= releaseTicks { done = true }             // released
            }
            if tuiWinchFlag != 0 { tuiWinchFlag = 0; updateSize(); render() }
        }

        if cancelled {
            _ = try? rec.stop(); lastStatus = ""; note("Voice cancelled."); return
        }
        do {
            let wav = try rec.stop()
            switch voiceMode {
            case .send: await sendVoice(wav)
            case .dictate: await transcribeVoice(wav)
            }
        } catch { note("\(error)") }
    }

    /// Summarize the conversation so far into a concise briefing and replace the
    /// history with it, freeing context while preserving continuity (like Claude
    /// Code's /compact). The on-screen view is reset to show the summary.
    private func compactConversation() async {
        guard !modelTurns.isEmpty else { note("Nothing to compact yet."); return }
        lastStatus = "Compacting..."
        render()
        var messages: [[String: String]] = []
        if let system, !system.isEmpty { messages.append(["role": "system", "content": system]) }
        for t in modelTurns { messages.append(["role": t.role, "content": t.content]) }
        messages.append(["role": "user", "content":
            "Summarize our conversation so far as a concise briefing that preserves all key facts, decisions, code, names, numbers, and open threads, so we can continue seamlessly. Output only the summary."])
        let gen = engine.generate(
            messages: messages, params: params, maxTokens: maxTokens,
            imageData: nil, audioData: nil, imagesData: [])
        let filter = StreamingReasoningFilter()
        var summary = ""
        for await event in gen.stream {
            if event.isEnd { break }
            summary += filter.consume(event.text)
        }
        summary += filter.finish()
        lastStatus = ""
        let clean = summary.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { note("Compaction failed; conversation unchanged."); return }
        modelTurns = [
            ("user", "Summary of our earlier conversation (for context):\n\(clean)"),
            ("assistant", "Understood. Let's continue."),
        ]
        view.removeAll()
        view.append(Msg(role: .note, text: "Conversation compacted into a summary."))
        view.append(Msg(role: .assistant, text: clean))
        scrollOffset = 0
    }

    /// Send a recorded clip as an audio turn the model answers directly - the
    /// "talk to it" path (works with Gemma 4 audio, which answers spoken input).
    private func sendVoice(_ wav: Data) async {
        pendingAudio = makeAtt(.audio, wav, "voice")
        await generate(prompt: "")   // audio-only turn; shown as "[voice message]"
    }

    /// Transcribe a recorded clip with the audio model and drop the text into the
    /// composer for the user to review, edit, and send (dictation) - we do NOT
    /// auto-send. The audio itself is discarded; the sent turn is plain text.
    private func transcribeVoice(_ wav: Data) async {
        lastStatus = "Transcribing..."
        render()
        // Native MLX Whisper, when selected: highest accuracy, fully local.
        if voiceEngine == .whisper {
            if let text = await transcribeWithWhisper(wav) {
                lastStatus = ""
                let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if clean.isEmpty { note("Didn't catch that - try again."); return }
                setInput(clean)
                return
            }
            // Whisper unavailable (declined / download or load failed): fall
            // through to the on-device / model paths so dictation still works.
        }
        // Prefer Apple's on-device speech-to-text: accurate, fully local, no
        // download. Falls through to the multimodal model only when the Speech
        // framework is unavailable or yields nothing.
        if SpeechRecognizer.isAvailable, await SpeechRecognizer.requestAuthorization() {
            if let text = await speech.transcribe(wav: wav) {
                lastStatus = ""
                let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if clean.isEmpty { note("Didn't catch that - try again."); return }
                setInput(clean)
                return
            }
        }
        // Force verbatim transcription. Gemma 4's audio path will otherwise
        // ANSWER the speech (its default behaviour) rather than transcribe it;
        // a firm "you are a transcription tool, do not answer" framing biases it
        // toward the words. (A dedicated ASR model would be the reliable fix.)
        let instruction = """
        You are an automatic speech-to-text transcription tool, not an assistant. \
        Transcribe the audio verbatim: output ONLY the exact words spoken, as a single line of plain text. \
        Do not answer, reply to, explain, translate, or react to the content. \
        If the audio asks a question, transcribe the question word for word - do NOT answer it.
        """
        let gen = engine.generate(
            messages: [["role": "user", "content": instruction]],
            params: params, maxTokens: min(maxTokens, 256),
            imageData: nil, audioData: wav, imagesData: [])
        var raw = ""
        for await event in gen.stream {
            if event.isEnd { break }
            raw += event.text
        }
        lastStatus = ""
        // The audio model can answer in its reasoning channel (a bare
        // `<|channel>thought` with no close marker when it produces nothing
        // else). Run the FULL transcript through the authoritative stripper -
        // it removes complete and unclosed Gemma channels and generic think
        // tags - then drop any residual special-token markers, so the composer
        // never shows raw control text.
        let (visible, _) = ReasoningParser.strip(raw)
        let clean = visible
            .replacingOccurrences(of: #"<\|?[a-zA-Z_]+\|?>"#, with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if clean.isEmpty { note("Didn't catch that - try again."); return }
        setInput(clean)   // populate the composer; user reviews and presses Enter
    }

    /// Transcribe with the native MLX Whisper runtime. On first use this asks
    /// consent and downloads the model; returns nil (so the caller falls back)
    /// if the user declines or anything fails.
    private func transcribeWithWhisper(_ wav: Data) async -> String? {
        if !WhisperModelManager.isInstalled(whisperSKU) {
            let mb = WhisperModelManager.sku(whisperSKU)?.approxMB ?? 290
            guard confirm("Whisper needs the \(whisperSKU) model (~\(mb)MB). Download now?") else {
                note("Whisper download declined - using on-device dictation.")
                return nil
            }
            lastStatus = "Downloading Whisper \(whisperSKU) (~\(mb)MB)..."
            render()
            do {
                try await WhisperModelManager.download(whisperSKU)
            } catch {
                lastStatus = ""
                note("Whisper download failed: \(error)")
                return nil
            }
        }
        if whisper == nil {
            lastStatus = "Loading Whisper..."
            render()
            do {
                whisper = try WhisperRuntime(modelDir: WhisperModelManager.modelDir(whisperSKU))
            } catch {
                lastStatus = ""
                note("Whisper load failed: \(error)")
                return nil
            }
        }
        do {
            let waveform = try AudioPreprocessor.monoWaveform(fromAudio: wav)
            lastStatus = "Transcribing (Whisper)..."
            render()
            return whisper?.transcribe(waveform: waveform)
        } catch {
            lastStatus = ""
            note("Whisper transcription failed: \(error)")
            return nil
        }
    }

    /// Blocking y/N confirmation in the full-screen TUI. Enter/Esc/n = no.
    private func confirm(_ question: String) -> Bool {
        note("\(question) [y/N]")
        render()
        while true {
            guard raw.waitForInput(timeoutMs: 200) else { continue }
            for k in reader.read() ?? [] {
                switch k {
                case .char("y"), .char("Y"): return true
                case .char("n"), .char("N"), .enter, .escape, .ctrlC: return false
                default: continue
                }
            }
        }
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
            if (reader.read() ?? []).contains(where: { $0 == .enter }) { break }
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
        let menuLines = menu.isActive ? renderMenu(width: width) : []
        let availRows = max(0, boxTop - paneTop)     // rows paneTop .. boxTop-1
        let convHeight = max(0, availRows - menuLines.count)

        // The modal model picker replaces the conversation pane (top-anchored);
        // otherwise the conversation bottom-anchors against the input box and the
        // splash stays vertically centered.
        let pane = picker.map { renderPicker($0, width: width, height: convHeight) }
            ?? paneLines(width: width)
        let blankTop = picker != nil
            ? 0
            : Chrome.anchorBlankTop(paneCount: pane.count, convHeight: convHeight, centered: view.isEmpty)
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
        let rule = Ansi.chrome(margin + String(repeating: "\u{2500}", count: min(w, 48)))
        var lines: [String] = []
        var sawTurn = false
        for msg in view {
            switch msg.role {
            case .user:
                if sawTurn { lines.append(rule); lines.append("") }
                sawTurn = true
                for l in Layout.wrap(msg.text, width: w) { lines.append(margin + Ansi.user(l)) }
            case .assistant:
                sawTurn = true
                lines.append("")
                for l in TUIMarkdown.render(msg.text, width: w) { lines.append(margin + Ansi.model(l)) }
            case .note:
                for l in Layout.wrap(msg.text, width: w) { lines.append(margin + Ansi.chrome(l)) }
            case .pre:
                // Preformatted (help / transcript): keep spacing verbatim - no
                // word-wrap that would collapse aligned columns. A non-indented,
                // non-empty line is a section header (rendered bright).
                for raw in msg.text.split(separator: "\n", omittingEmptySubsequences: false) {
                    let clipped = String(String(raw).prefix(w))
                    if !clipped.isEmpty && !clipped.hasPrefix(" ") {
                        lines.append(margin + Ansi.bold(clipped))
                    } else {
                        lines.append(margin + Ansi.chrome(clipped))
                    }
                }
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
        if winStart > 0 { out.append(Ansi.chrome("    ...")) }
        for i in winStart..<winEnd {
            let item = menu.matches[i]
            let marker = i == menu.selected ? ">" : " "
            let name = item.name.padding(toLength: 9, withPad: " ", startingAt: 0)
            let label = "  \(marker) \(name)  \(item.summary)"
            let clipped = String(label.prefix(width))
            out.append(i == menu.selected ? Ansi.inverse(clipped) : Ansi.chrome(clipped))
        }
        if winEnd < total { out.append(Ansi.chrome("    ...")) }
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
        let top = Ansi.chrome(Chrome.border(width: w, left: tl, fill: h, right: tr))
        let bottom = Ansi.chrome(Chrome.border(width: w, left: bl, fill: h, right: br))

        let fieldWidth = max(2, w - 4)         // inner span minus one pad space each side
        let promptStr = "> "
        let textWidth = max(1, fieldWidth - promptStr.count)
        let chars = Array(input)

        var body: String
        if chars.isEmpty {
            // Block cursor then a dim placeholder, clipped/padded to the field.
            let placeholder = engine.canUseNativeAudio
                ? "hold Space to talk, or type a message"
                : "type a message   /help for commands"
            let clipped = String(placeholder.prefix(max(0, textWidth - 1)))
            let pad = String(repeating: " ", count: max(0, textWidth - 1 - clipped.count))
            body = Ansi.bold(promptStr) + Ansi.inverse(" ") + Ansi.chrome(clipped) + pad
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
        let field = Ansi.chrome(v) + " " + body + " " + Ansi.chrome(v)
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

    /// A document-style help screen: section headers with one item per line and
    /// aligned descriptions (rendered preformatted as a `.pre` message so the
    /// columns are not collapsed by word-wrap). Commands come from the canonical
    /// `SlashMenu.all` so they never drift from the autosuggest list.
    private func helpText() -> String {
        let pad = (SlashMenu.all.map { $0.name.count }.max() ?? 8) + 2
        var lines = ["Commands"]
        for item in SlashMenu.all {
            lines.append("  " + item.name.padding(toLength: pad, withPad: " ", startingAt: 0) + item.summary)
        }
        if !customCommands.isEmpty {
            let cpad = (customCommands.commands.map { $0.name.count + 1 }.max() ?? 8) + 2
            lines.append("")
            lines.append("Custom commands  (~/.krillm/commands/*.md)")
            for c in customCommands.commands {
                lines.append("  " + "/\(c.name)".padding(toLength: cpad, withPad: " ", startingAt: 0) + c.description)
            }
        }
        lines.append("")
        lines.append("Keys")
        let keys: [(String, String)] = [
            ("Up / Down", "History, or cycle the slash menu"),
            ("Tab", "Accept the highlighted command"),
            ("Enter", "Send the message"),
            ("PgUp / PgDn", "Scroll the conversation"),
            ("Ctrl-C", "Cancel the reply, or clear the input"),
            ("Ctrl-D", "Quit"),
        ]
        for (k, desc) in keys {
            lines.append("  " + k.padding(toLength: 14, withPad: " ", startingAt: 0) + desc)
        }
        lines.append("")
        lines.append("Attachments")
        lines.append("  Drag a file into the window, or type @path in your message.")
        return lines.joined(separator: "\n")
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
