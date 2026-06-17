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
    // A modal info "screen" (help, /history, voice cards): replaces the
    // conversation pane so chat never shows through. Any key (Esc) closes it;
    // Up/Down/PgUp/PgDn scroll. Mutually exclusive with the picker.
    private struct Overlay { let title: String; let lines: [String]; var scroll: Int = 0 }
    private var overlay: Overlay?
    private var pendingVoiceCapture = false   // Space-to-talk armed; run loop records
    // The voice "posture" - what speaking does. One axis, four named states the
    // user cycles with Ctrl-V (or `/voice-mode`); the active one is always shown
    // in the composer hint so it is never a hidden surprise:
    //   .type      keyboard only - Space is a typed space, no push-to-talk.
    //   .dictate   hold Space -> transcribe into the composer -> review -> Enter.
    //   .handsfree hold Space -> transcribe -> auto-send (Esc within grace cancels).
    //   .send      hold Space -> send the clip as audio the model answers.
    // Engine (Apple vs Whisper) is a separate setting (`/voice engine`).
    private enum VoiceMode: CaseIterable { case type, dictate, handsfree, send }
    // Default OFF: KrillLM is a text chat first. Voice is opt-in via Ctrl-V or the
    // `voiceMode` config key; in `.type` the footer shows no voice chrome at all.
    private var voiceMode: VoiceMode = .type
    // Which speech-to-text engine dictation uses. `.apple` (default) is Apple's
    // on-device recognizer (no download). `.whisper` is KrillLM's native MLX
    // Whisper runtime (best accuracy; downloads an English model on first use).
    private enum VoiceEngine { case apple, whisper }
    private var voiceEngine: VoiceEngine = .apple
    // Live voice action shown in the footer (Listening / Transcribing /
    // Sending...). Empty when idle, where the footer shows the posture instead.
    private var voiceActivity = ""
    // Advanced each tick while recording to animate the footer VU meter.
    private var voiceFrame = 0
    private let speech = SpeechRecognizer()
    // Lazily loaded native Whisper runtime + its SKU (English dictation).
    private var whisper: WhisperRuntime?
    private var whisperSKU = WhisperModelManager.defaultSKU
    // Text-to-speech: read model replies aloud (voice phase 2). Opt-in, default
    // off; toggled with `/speak` or the `speak_replies` config key. Pairs with
    // the hands-free posture for a full talk/listen loop. The synthesizer is
    // created lazily so it costs nothing until the user turns speaking on.
    private var speakReplies = false
    // Reasoning ("thinking") channel for models that have one. ON by default
    // (no-op for models without a thinking channel); toggled with `/think` or
    // Ctrl-T, seeded from the `thinking` config key. Passed to each generate()
    // so the model reasons before answering when on.
    private var thinkingOn = true
    private lazy var synth = SpeechSynthesizer()
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
         initialImage: Data?, initialAudio: Data?, theme: String? = nil,
         voiceModeSetting: String = "off", speakRepliesSetting: Bool = false,
         thinkingSetting: Bool = true) {
        self.engine = engine
        self.modelName = modelName
        self.system = system
        self.params = params
        self.maxTokens = maxTokens
        self.registry = registry
        self.themeOverride = theme
        self.voiceMode = Self.parseVoiceMode(voiceModeSetting)
        self.speakReplies = speakRepliesSetting
        self.thinkingOn = thinkingSetting
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
        // Silence any in-flight spoken reply on EVERY exit path, including
        // `/quit`/`/exit`/`/q` and Ctrl-D (which only set `shouldQuit`); Ctrl-C
        // and new turns already stop it inline. Without this a reply still being
        // read aloud would keep talking after the session ends.
        defer { synth.stop(); raw.leave() }

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
        // Modal screens own the keyboard while open.
        if overlay != nil { handleOverlayKey(key); return nil }
        if picker != nil { handlePickerKey(key); return nil }
        switch key {
        case .ctrlD:
            if input.isEmpty { shouldQuit = true }
            return nil
        case .ctrlC:
            synth.stop()   // also silence a spoken reply that is still playing
            if input.isEmpty { shouldQuit = true } else { input = ""; cursor = 0; menu.close() }
            return nil
        case .ctrlU:
            input = ""; cursor = 0; menu.update(for: input); return nil
        case .ctrlV:
            if engine.canUseNativeAudio { cycleVoiceMode() }   // Ctrl-V: cycle voice posture
            return nil
        case .ctrlT:
            setThinking(!thinkingOn)   // Ctrl-T: toggle the reasoning channel
            return nil
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
            // (hold to talk, release to send) - UNLESS the posture is .type, where
            // Space is just a typed space (the "don't hijack my Space" posture).
            if c == " " && input.isEmpty && engine.canUseNativeAudio && voiceMode != .type {
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

    /// Open a modal info screen (help, history, voice cards). Replaces the
    /// conversation pane so chat never shows through; any key closes it.
    private func showOverlay(_ title: String, _ body: String) {
        overlay = Overlay(title: title, lines: body.components(separatedBy: "\n"))
    }

    /// Key handling while an info overlay is open: scroll or close.
    private func handleOverlayKey(_ key: Key) {
        guard var ov = overlay else { return }
        switch key {
        case .up, .scrollUp:     ov.scroll = max(0, ov.scroll - 1); overlay = ov
        case .down, .scrollDown: ov.scroll += 1; overlay = ov
        case .pageUp:            ov.scroll = max(0, ov.scroll - (paneHeight() - 1)); overlay = ov
        case .pageDown:          ov.scroll += paneHeight() - 1; overlay = ov
        default:                 overlay = nil      // Esc / Enter / any key closes
        }
    }

    /// Render an info overlay: bold title, the (scroll-windowed) body styled like a
    /// `.pre` block, and an "Esc to close" footer - sized to fit `height`.
    private func overlayBody(_ ov: Overlay, width: Int, height: Int) -> [String] {
        let margin = "  "
        let w = max(10, width - margin.count)
        func styled(_ raw: String) -> String {
            let clipped = String(raw.prefix(w))
            // A non-indented, non-empty line is a section header (bold); else dim.
            if !clipped.isEmpty && !clipped.hasPrefix(" ") { return margin + Ansi.bold(clipped) }
            return margin + Ansi.chrome(clipped)
        }
        let head = [margin + Ansi.bold(ov.title), ""]
        let bodyHeight = max(1, height - head.count - 2)   // 2 = blank + footer
        let maxStart = max(0, ov.lines.count - bodyHeight)
        let start = min(max(0, ov.scroll), maxStart)
        var out = head
        for i in 0..<bodyHeight {
            let idx = start + i
            out.append(idx < ov.lines.count ? styled(ov.lines[idx]) : "")
        }
        let more = maxStart > 0 ? "  \u{00B7}  more (Up/Down)" : ""
        out.append("")
        out.append(margin + Ansi.chrome("Esc to close" + more))
        return out
    }

    /// Key handling while the modal model picker is open: Up/Down to move, Enter
    /// to choose (the run loop then loads or downloads it), Esc/Ctrl-C to cancel.
    private func handlePickerKey(_ key: Key) {
        switch key {
        case .up, .scrollUp: picker?.selectPrevious()
        case .down, .scrollDown: picker?.selectNext()
        case .enter:
            if let chosen = picker?.current?.name { pendingModelLoad = chosen }
            picker = nil
        case .right, .char("i"), .char("I"):
            if let name = picker?.current?.name { picker = nil; showModelDeepDive(name) }
        case .escape, .ctrlC: picker = nil
        default: break
        }
    }

    /// Build and open the model deep-dive screen: a stylized family wordmark, the
    /// live specs (params/quant/context/size/inputs derived from the registry and
    /// capabilities), and the curated profile (vendor, release, strengths,
    /// weaknesses, good-for). Falls back with a note for an unknown model.
    private func showModelDeepDive(_ name: String) {
        let store = ModelCatalogStore(baseDir: registry.baseDir)
        guard let m = AliasMap.resolve(name, catalog: store) else { note("No info for \(name)."); return }
        let caps = ModelCapabilities.capabilities(for: m.family)
        let profile = ModelProfiles.profile(for: m.family)
        // Use the resolved canonical name for registry lookups (the arg may be an
        // alias the registry does not key by).
        let onDisk = registry.hasModel(m.name) ? directorySize(registry.modelPath(m.name)) : 0
        let sizeStr: String
        if onDisk > 0 { sizeStr = "\(formatSize(onDisk)) (installed)" }
        else if let est = estimatedSizeBytes(params: m.params, quant: m.quant) { sizeStr = "~\(formatSize(est)) (download)" }
        else { sizeStr = "unknown" }
        // Inputs reflect declared capabilities: text only for generative models;
        // encoders (embeddings / reranker) are not chat inputs.
        var inputs: [String] = []
        if caps.contains(.textGeneration) { inputs.append("text") }
        if caps.contains(.visionInput) { inputs.append("image") }
        if caps.contains(.audioInput) { inputs.append("audio") }
        if inputs.isEmpty { inputs.append(caps.contains(.embeddings) ? "text (embeddings)" : "n/a") }
        var feats: [String] = []
        if caps.contains(.tools) { feats.append("tools") }
        if caps.contains(.moe) { feats.append("MoE") }
        if caps.contains(.reranker) { feats.append("reranker") }

        var lines = BlockFont.render(profile?.displayName ?? m.name.uppercased())
        lines.append("")
        if let p = profile {
            lines.append("  \(p.vendor)  \u{00B7}  released \(p.released)")
            lines.append("  training cutoff: \(p.trainingCutoff)")
            lines.append("  \(p.tagline)")
            lines.append("")
        }
        lines.append("Specs")
        lines.append("  params      \(m.params)")
        lines.append("  quant       \(m.quant)")
        lines.append("  context     \(formatContext(m.context))")
        lines.append("  size        \(sizeStr)")
        lines.append("  inputs      \(inputs.joined(separator: ", "))")
        if !feats.isEmpty { lines.append("  features    \(feats.joined(separator: ", "))") }
        lines.append("  repo        \(m.repo)")
        if let p = profile {
            lines.append("")
            lines.append("Strengths")
            for s in p.strengths { lines.append("  + \(s)") }
            lines.append("")
            lines.append("Weaknesses")
            for w in p.weaknesses { lines.append("  - \(w)") }
            lines.append("")
            lines.append("Good for")
            lines.append("  \(p.goodFor.joined(separator: ", "))")
        }
        showOverlay(m.name, lines.joined(separator: "\n"))
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
            entries.append(.init(name: name, params: params, quant: quant,
                                 size: size, downloaded: downloaded))
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
            margin + Ansi.chrome("press i (or right-arrow) for a model deep-dive"),
            "",
        ]
        let total = p.entries.count
        // Column widths across the whole list so rows line up in both sections.
        func rpad(_ s: String, _ w: Int) -> String { s.padding(toLength: w, withPad: " ", startingAt: 0) }
        func lpad(_ s: String, _ w: Int) -> String { String(repeating: " ", count: max(0, w - s.count)) + s }
        let nameW  = min(22, max(8, p.entries.map { $0.name.count }.max() ?? 8))
        let paramW = max(3, p.entries.map { $0.params.count }.max() ?? 3)
        let quantW = max(4, p.entries.map { $0.quant.count }.max() ?? 4)
        let sizeW  = max(5, p.entries.map { $0.size.count }.max() ?? 5)
        // Window the visible rows around the selection (headers/blanks are added
        // around them, so reserve a few lines of chrome).
        let maxVisible = max(3, height - 8)
        var winStart = 0
        if total > maxVisible { winStart = min(max(0, p.selected - maxVisible / 2), total - maxVisible) }
        let winEnd = min(winStart + maxVisible, total)
        if winStart > 0 { out.append(margin + Ansi.chrome("   ...")) }
        var lastSection: Bool? = nil
        for i in winStart..<winEnd {
            let e = p.entries[i]
            if lastSection != e.downloaded {
                if i > winStart { out.append("") }
                out.append(margin + Ansi.chrome(e.downloaded ? "Installed" : "Available"))
                lastSection = e.downloaded
            }
            let chevron = i == p.selected ? "\u{25B8}" : " "        // selected marker
            let row = "\(chevron) \(rpad(e.name, nameW))  \(lpad(e.params, paramW))  \(rpad(e.quant, quantW))  \(lpad(e.size, sizeW))"
            let line = margin + String(row.prefix(max(0, width - margin.count)))
            if i == p.selected { out.append(Ansi.bold(line)) }      // light: chevron + bold
            else if e.downloaded { out.append(Ansi.user(line)) }
            else { out.append(Ansi.chrome(line)) }
        }
        if winEnd < total { out.append(margin + Ansi.chrome("   ...")) }
        out.append("")
        out.append(margin + Ansi.chrome("Up/Down  \u{00B7}  Enter switch  \u{00B7}  i details  \u{00B7}  Esc cancel"))
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
    private static let commandAliases: Set<String> = ["/img", "/exit", "/q", "/reset", "/vmode"]

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
        case "/help": showOverlay("Help", helpText())
        case "/drop": pendingImages.removeAll(); pendingAudio = nil; note("Dropped pending attachments.")
        case "/clear", "/reset":   // /reset kept as an alias for clearing the chat
            modelTurns.removeAll(); view.removeAll(); pendingImages.removeAll(); pendingAudio = nil
            note("Conversation cleared.")
        case "/history":
            showOverlay("Conversation", modelTurns.isEmpty ? "No conversation yet."
                 : modelTurns.map { "\($0.role)\n  \($0.content)" }.joined(separator: "\n\n"))
        case "/compact": await compactConversation()
        case "/system":
            if arg.isEmpty {
                showOverlay("System prompt", system.map { "  \($0)" } ?? "(none set)\n\nUsage: /system <text>")
            } else { system = arg; note("System prompt updated.") }
        case "/model":
            if arg.isEmpty { picker = ModelPicker(entries: modelPickerEntries(), current: modelName) }
            else if arg.lowercased() == "info" || arg.lowercased().hasPrefix("info ") {
                let n = String(arg.dropFirst(4)).trimmingCharacters(in: .whitespaces)
                showModelDeepDive(n.isEmpty ? modelName : n)
            }
            else { await switchOrDownload(arg) }
        case "/save": saveTranscript(arg)
        case "/attach": note(attachSummary())
        case "/remove": removeAttachment(arg)
        case "/image", "/img", "/audio":
            if arg.isEmpty { note("Usage: \(cmd) <path>") }
            else if attach(path: arg) { note(attachSummary()) }
        case "/mic": await recordMic()
        case "/voice-mode", "/vmode":
            switch arg.lowercased().split(separator: " ").map(String.init).first ?? "" {
            case "type": setVoiceMode(.type)
            case "dictate": setVoiceMode(.dictate)
            case "handsfree", "hands-free": setVoiceMode(.handsfree)
            case "send": setVoiceMode(.send)
            case "", "next", "cycle": cycleVoiceMode()
            default: note("Usage: /voice-mode type|dictate|handsfree|send  (or Ctrl-V to cycle)")
            }
        case "/speak", "/tts":
            switch arg.lowercased().trimmingCharacters(in: .whitespaces) {
            case "on": setSpeakReplies(true)
            case "off": setSpeakReplies(false)
            case "", "toggle": setSpeakReplies(!speakReplies)
            default: note("Usage: /speak on|off  (read model replies aloud)")
            }
        case "/think", "/thinking":
            switch arg.lowercased().trimmingCharacters(in: .whitespaces) {
            case "on": setThinking(true)
            case "off": setThinking(false)
            case "", "toggle": setThinking(!thinkingOn)
            default: note("Usage: /think on|off  (reason before answering; Ctrl-T to toggle)")
            }
        case "/voice":
            let parts = arg.lowercased().split(separator: " ").map(String.init)
            switch parts.first ?? "" {
            case "type": setVoiceMode(.type)
            case "send": setVoiceMode(.send)
            case "dictate": setVoiceMode(.dictate)
            case "handsfree", "hands-free": setVoiceMode(.handsfree)
            case "engine":
                switch parts.count > 1 ? parts[1] : "" {
                case "apple": voiceEngine = .apple
                    note("Voice engine: Apple on-device speech-to-text (no download).")
                case "whisper": voiceEngine = .whisper
                    // Optional SKU: /voice engine whisper <tiny|base|small|*.en>
                    if parts.count > 2 {
                        if let sku = WhisperModelManager.sku(parts[2]) {
                            if sku.id != whisperSKU { whisper = nil }   // drop cached runtime
                            whisperSKU = sku.id
                        } else {
                            note("Unknown Whisper model '\(parts[2])'. Options: "
                                + WhisperModelManager.skus.map { $0.id }.joined(separator: ", "))
                        }
                    }
                    let mb = WhisperModelManager.sku(whisperSKU)?.approxMB ?? 290
                    let installed = WhisperModelManager.isInstalled(whisperSKU)
                    let lang = WhisperModelManager.isMultilingual(whisperSKU)
                        ? "~99 languages, auto-detected" : "English"
                    note("Voice engine: native MLX Whisper (\(whisperSKU), \(lang)). "
                        + (installed ? "Model installed." : "Downloads ~\(mb)MB on first dictation."))
                default: showOverlay("Voice engine", voiceEngineInfo())
                }
            case "": showOverlay("Voice", voiceStatusCard())
            default: note("Usage: /voice-mode type|dictate|handsfree|send  |  /voice engine apple|whisper [model]")
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

    /// Map the `voice_mode` config string to a posture (default off/text).
    private static func parseVoiceMode(_ s: String) -> VoiceMode {
        switch s.lowercased() {
        case "dictate": return .dictate
        case "handsfree", "hands-free": return .handsfree
        case "send": return .send
        default: return .type   // "off" / "text" / anything unknown
        }
    }

    /// Short label for the active dictation engine (Apple vs native Whisper).
    private var engineLabel: String { voiceEngine == .whisper ? "Whisper" : "Apple" }

    /// Set the voice posture and confirm it on screen.
    private func setVoiceMode(_ m: VoiceMode) { voiceMode = m; note(voiceModeNote()) }

    /// Turn spoken replies (TTS) on or off and confirm it on screen. Stops any
    /// in-flight speech when turning off.
    private func setSpeakReplies(_ on: Bool) {
        speakReplies = on
        if !on { synth.stop() }
        if on, !SpeechSynthesizer.isAvailable {
            note("Spoken replies aren't available on this system.")
            speakReplies = false
            return
        }
        note(on
            ? "Spoken replies: ON. Replies are read aloud (pairs with hands-free voice). /speak off to stop."
            : "Spoken replies: OFF.")
    }

    /// Turn the reasoning ("thinking") channel on or off for the rest of the
    /// session and confirm it on screen. A no-op for models without a thinking
    /// channel. Toggled with `/think` or Ctrl-T.
    private func setThinking(_ on: Bool) {
        thinkingOn = on
        note(on
            ? "Thinking: ON. The model reasons before answering (no-op if the model has no thinking channel). /think off or Ctrl-T to stop."
            : "Thinking: OFF. The model answers directly.")
    }

    /// Advance to the next posture, the Ctrl-V action. The cycle starts and ends
    /// at `.type` (voice off) so one key both enables voice and turns it back off:
    /// off -> dictate -> handsfree -> send -> off.
    private func cycleVoiceMode() {
        let cycle: [VoiceMode] = [.type, .dictate, .handsfree, .send]
        let i = cycle.firstIndex(of: voiceMode) ?? 0
        voiceMode = cycle[(i + 1) % cycle.count]
        note(voiceModeNote())
    }

    private func voiceModeNote() -> String {
        switch voiceMode {
        case .type:
            return "Voice: Type. Space types a space; Enter sends. Ctrl-V to cycle voice."
        case .dictate:
            return "Voice: Dictate (\(engineLabel)). Hold Space to talk; text lands in the composer to review. Ctrl-V to cycle."
        case .handsfree:
            return "Voice: Hands-free (\(engineLabel)). Hold Space to talk; auto-sends after a short grace (Esc cancels). Ctrl-V to cycle."
        case .send:
            return "Voice: Send audio. Hold Space; the clip is sent as audio the model answers. Ctrl-V to cycle."
        }
    }

    /// Footer halves: the voice posture / live activity on the LEFT (a mono dot
    /// that pulses while recording), generation stats + version on the RIGHT. On
    /// non-audio models the left falls back to the generation status. Folding the
    /// voice state into the footer keeps it persistent without a second status row.
    private func footerContent() -> (left: String, right: String) {
        let sep = " \u{00B7} "
        // Spoken-replies (TTS) indicator: independent of the mic, so it shows even
        // on text-only models. Leads the right half when on.
        let speakTag = speakReplies ? "\u{25CF} speak\(sep)" : ""
        // Reasoning indicator: a mono dot when the thinking channel is on.
        let thinkTag = thinkingOn ? "\u{25CF} think\(sep)" : ""
        let cleanRight = "\(thinkTag)\(speakTag)\(cwdLabel)\(sep)\(KrillLMVersionTag)"
        // Voice OFF (text mode) or a non-audio model: no voice chrome at all - the
        // footer is just the generation status and cwd/version (plus speak tag).
        guard engine.canUseNativeAudio, voiceMode != .type || !voiceActivity.isEmpty else {
            return (lastStatus.isEmpty ? "ready" : lastStatus, cleanRight)
        }
        let dot = "\u{25CF}"
        let left: String
        if !voiceActivity.isEmpty {
            left = "\(vuMeter(voiceFrame)) \(voiceActivity)"   // animated meter while live
        } else {
            switch voiceMode {
            case .dictate:   left = "\(dot) dictate\(sep)\(engineLabel)\(sep)Ctrl-V to cycle"
            case .handsfree: left = "\(dot) hands-free\(sep)\(engineLabel)\(sep)Ctrl-V to cycle"
            case .send:      left = "\(dot) send audio (talk to it)\(sep)Ctrl-V to cycle"
            case .type:      left = ""   // unreachable (guarded above)
            }
        }
        let right = lastStatus.isEmpty ? cleanRight : "\(speakTag)\(lastStatus)\(sep)\(KrillLMVersionTag)"
        return (left, right)
    }

    /// A small monochrome VU meter that "dances" while recording - purely cosmetic
    /// (a rolling triangle wave, not real audio levels), so it reads as live.
    private static let vuBars = Array("\u{2581}\u{2582}\u{2583}\u{2584}\u{2585}\u{2586}\u{2587}\u{2588}")  // bar heights
    private func vuMeter(_ frame: Int) -> String {
        let bars = Self.vuBars, n = 7, span = bars.count * 2 - 2
        var out = ""
        for i in 0..<n {
            let p = (frame + i * 2) % span
            let h = p < bars.count ? p : span - p   // triangle wave 0..max..0
            out.append(bars[max(0, min(bars.count - 1, h))])
        }
        return out
    }

    /// Short contextual hint shown faded + italic, right-aligned above the input
    /// box: what Space does in the current posture (and how to reach voice when
    /// it is off). Empty on non-audio models, where there is nothing to hint.
    private func composerHint() -> String {
        let sep = " \u{00B7} "
        guard engine.canUseNativeAudio else { return "" }
        switch voiceMode {
        case .type:      return "activate voice mode: Ctrl-V"
        case .dictate:   return "hold Space to dictate\(sep)Ctrl-V to cycle"
        case .handsfree: return "hold Space, hands-free\(sep)Ctrl-V to cycle"
        case .send:      return "hold Space to talk to it\(sep)Ctrl-V to cycle"
        }
    }

    /// Full voice state for a bare `/voice` (or `/voice-mode` with no arg shows
    /// it too). Lists every posture with the active one marked, plus the engine
    /// card. Replaces the old silent dictate<->send toggle.
    private func voiceStatusCard() -> String {
        func mark(_ on: Bool) -> String { on ? ">" : " " }
        let eng = voiceEngine == .whisper ? "Whisper (\(whisperSKU))" : "Apple on-device"
        return """
        Voice posture               Ctrl-V to cycle  |  /voice-mode <name>
        \(mark(voiceMode == .type)) type       keyboard only - Space types a space, Enter sends
        \(mark(voiceMode == .dictate)) dictate    hold Space -> transcribe to composer -> review -> Enter (engine: \(eng))
        \(mark(voiceMode == .handsfree)) handsfree  hold Space -> transcribe -> auto-send (Esc cancels)
        \(mark(voiceMode == .send)) send       hold Space -> talk to it; the model answers your speech in text

        \(voiceEngineInfo())
        """
    }

    /// A small preformatted card showing the dictation engine choice and the
    /// tradeoffs, so `/voice engine` lets the user pick with eyes open.
    private func voiceEngineInfo() -> String {
        func mark(_ e: VoiceEngine) -> String { voiceEngine == e ? ">" : " " }
        let mb = WhisperModelManager.sku(whisperSKU)?.approxMB ?? 290
        let have = WhisperModelManager.isInstalled(whisperSKU) ? " (installed)" : ""
        let lang = WhisperModelManager.isMultilingual(whisperSKU) ? "multilingual" : "English"
        return """
        Dictation engine            /voice engine apple | whisper [model]
        \(mark(.apple)) apple      Apple on-device speech. No download, instant, macOS-only.
        \(mark(.whisper)) whisper    Native MLX Whisper (\(whisperSKU), \(lang)). Higher accuracy,
                     fully local; downloads ~\(mb)MB on first use\(have).
                     Models: tiny|base|small (multilingual) or *.en (English).
        """
    }

    // MARK: - Generation

    private func generate(prompt: String, displayAs: String? = nil) async {
        synth.stop()   // a new turn hushes any still-speaking previous reply
        // `displayAs` is the on-screen bubble; the model always receives `prompt`.
        // They differ for a sent voice clip: the user sees "[voice message]" but
        // the model gets a plain instruction to answer the audio. Feeding the
        // literal "[voice message]" to the model made gemma reply to those words
        // ("I cannot provide a voice message. I am a text-based AI") instead of
        // responding to the speech.
        let shown = displayAs ?? (!prompt.isEmpty ? prompt
            : (pendingAudio != nil ? "[voice message]"
               : (!pendingImages.isEmpty ? "[media]" : prompt)))
        view.append(Msg(role: .user, text: shown))
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
            imageData: imgs.first, audioData: pendingAudio?.data, imagesData: imgs,
            enableThinking: thinkingOn)
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
        // Defensive final pass: strip any residual reasoning markers a degenerate
        // think-loop may have leaked (e.g. a stray Gemma channel close token), and
        // surface an all-thinking / empty reply cleanly instead of as raw markers.
        // The regex is scoped to the channel/think tokens ONLY so legitimate
        // angle-bracket content in a reply (e.g. <html>, <div>) is left intact.
        let (cleanVisible, _) = ReasoningParser.strip(assistant)
        assistant = cleanVisible
            .replacingOccurrences(of: #"</?\|?(?:channel|think|thinking)\|?>"#, with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        view[aIdx].text = assistant.isEmpty ? "(no response)" : assistant

        if cancelled {
            synth.stop()                       // hush any in-flight speech
            view.append(Msg(role: .note, text: "(cancelled)"))
        } else {
            modelTurns.append((role: "assistant", content: assistant))
            if let st = gen.stats() { lastStatus = statusText(st, images: usedImgs, audio: usedAud) }
            // Read the reply aloud when speaking is on (voice phase 2). Cleaned of
            // markdown that reads badly; a new reply interrupts the previous one.
            if speakReplies, !assistant.isEmpty { synth.speak(assistant) }
        }
        pendingImages.removeAll(); pendingAudio = nil
        render()
    }

    private func statusText(_ st: GenerationStats, images: Int, audio: Bool) -> String {
        var parts = [modelName]
        if images > 0 { parts.append("\(images) img") }
        if audio { parts.append("audio") }
        parts.append(String(format: "\u{00BB} %.0f tok/s", st.decodeTokensPerSecond))   // flow glyph
        let ctx = st.promptTokens + st.generatedTokens
        if contextWindow > 0 {
            let frac = min(1.0, Double(ctx) / Double(contextWindow))
            let pct = Int((frac * 100).rounded())
            parts.append("ctx \(contextBar(frac)) \(ctx)/\(formatContext(contextWindow)) \(pct)%")
        } else {
            parts.append("ctx \(ctx)")
        }
        return parts.joined(separator: " \u{00B7} ")
    }

    /// A small monochrome fill bar for context usage: filled vs empty squares.
    private func contextBar(_ frac: Double, cells: Int = 8) -> String {
        let filled = max(0, min(cells, Int((frac * Double(cells)).rounded())))
        return String(repeating: "\u{25B0}", count: filled)        // filled square
             + String(repeating: "\u{25B1}", count: cells - filled) // empty square
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
        guard engine.canUseNativeAudio, voiceMode != .type else { return }
        // Show this BEFORE the access await: a re-signed bundle resets TCC, so the
        // system mic/Speech prompt may sit behind the terminal - without a hint the
        // app looks frozen ("nothing happens" on Space).
        voiceActivity = "Requesting mic access (allow the system prompt)..."
        render()
        guard await MicrophoneRecorder.requestAccess() else {
            voiceActivity = ""
            note("Microphone access denied. Enable KrillLM under System Settings > Privacy & Security > Microphone (and Speech Recognition), then try again.")
            return
        }
        let rec = MicrophoneRecorder()
        do { try rec.start() } catch { note("\(error)"); return }
        voiceActivity = "Listening... (release Space to send, Esc to cancel)"
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
            // Live duration readout so it is obvious the mic is actually capturing;
            // advance the footer VU meter.
            voiceFrame += 1
            voiceActivity = String(format: "listening %.1fs... (release Space to send, Esc to cancel)",
                                   rec.capturedSeconds)
            render()
            if tuiWinchFlag != 0 { tuiWinchFlag = 0; updateSize(); render() }
        }

        if cancelled {
            _ = try? rec.stop(); voiceActivity = ""; note("Voice cancelled."); return
        }
        do {
            let wav = try rec.stop()
            // Guard against empty / silent captures (a too-quick tap, a muted mic,
            // or the wrong input device). Sending silence makes the model answer
            // "I can't hear any audio" and dictation returns nothing - both
            // baffling. Warn with the likely cause instead.
            if isSilentClip(wav) {
                voiceActivity = ""
                note("Didn't catch any audio - hold Space, speak, then release. (Check the mic input.)")
                return
            }
            switch voiceMode {
            case .type: break                          // PTT is disabled in this posture
            case .dictate: await transcribeVoice(wav)
            case .handsfree: await handsfreeVoice(wav)
            case .send: await sendVoice(wav)
            }
        } catch { note("\(error)") }
    }

    /// True when a recorded clip is empty or effectively silent - too short or too
    /// quiet for the model or recognizer to use. Decodes to mono PCM and checks
    /// both duration and peak amplitude (16 kHz; speech peaks well above 0.01).
    private func isSilentClip(_ wav: Data) -> Bool {
        guard let wave = try? AudioPreprocessor.monoWaveform(fromAudio: wav) else { return true }
        if wave.count < 16_000 / 8 { return true }            // under ~0.125s
        var peak: Float = 0
        for s in wave { let a = abs(s); if a > peak { peak = a } }
        return peak < 0.005                                    // true silence; quiet speech peaks well above
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
            imageData: nil, audioData: nil, imagesData: [], enableThinking: thinkingOn)
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
        // Give the model a plain instruction to answer the audio (shown as
        // "[voice message]"); an empty turn can be dropped by the chat template,
        // and the literal placeholder made gemma refuse as "a text-based AI".
        await generate(prompt: "Listen to the audio and respond to what was said.",
                       displayAs: "[voice message]")
    }

    /// Hands-free: transcribe the clip, drop the text in the composer so the user
    /// always SEES what will be sent, then auto-send after a short grace window.
    /// Esc (or editing the text) cancels the auto-send and keeps it in the
    /// composer; Enter sends immediately. The reply is shown on screen - spoken
    /// replies (TTS) are a planned follow-up to make this fully hands-free.
    private func handsfreeVoice(_ wav: Data) async {
        await transcribeVoice(wav)            // fills `input` with the transcript (or notes a miss)
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }   // nothing transcribed; stay in the composer
        // Grace window: count down while showing the transcript, cancellable.
        let graceTicks = 15                   // ~1.5s at 100ms/tick
        var t = 0
        var send = true
        while t < graceTicks {
            voiceFrame += 1
            voiceActivity = String(format: "sending in %.1fs - Esc to cancel, Enter to send now",
                                   Double(graceTicks - t) / 10.0)
            render()
            if raw.waitForInput(timeoutMs: 100) {
                for k in reader.read() ?? [] {
                    if k == .escape || k == .ctrlC { send = false; t = graceTicks; break }
                    if k == .enter { t = graceTicks; break }      // send now
                }
            } else {
                t += 1
            }
            if tuiWinchFlag != 0 { tuiWinchFlag = 0; updateSize() }
        }
        voiceActivity = ""
        // Bail if the user edited the composer during the grace window (their
        // text wins over the auto-send).
        guard send, input.trimmingCharacters(in: .whitespacesAndNewlines) == text else {
            if !send { note("Auto-send cancelled - edit and press Enter when ready.") }
            render(); return
        }
        input = ""; cursor = 0; menu.close()
        await processSubmit(text)
    }

    /// Transcribe a recorded clip with the audio model and drop the text into the
    /// composer for the user to review, edit, and send (dictation) - we do NOT
    /// auto-send. The audio itself is discarded; the sent turn is plain text.
    private func transcribeVoice(_ wav: Data) async {
        voiceActivity = "Transcribing..."
        render()
        // Native MLX Whisper, when selected: highest accuracy, fully local.
        if voiceEngine == .whisper {
            if let text = await transcribeWithWhisper(wav) {
                voiceActivity = ""
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
                voiceActivity = ""
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
            imageData: nil, audioData: wav, imagesData: [], enableThinking: thinkingOn)
        var raw = ""
        for await event in gen.stream {
            if event.isEnd { break }
            raw += event.text
        }
        voiceActivity = ""
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
            voiceActivity = "Downloading Whisper \(whisperSKU) (~\(mb)MB)..."
            render()
            do {
                try await WhisperModelManager.download(whisperSKU)
            } catch {
                voiceActivity = ""
                note("Whisper download failed: \(error)")
                return nil
            }
        }
        if whisper == nil {
            voiceActivity = "Loading Whisper..."
            render()
            do {
                whisper = try WhisperRuntime(modelDir: WhisperModelManager.modelDir(whisperSKU))
            } catch {
                voiceActivity = ""
                note("Whisper load failed: \(error)")
                return nil
            }
        }
        do {
            let waveform = try AudioPreprocessor.monoWaveform(fromAudio: wav)
            voiceActivity = "Transcribing (Whisper)..."
            render()
            return whisper?.transcribe(waveform: waveform)
        } catch {
            voiceActivity = ""
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
    // The voice posture/activity rides the footer's left side (no separate row).
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
        let modal = picker != nil || overlay != nil
        let menuLines = menu.isActive ? renderMenu(width: width) : []
        // A faded, italic, right-aligned hint sits on the row just above the input
        // box (hidden while the slash popup or a modal screen owns that space).
        let hintText = (menu.isActive || modal) ? "" : composerHint()
        let hintRows = hintText.isEmpty ? 0 : 1
        let availRows = max(0, boxTop - paneTop)     // rows paneTop .. boxTop-1
        let convHeight = max(0, availRows - menuLines.count - hintRows)

        // A modal screen (info overlay or model picker) replaces the conversation
        // pane top-anchored; otherwise the conversation bottom-anchors against the
        // input box and the splash stays vertically centered.
        let pane: [String]
        if let ov = overlay { pane = overlayBody(ov, width: width, height: convHeight) }
        else if let p = picker { pane = renderPicker(p, width: width, height: convHeight) }
        else { pane = paneLines(width: width) }
        let blankTop = modal
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
        // Faded italic hint, right-aligned on the row just above the input box.
        if hintRows > 0 {
            let clipped = String(hintText.prefix(max(0, width - 2)))
            let pad = max(0, width - clipped.count - 2)
            frame += positioned(boxTop - 1, String(repeating: " ", count: pad) + Ansi.hint(clipped))
        }
        // Input box, then the footer (which carries the voice posture on its left).
        for (i, line) in box.enumerated() {
            frame += positioned(boxTop + i, line)
        }
        let (footerLeft, footerRight) = footerContent()
        frame += positioned(footerRow, Brand.footer(width: width, left: footerLeft, right: footerRight))
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
            // Block cursor then a dim placeholder. The action hint lives faded +
            // italic above the box (composerHint), so the placeholder stays plain.
            let placeholder = "type a message   /help for commands"
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
            ("Hold Space", "Push-to-talk (dictate/handsfree/send postures)"),
            ("Ctrl-V", "Cycle voice posture: type/dictate/handsfree/send"),
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
