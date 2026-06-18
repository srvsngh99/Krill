import CEditLine
import Foundation
import KLMCore
import KLMEngine
import KLMRegistry
import KLMSampler
import KLMServer

// SIGINT during a reply sets this flag (async-signal-safe: only a sig_atomic_t
// is touched). The decode display loop polls it and breaks; breaking ends the
// for-await, the stream deinits, its onTermination fires, and the engine decode
// loop stops (see GenerationCancelToken). So Ctrl-C halts the GPU work, not just
// the on-screen stream. Installed only for the duration of a reply.
private nonisolated(unsafe) var replyCancelFlag: sig_atomic_t = 0
private func replySigintHandler(_ sig: Int32) { replyCancelFlag = 1 }

/// Install the reply-cancel SIGINT handler and make sure SIGINT is unblocked on
/// this thread (the Swift/dispatch runtime blocks it by default, which would
/// leave a Ctrl-C pending and undelivered).
private func installReplySigint() {
    var unblock = sigset_t()
    sigemptyset(&unblock)
    sigaddset(&unblock, SIGINT)
    pthread_sigmask(SIG_UNBLOCK, &unblock, nil)
    signal(SIGINT, replySigintHandler)
}

/// The interactive chat REPL: multi-turn conversation memory, libedit line
/// editing + history + tab completion, streamed markdown-lite output with a
/// thinking spinner and a per-turn status line, and mid-conversation media
/// attachment (image / audio / mic). Driven from `RunCommand` when no prompt is
/// given.
final class InteractiveSession {
    /// A pending attachment carried into the next user message.
    struct Attachment {
        let kind: MediaKind
        let data: Data
        let name: String
        let dims: (width: Int, height: Int)?

        var sizeLabel: String {
            let bytes = data.count
            if bytes < 1024 { return "\(bytes) B" }
            let kb = Double(bytes) / 1024
            return kb < 1024 ? String(format: "%.0f KB", kb) : String(format: "%.1f MB", kb / 1024)
        }
    }

    private var engine: InferenceEngine
    private var modelName: String
    private var system: String?
    private let params: SamplingParams
    private let maxTokens: Int
    private let registry: Registry
    // Reasoning channel default for this REPL session (from the `thinking` config
    // key; on by default). No-op for models without a thinking channel.
    private let thinking: Bool

    private var history: [(role: String, content: String)] = []
    private var pendingImages: [Attachment] = []
    private var pendingAudio: Attachment?

    private static let separator = " | "

    init(
        engine: InferenceEngine,
        modelName: String,
        system: String?,
        params: SamplingParams,
        maxTokens: Int,
        registry: Registry,
        initialImage: Data? = nil,
        initialAudio: Data? = nil,
        thinking: Bool = true
    ) {
        self.engine = engine
        self.modelName = modelName
        self.system = system
        self.params = params
        self.maxTokens = maxTokens
        self.registry = registry
        self.thinking = thinking
        if let initialImage { pendingImages.append(makeAttachment(.image, initialImage, name: "image")) }
        if let initialAudio { pendingAudio = makeAttachment(.audio, initialAudio, name: "audio") }
    }

    // MARK: - Main loop

    func run() async throws {
        ReplCompletion.install()
        installReplySigint()
        print(Ansi.bold("\nKrill interactive chat") + " " + Ansi.dim("(\(modelName))"))
        print(Ansi.dim("Type a message. /help for commands, Tab to complete, Up/Down for history, /quit to exit.\n"))
        if !pendingImages.isEmpty || pendingAudio != nil { printPending() }

        while true {
            guard let line = LineEditor.readLine(prompt: Ansi.prompt("> ", "32")) else {
                print()
                break   // Ctrl-D
            }
            if line.isEmpty { continue }
            LineEditor.addHistory(line)
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }

            if try await handleLine(trimmed) { continue }

            // Not a command or bare attachment path: pull inline @path media,
            // then send whatever text remains.
            let (cleaned, inline) = extractInlineMedia(trimmed)
            pendingImages.append(contentsOf: inline.filter { $0.kind == .image })
            if let aud = inline.last(where: { $0.kind == .audio }) { pendingAudio = aud }
            let promptText = cleaned.trimmingCharacters(in: .whitespaces)
            if promptText.isEmpty {
                if !inline.isEmpty { printPending() }
                continue
            }
            await generate(userText: promptText)
        }
    }

    /// Handle slash commands and bare attachment paths. Returns true when the
    /// line was fully handled (caller should read the next line).
    private func handleLine(_ trimmed: String) async throws -> Bool {
        let parts = trimmed.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
        let cmd = String(parts.first ?? "")
        let arg = parts.count > 1 ? String(parts[1]).trimmingCharacters(in: .whitespaces) : ""
        let known: Set<String> = [
            "/quit", "/exit", "/q", "/help", "/?", "/clear", "/reset", "/history",
            "/system", "/model", "/save", "/attach", "/remove", "/image", "/img", "/audio", "/mic",
        ]
        if known.contains(cmd) {
            switch cmd {
            case "/quit", "/exit", "/q":
                print(Ansi.dim("Goodbye."))
                throw CleanExit()
            case "/help", "/?":
                printHelp()
            case "/clear":
                pendingImages.removeAll(); pendingAudio = nil
                print(Ansi.dim("Cleared pending attachments."))
            case "/reset":
                history.removeAll(); pendingImages.removeAll(); pendingAudio = nil
                print(Ansi.dim("Conversation reset."))
            case "/history":
                printHistory()
            case "/system":
                if arg.isEmpty {
                    print(Ansi.dim(system.map { "System: \($0)" } ?? "No system prompt set. Usage: /system <text>"))
                } else {
                    system = arg
                    print(Ansi.dim("System prompt updated."))
                }
            case "/model":
                if arg.isEmpty { print(Ansi.dim("Current model: \(modelName)")) }
                else { await switchModel(to: arg) }
            case "/save":
                saveTranscript(to: arg)
            case "/attach":
                printPending()
            case "/remove":
                removeAttachment(arg)
            case "/image", "/img", "/audio":
                if arg.isEmpty { print("Usage: \(cmd) <path>") }
                else if attach(path: arg) { printPending() }
            case "/mic":
                if !engine.canUseNativeAudio { print("This model cannot process audio; /mic is unavailable.") }
                else if let wav = await recordFromMic() {
                    pendingAudio = makeAttachment(.audio, wav, name: "microphone")
                    printPending()
                }
            default:
                break
            }
            return true
        }

        // Bare dropped/typed path resolving to media.
        switch loadMedia(trimmed) {
        case .ok(let kind, let data):
            addAttachment(makeAttachment(kind, data, name: (MediaAttachment.normalizePath(trimmed) as NSString).lastPathComponent))
            printPending()
            return true
        case .unsupported(let k):
            print("This model cannot process \(k.rawValue) input.")
            return true
        case .notFound, .notMedia:
            break
        }

        // A "/word" that is neither a known command nor a path is a typo.
        if cmd.hasPrefix("/"), String(cmd.dropFirst()).allSatisfy({ $0.isLetter || $0 == "?" }) {
            print("Unknown command \(cmd). Type /help for the list.")
            return true
        }
        return false
    }

    // MARK: - Generation

    private func generate(userText: String) async {
        var messages: [[String: String]] = []
        if let system, !system.isEmpty { messages.append(["role": "system", "content": system]) }
        for turn in history { messages.append(["role": turn.role, "content": turn.content]) }
        messages.append(["role": "user", "content": userText])

        let imgs = pendingImages.map(\.data)
        let usedImages = pendingImages.count
        let usedAudio = pendingAudio != nil

        let filter = StreamingReasoningFilter()
        let md = MarkdownStream()
        let spinner = Spinner("thinking")

        // The SIGINT handler is installed for the whole session (see
        // installReplySigint); just arm it for this reply.
        replyCancelFlag = 0

        print()
        spinner.start()
        let generation = engine.generate(
            messages: messages, params: params, maxTokens: maxTokens,
            imageData: imgs.first, audioData: pendingAudio?.data, imagesData: imgs,
            enableThinking: thinking)
        var stream: AsyncStream<TokenEvent>? = generation.stream
        let getStats = generation.stats

        var assistant = ""
        var first = true
        for await event in stream! {
            if replyCancelFlag != 0 { break }
            if event.isEnd { break }
            if first { await spinner.stop(); first = false }
            let visible = filter.consume(event.text)
            if !visible.isEmpty {
                assistant += visible
                print(md.consume(visible), terminator: "")
                fflush(stdout)
            }
        }
        if first { await spinner.stop() }
        let cancelled = replyCancelFlag != 0
        // Drop the stream now so its onTermination fires immediately and the
        // engine stops decoding an abandoned reply (rather than at scope exit).
        stream = nil

        let tail = filter.finish()
        if !tail.isEmpty { assistant += tail; print(md.consume(tail), terminator: "") }
        print(md.finish(), terminator: "")
        print()
        if cancelled { print(Ansi.yellow("(cancelled)")) }

        // Record the turn only when it completed normally, so a cancelled (often
        // empty) reply does not pollute the conversation context.
        if !cancelled {
            history.append((role: "user", content: userText))
            history.append((role: "assistant", content: assistant))
            if let stats = getStats() { printStatusLine(stats, images: usedImages, audio: usedAudio) }
        }
        pendingImages.removeAll(); pendingAudio = nil
        print()
    }

    private func printStatusLine(_ stats: GenerationStats, images: Int, audio: Bool) {
        var parts = [modelName]
        if images > 0 { parts.append("\(images) image\(images == 1 ? "" : "s")") }
        if audio { parts.append("audio") }
        parts.append("\(stats.generatedTokens) tok")
        parts.append(String(format: "%.0f tok/s", stats.decodeTokensPerSecond))
        let ctx = stats.promptTokens + stats.generatedTokens
        parts.append("ctx \(ctx)")
        print(Ansi.dim("  " + parts.joined(separator: Self.separator)))
    }

    // MARK: - Model switching

    private func switchModel(to name: String) async {
        let dir: URL = registry.hasModel(name) ? registry.modelPath(name) : URL(fileURLWithPath: name)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            print("Model not found: \(name). Install with: krill pull \(name)")
            return
        }
        print(Ansi.dim("Loading \(name)..."))
        let newEngine = InferenceEngine(modelDirectory: dir)
        do {
            let t = CFAbsoluteTimeGetCurrent()
            try await newEngine.load()
            engine = newEngine
            modelName = name
            print(Ansi.dim(String(format: "Switched to %@ (%.1fs). Conversation kept.", name, CFAbsoluteTimeGetCurrent() - t)))
        } catch {
            print("Failed to load \(name): \(error)")
        }
    }

    // MARK: - Attachments

    private func makeAttachment(_ kind: MediaKind, _ data: Data, name: String) -> Attachment {
        Attachment(kind: kind, data: data, name: name,
                   dims: kind == .image ? MediaAttachment.imageDimensions(data) : nil)
    }

    private func addAttachment(_ a: Attachment) {
        if a.kind == .image { pendingImages.append(a) } else { pendingAudio = a }
    }

    /// Resolve a path and attach it, routing by detected kind. Returns success.
    private func attach(path: String) -> Bool {
        switch loadMedia(path) {
        case .notFound: print("File not found: \(path)"); return false
        case .notMedia: print("Not a recognized image or audio file: \(path)"); return false
        case .unsupported(let k): print("This model cannot process \(k.rawValue) input."); return false
        case .ok(let kind, let data):
            addAttachment(makeAttachment(kind, data, name: (MediaAttachment.normalizePath(path) as NSString).lastPathComponent))
            return true
        }
    }

    private func removeAttachment(_ arg: String) {
        guard let n = Int(arg) else { print("Usage: /remove <number> (see /attach)"); return }
        let total = pendingImages.count + (pendingAudio != nil ? 1 : 0)
        guard n >= 1, n <= total else { print("No attachment \(n). \(total) attached."); return }
        if n <= pendingImages.count {
            let removed = pendingImages.remove(at: n - 1)
            print(Ansi.dim("Removed \(removed.name)."))
        } else {
            print(Ansi.dim("Removed \(pendingAudio?.name ?? "audio")."))
            pendingAudio = nil
        }
        printPending()
    }

    private func printPending() {
        let total = pendingImages.count + (pendingAudio != nil ? 1 : 0)
        if total == 0 { print(Ansi.dim("No attachments pending.")); return }
        print(Ansi.dim("Pending attachments (sent with your next message):"))
        var i = 1
        for img in pendingImages {
            let dim = img.dims.map { " \($0.width)x\($0.height)" } ?? ""
            print(Ansi.dim("  [\(i)] image  \(img.name)\(dim)  \(img.sizeLabel)"))
            i += 1
        }
        if let aud = pendingAudio {
            print(Ansi.dim("  [\(i)] audio  \(aud.name)  \(aud.sizeLabel)"))
        }
        print(Ansi.dim("  /remove <n> to drop one, /clear to drop all."))
    }

    // MARK: - Media resolution (shared by commands, bare paths, inline @path)

    private enum MediaLoadResult {
        case notFound, notMedia
        case unsupported(MediaKind)
        case ok(MediaKind, Data)
    }

    private func loadMedia(_ token: String) -> MediaLoadResult {
        let path = MediaAttachment.normalizePath(token)
        guard !path.isEmpty else { return .notFound }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), !isDir.boolValue else {
            return .notFound
        }
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return .notFound }
        let ext = (path as NSString).pathExtension
        guard let kind = MediaAttachment.detectKind(data: data, pathExtension: ext) else { return .notMedia }
        switch kind {
        case .image where !engine.supportsNativeImage: return .unsupported(.image)
        case .audio where !engine.canUseNativeAudio: return .unsupported(.audio)
        default: return .ok(kind, data)
        }
    }

    /// Pull inline `@path` references out of a prompt line, leaving non-file
    /// `@mentions` in the returned text verbatim.
    private func extractInlineMedia(_ line: String) -> (cleaned: String, attachments: [Attachment]) {
        var cleaned = ""
        var attachments: [Attachment] = []
        let chars = Array(line)
        var i = 0
        var atBoundary = true
        while i < chars.count {
            let ch = chars[i]
            if ch == "@", atBoundary {
                var j = i + 1
                var tok = ""
                while j < chars.count {
                    let c = chars[j]
                    if c == "\\", j + 1 < chars.count { tok.append(c); tok.append(chars[j + 1]); j += 2; continue }
                    if c == " " || c == "\t" { break }
                    tok.append(c); j += 1
                }
                if !tok.isEmpty {
                    switch loadMedia(tok) {
                    case .ok(let kind, let data):
                        attachments.append(makeAttachment(kind, data, name: (MediaAttachment.normalizePath(tok) as NSString).lastPathComponent))
                        i = j; atBoundary = false; continue
                    case .unsupported(let k):
                        print("This model cannot process \(k.rawValue) input (@\(tok)).")
                        i = j; atBoundary = false; continue
                    case .notFound, .notMedia:
                        break
                    }
                }
            }
            cleaned.append(ch)
            atBoundary = (ch == " " || ch == "\t")
            i += 1
        }
        return (cleaned, attachments)
    }

    private func recordFromMic() async -> Data? {
        guard await MicrophoneRecorder.requestAccess() else {
            print(MicrophoneCaptureError.permissionDenied.description)
            return nil
        }
        let recorder = MicrophoneRecorder()
        do { try recorder.start() } catch { print("\(error)"); return nil }
        print(Ansi.yellow("Recording... press Enter to stop."))
        _ = LineEditor.readLine(prompt: "")
        do {
            let wav = try recorder.stop()
            print(Ansi.dim(String(format: "Captured %.1fs of audio.", recorder.capturedSeconds)))
            return wav
        } catch { print("\(error)"); return nil }
    }

    // MARK: - Transcript / history / help

    private func printHistory() {
        if history.isEmpty { print(Ansi.dim("No conversation yet.")); return }
        for turn in history {
            let who = turn.role == "user" ? Ansi.green("you") : Ansi.cyan("bot")
            print("\(who): \(turn.content)")
        }
    }

    private func saveTranscript(to arg: String) {
        let path = arg.isEmpty ? "krill-transcript.txt" : MediaAttachment.normalizePath(arg)
        var text = ""
        if let system, !system.isEmpty { text += "system: \(system)\n\n" }
        for turn in history { text += "\(turn.role): \(turn.content)\n\n" }
        do {
            try text.write(toFile: path, atomically: true, encoding: .utf8)
            print(Ansi.dim("Saved \(history.count / 2) turn(s) to \(path)"))
        } catch {
            print("Could not save transcript: \(error)")
        }
    }

    private func printHelp() {
        print("""
        \(Ansi.bold("Commands"))
          \(Ansi.cyan("/image <path>"))   Attach an image to your next message (\(Ansi.cyan("/img")) alias)
          \(Ansi.cyan("/audio <path>"))   Attach an audio clip to your next message
          \(Ansi.cyan("/mic"))            Record from the microphone (Enter to stop)
          \(Ansi.cyan("/attach"))         List pending attachments
          \(Ansi.cyan("/remove <n>"))     Drop attachment number n
          \(Ansi.cyan("/clear"))          Drop all pending attachments
          \(Ansi.cyan("/system <text>"))  Set the system prompt
          \(Ansi.cyan("/model <name>"))   Switch to another model
          \(Ansi.cyan("/history"))        Show the conversation so far
          \(Ansi.cyan("/save [file]"))    Save the transcript
          \(Ansi.cyan("/reset"))          Clear the conversation
          \(Ansi.cyan("/help"))           This help
          \(Ansi.cyan("/quit"))           Exit
        \(Ansi.dim("Attach a file by dragging it into the terminal, or inline with @path."))
        \(Ansi.dim("Tab completes commands and paths; Up/Down recall history; Ctrl-C cancels a reply."))
        """)
    }
}

/// Thrown by /quit to unwind the run loop cleanly.
struct CleanExit: Error {}
