import ArgumentParser
import Foundation
import KLMEngine
import KLMCore
import KLMSampler
import KLMRegistry
import KLMServer

struct RunCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Load a model and run interactive chat or single-shot generation"
    )

    @Argument(help: "Model name (from registry) or path to model directory")
    var modelPath: String

    @Argument(help: "Optional prompt for single-shot mode (omit for interactive REPL)")
    var prompt: String?

    @Option(name: .long, help: "Sampling temperature (0 = greedy)")
    var temp: Float = 0.0

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 512

    @Option(name: .long, help: "Top-p (nucleus) sampling threshold")
    var topP: Float = 1.0

    @Option(name: .long, help: "Random seed for reproducible generation")
    var seed: UInt64?

    @Option(name: .long, help: "System prompt")
    var system: String?

    @Option(name: .long, help: "Image file path for vision-capable models (Gemma 4, Qwen2.5-VL, LLaVA, mllama). In interactive mode, attach with /image, a dragged path, or @path.")
    var image: String?

    @Option(name: .long, help: "Audio file path for audio-capable Gemma 4 (native USM; wav/mp3/flac/ogg/m4a). In interactive mode, attach with /audio, /mic, a dragged path, or @path.")
    var audio: String?

    @Option(name: .long, help: "Tools JSON file for function calling (not yet supported)")
    var tools: String?

    @Option(name: .long, help: "Draft model for speculative decoding. Pass an alias, a path, or 'auto' to use the curated pair from draftPairs.")
    var draftModel: String?

    func run() async throws {
        let modelDir: URL

        // Resolve: check registry first, then treat as file path
        let registry = Registry()
        if registry.hasModel(modelPath) {
            modelDir = registry.modelPath(modelPath)
        } else {
            modelDir = URL(fileURLWithPath: modelPath)
        }

        // Validate directory exists
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("Error: model '\(modelPath)' not found.")
            print("Install with: krillm pull \(modelPath)")
            print("Or provide a full path to a model directory.")
            throw ExitCode.failure
        }

        // Validate unsupported flags early
        if tools != nil {
            print("Error: --tools is not yet supported. Tool definitions are not loaded, sent to the model, or executed by krillm run yet.")
            throw ExitCode.failure
        }

        if let image {
            try validateInputFile(image, flagName: "--image")
        }
        if let audio {
            try validateInputFile(audio, flagName: "--audio")
        }

        // Detect-only daemon mode (ladder rung 1): if a krillm serve is
        // already up on the configured port AND has this model loaded
        // AND the request is text-only single-shot with no draft model,
        // route through HTTP and skip the per-call model load entirely.
        // A failed probe (default 200 ms timeout) falls through silently
        // to the in-process path. Set KRILL_NO_AUTO_DAEMON=1 to disable.
        //
        // Restrict daemon routing to registry-alias invocations:
        // comparing a filesystem-path argument against the daemon's
        // alias string is unreliable (a path's last component could
        // collide with a loaded alias name, and the daemon would then
        // reject the path-form `model` field with HTTP 400).
        //
        // Skip the route if the alias has Modelfile overrides: the
        // server's /v1/chat/completions path applies
        // applyModelSystemOverride / applyModelParams (so SYSTEM /
        // PARAMETER from the Modelfile take effect), but the
        // in-process krillm run path below does not. Routing only
        // when both paths would produce identical behaviour keeps the
        // optimisation observability-free.
        let aliasHasOverrides = registry.getModel(modelPath)?.overrides != nil
        if let prompt,
           image == nil, audio == nil, draftModel == nil,
           registry.hasModel(modelPath),
           !aliasHasOverrides,
           ProcessInfo.processInfo.environment["KRILL_NO_AUTO_DAEMON"] != "1" {
            if try await tryDaemonRoute(modelName: modelPath, prompt: prompt) {
                return
            }
        }

        // Load model (native Swift path for Llama, Qwen, Mistral, etc.)
        print("Loading model from \(modelPath)...")
        let engine = InferenceEngine(modelDirectory: modelDir)

        let loadStart = CFAbsoluteTimeGetCurrent()
        try await engine.load()
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        print(String(format: "Ready (%.1fs load time)", loadTime))

        if let draftSpec = draftModel {
            try DraftModelResolver.load(
                draftSpec: draftSpec, target: modelPath,
                registry: registry, engine: engine)
        }

        // Native vision/audio runtimes exist for several families (Gemma 4,
        // Qwen2.5-VL, LLaVA, mllama). Gate on the loaded model's actual
        // capability rather than a model-name allowlist, so `--image` works for
        // every vision-capable family and fails loudly for text-only ones —
        // never a silent text-only run. (The mlx-vlm bridge was removed in WS6.)
        if image != nil && !engine.supportsNativeImage {
            print("Error: --image requires a vision-capable model; this model cannot process images.")
            throw ExitCode.failure
        }
        if audio != nil && !engine.canUseNativeAudio {
            print("Error: --audio requires an audio-capable Gemma 4 checkpoint; this model cannot process audio.")
            throw ExitCode.failure
        }

        let params = SamplingParams(
            temperature: temp,
            topP: topP,
            seed: seed
        )

        // Load image/audio data if provided
        var imageData: Data?
        var audioData: Data?
        if let image {
            imageData = try Data(contentsOf: URL(fileURLWithPath: image))
        }
        if let audio {
            audioData = try Data(contentsOf: URL(fileURLWithPath: audio))
        }

        if let prompt {
            // Single-shot mode
            try await generateAndPrint(
                engine: engine, prompt: prompt, system: system,
                params: params, maxTokens: maxTokens,
                imageData: imageData, audioData: audioData
            )
        } else {
            // Interactive REPL. Any media passed via --image/--audio is
            // pre-attached to the first turn; further media can be attached
            // mid-conversation (see interactiveMode).
            try await interactiveMode(
                engine: engine, system: system,
                params: params, maxTokens: maxTokens,
                initialImage: imageData, initialAudio: audioData
            )
        }
    }

    /// Probe a locally running daemon and, if it has `modelName`
    /// loaded, stream the chat through it. Returns true when the
    /// request was served by the daemon, false when the probe
    /// did not match and the caller should fall through to the
    /// in-process path. A mid-stream failure throws so the user
    /// gets a clear error instead of a silent partial output.
    private func tryDaemonRoute(modelName: String, prompt: String) async throws -> Bool {
        let port = Int(ProcessInfo.processInfo.environment["KRILL_PORT"] ?? "") ?? 57455
        guard let status = await DaemonClient.probeStatus(port: port) else { return false }
        guard status.modelLoaded, status.model == modelName else { return false }

        var messages: [(role: String, content: String)] = []
        if let system, !system.isEmpty {
            messages.append((role: "system", content: system))
        }
        messages.append((role: "user", content: prompt))

        let result = try await DaemonClient.streamChat(
            port: port,
            model: modelName,
            messages: messages,
            temperature: temp,
            topP: topP,
            maxTokens: maxTokens,
            seed: seed,
            onToken: { token in
                print(token, terminator: "")
                fflush(stdout)
            }
        )
        print()
        // `contentChunkCount` is non-empty SSE delta chunks, not true
        // tokens (the server's StreamingReasoningFilter buffers and may
        // drop chunks around `<think>` blocks, and a single chunk may
        // carry multiple tokens). Label it as such instead of misnaming
        // it "tokens", which would silently disagree with the
        // in-process stats line above.
        print(String(
            format: "--- (via daemon @ :%d) %d chunks, wall %.2fs",
            port, result.contentChunkCount, result.wallTimeSec
        ))
        return true
    }
}

private func validateInputFile(_ path: String, flagName: String) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), !isDirectory.boolValue else {
        print("Error: \(flagName) file not found: \(path)")
        throw ExitCode.failure
    }
}

// MARK: - Generation

private func generateAndPrint(
    engine: InferenceEngine,
    prompt: String,
    system: String?,
    params: SamplingParams,
    maxTokens: Int,
    imageData: Data? = nil,
    audioData: Data? = nil,
    imagesData: [Data] = []
) async throws {
    let (stream, getStats) = engine.generate(
        prompt: prompt,
        systemPrompt: system,
        params: params,
        maxTokens: maxTokens,
        imageData: imageData,
        audioData: audioData,
        imagesData: imagesData
    )

    // Stream tokens to stdout, filtering reasoning blocks the same way the
    // server does so CLI output never leaks `<think>` / Gemma 4 `<|channel>`
    // markers (or their inner reasoning) into the visible answer.
    let reasoningFilter = StreamingReasoningFilter()
    for await event in stream {
        if event.isEnd { break }
        let visible = reasoningFilter.consume(event.text)
        if !visible.isEmpty {
            print(visible, terminator: "")
            fflush(stdout)
        }
    }
    let tail = reasoningFilter.finish()
    if !tail.isEmpty { print(tail, terminator: "") }
    print() // Final newline

    // Print stats
    if let stats = getStats() {
        printStats(stats)
    }
}

// MARK: - Interactive REPL

private func interactiveMode(
    engine: InferenceEngine,
    system: String?,
    params: SamplingParams,
    maxTokens: Int,
    initialImage: Data? = nil,
    initialAudio: Data? = nil
) async throws {
    print("\nKrillLM Interactive Mode")
    print("Type your message and press Enter. Type /help for commands, /quit to exit.\n")

    // Pending attachments apply to the NEXT user message, then clear. Images
    // accumulate (multi-image models consume all; single-image models use the
    // first); audio holds the most recent clip.
    var pendingImages: [Data] = []
    if let initialImage { pendingImages.append(initialImage) }
    var pendingAudio: Data? = initialAudio
    if !pendingImages.isEmpty || pendingAudio != nil {
        printPending(pendingImages, pendingAudio)
    }

    while true {
        print("> ", terminator: "")
        fflush(stdout)

        guard let line = readLine() else {
            print()           // EOF (Ctrl-D): finish the prompt line
            break
        }
        if line.isEmpty { continue }
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { continue }

        // Slash commands. A "/"-prefixed line is only a command when its first
        // token matches a known command; otherwise it may be an absolute path
        // (e.g. a dragged "/Users/me/cat.png"), handled as media just below.
        let parts = trimmed.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
        let firstToken = String(parts.first ?? "")
        let arg = parts.count > 1 ? String(parts[1]).trimmingCharacters(in: .whitespaces) : ""
        let knownCommands: Set<String> = [
            "/quit", "/exit", "/q", "/help", "/?", "/clear", "/attach",
            "/image", "/img", "/audio", "/mic",
        ]
        if knownCommands.contains(firstToken) {
            switch firstToken {
            case "/quit", "/exit", "/q":
                print("Goodbye.")
                return
            case "/help", "/?":
                printREPLHelp()
            case "/clear":
                pendingImages.removeAll(); pendingAudio = nil
                print("Cleared pending attachments.")
            case "/attach":
                printPending(pendingImages, pendingAudio)
            case "/image", "/img", "/audio":
                if arg.isEmpty {
                    print("Usage: \(firstToken) <path>")
                } else if attachToken(arg, engine: engine, images: &pendingImages, audio: &pendingAudio) {
                    printPending(pendingImages, pendingAudio)
                }
            case "/mic":
                if !engine.canUseNativeAudio {
                    print("This model cannot process audio; /mic is unavailable.")
                } else if let wav = await recordFromMic() {
                    pendingAudio = wav
                    printPending(pendingImages, pendingAudio)
                }
            default:
                break
            }
            continue
        }

        // Bare dropped/typed path: the whole line resolves to a media file.
        switch loadMedia(trimmed, engine: engine) {
        case .ok(.image, let d):
            pendingImages.append(d); printPending(pendingImages, pendingAudio); continue
        case .ok(.audio, let d):
            pendingAudio = d; printPending(pendingImages, pendingAudio); continue
        case .unsupported(let k):
            print("This model cannot process \(k.rawValue) input."); continue
        case .notFound, .notMedia:
            break   // not a media path — fall through
        }

        // A "/word" line that is neither a known command nor an existing path is
        // a mistyped command, not a prompt to send to the model.
        if firstToken.hasPrefix("/"),
           String(firstToken.dropFirst()).allSatisfy({ $0.isLetter || $0 == "?" }) {
            print("Unknown command \(firstToken). Type /help for the list.")
            continue
        }

        // Inline @path references embedded in the prompt text.
        let (cleaned, inlineImages, inlineAudio) = extractInlineMedia(trimmed, engine: engine)
        pendingImages.append(contentsOf: inlineImages)
        if let inlineAudio { pendingAudio = inlineAudio }
        let promptText = cleaned.trimmingCharacters(in: .whitespaces)

        if promptText.isEmpty {
            // Line carried only attachments; keep them pending for the next turn.
            if !inlineImages.isEmpty || inlineAudio != nil {
                printPending(pendingImages, pendingAudio)
            }
            continue
        }

        // Generate, consuming the pending attachments. Pass both the first
        // image (single-image runtimes) and the full list (mllama multi-image),
        // mirroring the server's loadImages contract.
        let imgs = pendingImages
        try await generateAndPrint(
            engine: engine, prompt: promptText, system: system,
            params: params, maxTokens: maxTokens,
            imageData: imgs.first, audioData: pendingAudio, imagesData: imgs
        )
        print()
        pendingImages.removeAll(); pendingAudio = nil
    }
}

// MARK: - Interactive media attachment

/// Outcome of resolving a path token to attachable media.
private enum MediaLoadResult {
    case notFound          // path does not resolve to an existing file
    case notMedia          // file exists but is not a recognized image/audio
    case unsupported(MediaKind)  // recognized media the loaded model can't process
    case ok(MediaKind, Data)
}

/// Resolve a path token (raw from the terminal) to attachable media, applying
/// path normalization and the loaded model's capability gate.
private func loadMedia(_ token: String, engine: InferenceEngine) -> MediaLoadResult {
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

/// Load `token` and route it into the pending image/audio state by detected
/// kind, printing a reason on failure. Returns true when something was attached.
private func attachToken(_ token: String, engine: InferenceEngine,
                         images: inout [Data], audio: inout Data?) -> Bool {
    switch loadMedia(token, engine: engine) {
    case .notFound:
        print("File not found: \(token)"); return false
    case .notMedia:
        print("Not a recognized image or audio file: \(token)"); return false
    case .unsupported(let k):
        print("This model cannot process \(k.rawValue) input."); return false
    case .ok(.image, let d):
        images.append(d); return true
    case .ok(.audio, let d):
        audio = d; return true
    }
}

/// Pull inline `@path` references out of a prompt line. A token that does not
/// resolve to a media file (e.g. an `@mention`) is left in the returned text
/// verbatim, so only real attachments are stripped.
private func extractInlineMedia(_ line: String, engine: InferenceEngine)
    -> (cleaned: String, images: [Data], audio: Data?) {
    var cleaned = ""
    var images: [Data] = []
    var audio: Data?
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
                switch loadMedia(tok, engine: engine) {
                case .ok(.image, let d): images.append(d); i = j; atBoundary = false; continue
                case .ok(.audio, let d): audio = d; i = j; atBoundary = false; continue
                case .unsupported(let k):
                    print("This model cannot process \(k.rawValue) input (@\(tok)).")
                    i = j; atBoundary = false; continue
                case .notFound, .notMedia:
                    break   // not media — fall through and keep the '@' literally
                }
            }
        }
        cleaned.append(ch)
        atBoundary = (ch == " " || ch == "\t")
        i += 1
    }
    return (cleaned, images, audio)
}

/// Record from the microphone until the user presses Enter; returns WAV bytes.
private func recordFromMic() async -> Data? {
    guard await MicrophoneRecorder.requestAccess() else {
        print(MicrophoneCaptureError.permissionDenied.description)
        return nil
    }
    let recorder = MicrophoneRecorder()
    do {
        try recorder.start()
    } catch {
        print("\(error)")
        return nil
    }
    print("🎙  Recording… press Enter to stop.")
    _ = readLine()
    do {
        let wav = try recorder.stop()
        print(String(format: "Captured %.1fs of audio.", recorder.capturedSeconds))
        return wav
    } catch {
        print("\(error)")
        return nil
    }
}

private func printPending(_ images: [Data], _ audio: Data?) {
    var parts: [String] = []
    if !images.isEmpty { parts.append("\(images.count) image\(images.count == 1 ? "" : "s")") }
    if audio != nil { parts.append("audio") }
    if parts.isEmpty {
        print("No attachments pending.")
    } else {
        print("Attached (applies to your next message): \(parts.joined(separator: ", ")). /clear to discard.")
    }
}

private func printREPLHelp() {
    print("""
    Commands:
      /image <path>   Attach an image to your next message
      /audio <path>   Attach an audio clip to your next message
      /mic            Record from the microphone (press Enter to stop)
      /attach         Show pending attachments
      /clear          Discard pending attachments
      /help           Show this help
      /quit           Exit
    Tips: drag a file into the terminal to attach it, or reference one inline
    with @path (e.g. "describe @~/Pictures/cat.png").
    """)
}

// MARK: - Stats Display

private func printStats(_ stats: GenerationStats) {
    let prefillTps = String(format: "%.1f", stats.prefillTokensPerSecond)
    let decodeTps = String(format: "%.1f", stats.decodeTokensPerSecond)
    let ttft = String(format: "%.0f", stats.ttft * 1000)
    let total = String(format: "%.2f", stats.totalTime)

    print("""
    ---
    prompt: \(stats.promptTokens) tokens, prefill: \(prefillTps) tok/s, \
    decode: \(stats.generatedTokens) tokens at \(decodeTps) tok/s, \
    TTFT: \(ttft)ms, total: \(total)s
    """)

    if let spec = stats.speculative {
        let rate = String(format: "%.2f", spec.acceptanceRate)
        print("spec: rounds=\(spec.rounds), accepted=\(spec.acceptedTokens), final_k=\(spec.finalK), acceptance=\(rate)")
    }

    if let moe = stats.moe, moe.sparseLayers > 0 {
        let pct = String(format: "%.0f", moe.utilizationRatio * 100)
        print("moe: \(moe.activeExpertSlots)/\(moe.totalExpertSlots) expert slots active (\(pct)%), \(moe.sparseLayers) sparse layers, peak slot load=\(moe.maxExpertLoad)")
    }
}
