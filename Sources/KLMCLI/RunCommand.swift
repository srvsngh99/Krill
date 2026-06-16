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

    @Argument(help: "Model name (from registry) or path to model directory. Optional: falls back to default_model in ~/.krillm/config.toml (or KRILL_DEFAULT_MODEL).")
    var modelPath: String?

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

    @Flag(name: .long, help: "Use the classic line REPL instead of the full-screen TUI")
    var classic = false

    @Option(name: .long, help: "TUI color theme: light, dark, or auto (default; detects via COLORFGBG / OSC 11). Also settable with KRILL_TUI_THEME.")
    var theme: String?

    @Option(name: .long, help: "Tools JSON file for function calling (not yet supported)")
    var tools: String?

    @Option(name: .long, help: "Draft model for speculative decoding. Pass an alias, a path, or 'auto' to use the curated pair from draftPairs.")
    var draftModel: String?

    func run() async throws {
        let registry = Registry()

        // Resolve the model: explicit argument, else the configured default
        // (config.toml default_model / KRILL_DEFAULT_MODEL). A blank value
        // counts as unset.
        func nonEmpty(_ s: String?) -> String? {
            guard let s, !s.trimmingCharacters(in: .whitespaces).isEmpty else { return nil }
            return s
        }
        let defaultModel = nonEmpty(KrillConfig.load().defaultModel)

        // Disambiguate the leading positional. `krillm run <model> [prompt]` is
        // the canonical form, but with a default model set, `krillm run "<text>"`
        // should run the default on that text rather than mistake the prompt for
        // a model name. So when the sole positional is clearly NOT a model (not a
        // known alias, installed model, or path) and a default exists, treat it
        // as the prompt.
        var resolvedModel = nonEmpty(modelPath)
        var prompt = self.prompt
        if let positional = resolvedModel, prompt == nil, let def = defaultModel,
           !looksLikeModelRef(positional, registry) {
            resolvedModel = def
            prompt = positional
        }
        guard let model = resolvedModel ?? defaultModel else {
            printNoModelError(registry)
            throw ExitCode.failure
        }

        // Resolve: check registry first, then treat as file path
        let modelDir: URL
        if registry.hasModel(model) {
            modelDir = registry.modelPath(model)
        } else {
            modelDir = URL(fileURLWithPath: model)
        }

        // Validate directory exists
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("Error: model '\(model)' not found.")
            print("Install with: krillm pull \(model)")
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
        let aliasHasOverrides = registry.getModel(model)?.overrides != nil
        if let prompt,
           image == nil, audio == nil, draftModel == nil,
           registry.hasModel(model),
           !aliasHasOverrides,
           ProcessInfo.processInfo.environment["KRILL_NO_AUTO_DAEMON"] != "1" {
            if try await tryDaemonRoute(modelName: model, prompt: prompt) {
                return
            }
        }

        // Load model (native Swift path for Llama, Qwen, Mistral, etc.)
        print("Loading model from \(model)...")
        let engine = InferenceEngine(modelDirectory: modelDir)

        let loadStart = CFAbsoluteTimeGetCurrent()
        try await engine.load()
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        print(String(format: "Ready (%.1fs load time)", loadTime))

        if let draftSpec = draftModel {
            try DraftModelResolver.load(
                draftSpec: draftSpec, target: model,
                registry: registry, engine: engine)
        }

        // Native vision/audio runtimes exist for several families (Gemma 4,
        // Qwen2.5-VL, LLaVA, mllama). Gate on the loaded model's actual
        // capability rather than a model-name allowlist, so `--image` works for
        // every vision-capable family and fails loudly for text-only ones -
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
        } else if RawTerminal.isInteractive && !classic {
            // Full-screen opencode-style chat TUI (Sourav AI Labs identity):
            // branded masthead, scrollable pane, slash-command autosuggest with
            // Up/Down cycling, status footer. Any media passed via
            // --image/--audio pre-attaches to the first turn.
            let tuiConfig = KrillConfig.load()
            let tui = ChatTUI(
                engine: engine, modelName: model, system: system,
                params: params, maxTokens: maxTokens, registry: registry,
                initialImage: imageData, initialAudio: audioData, theme: theme,
                voiceModeSetting: tuiConfig.voiceMode,
                speakRepliesSetting: tuiConfig.speakReplies)
            await tui.run()
        } else {
            // Classic line REPL (forced with --classic, or auto when stdout is
            // not a TTY, e.g. piped/redirected): multi-turn memory, libedit
            // editing/history/tab-completion, streamed markdown, media attach.
            let session = InteractiveSession(
                engine: engine, modelName: model, system: system,
                params: params, maxTokens: maxTokens, registry: registry,
                initialImage: imageData, initialAudio: audioData)
            do {
                try await session.run()
            } catch is CleanExit {
                // user typed /quit
            }
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

    /// True when `s` denotes a model rather than a prompt: an installed model, a
    /// known alias, an HF repo path (`org/model`), or an existing local path.
    private func looksLikeModelRef(_ s: String, _ registry: Registry) -> Bool {
        registry.hasModel(s)
            || AliasMap.resolve(s) != nil
            || s.contains("/")
            || FileManager.default.fileExists(atPath: s)
    }

    /// Guidance when `krillm run` is invoked with no model and no configured
    /// default. On a fresh install (nothing installed) this is the branded
    /// first-run welcome; otherwise it lists installed models and how to set a
    /// default.
    private func printNoModelError(_ registry: Registry) {
        let installed = registry.listModels().map { $0.name }.sorted()
        guard !installed.isEmpty else { printWelcome(); return }
        print("No model specified, and no default is set.\n")
        print("Installed models:")
        for name in installed { print("  \(name)") }
        print("\nRun one:        krillm run \(installed[0])")
        print("Set a default:  echo 'default_model = \"\(installed[0])\"' >> ~/.krillm/config.toml")
        print("            or:  export KRILL_DEFAULT_MODEL=\(installed[0])")
    }

    /// Branded first-run welcome (fresh install, no models yet), in the Sourav AI
    /// Labs identity. Plain stdout (not the alt-screen TUI), styling auto-disabled
    /// when not a TTY / under NO_COLOR.
    private func printWelcome() {
        print("")
        print("  " + Ansi.bold(Brand.wordmark) + "  " + Ansi.dim(Brand.lab))
        print("  " + Ansi.dim(Brand.tagline))
        print("")
        print("  Get started:")
        print("    krillm pull gemma-4-e2b      " + Ansi.dim("# a small, fast model to begin"))
        print("    krillm run gemma-4-e2b       " + Ansi.dim("# open the chat"))
        print("")
        print("  Browse models:   " + Ansi.dim("krillm catalog"))
        print("  Set a default:   " + Ansi.dim("default_model in ~/.krillm/config.toml"))
        print("")
        print("  " + Ansi.bold(Brand.labMark) + "  " + Ansi.dim("\(Brand.labTagline)  \u{00B7}  \(Brand.site)"))
        print("")
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
