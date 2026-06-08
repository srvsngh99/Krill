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

    @Option(name: .long, help: "Image file path for Gemma 4 (native vision)")
    var image: String?

    @Option(name: .long, help: "Audio file path for Gemma 4 (native USM; wav/mp3/flac/ogg/m4a)")
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

        // Check if this is a Gemma 4 model
        let configURL = modelDir.appendingPathComponent("config.json")
        if let configData = try? Data(contentsOf: configURL),
           let configJSON = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
           isGemma4Config(configJSON) {
            // Gemma 4 text, image, and audio all run on the native Swift
            // engine below (the mlx-vlm bridge was retired in WS6 Step 4).
        }

        // Image/audio only supported for Gemma 4
        let configJSON2 = (try? Data(contentsOf: configURL))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        let isGemma4Model = configJSON2.map(isGemma4Config) ?? false
        if image != nil && !isGemma4Model {
            print("Error: --image is only supported for Gemma 4 models.")
            throw ExitCode.failure
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

        // The mlx-vlm bridge was removed (WS6 Step 4). Reject --audio for any
        // model that cannot process it natively — including text-only Gemma 4
        // checkpoints — so it is a hard error, never a silent text-only run.
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
            // Interactive REPL
            try await interactiveMode(
                engine: engine, system: system,
                params: params, maxTokens: maxTokens
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

private func isGemma4Config(_ configJSON: [String: Any]) -> Bool {
    let modelType = configJSON["model_type"] as? String
    let architectures = configJSON["architectures"] as? [String] ?? []
    return modelType == "gemma4"
        || modelType == "gemma4_text"
        || architectures.contains { $0.lowercased().contains("gemma4") }
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
    audioData: Data? = nil
) async throws {
    let (stream, getStats) = engine.generate(
        prompt: prompt,
        systemPrompt: system,
        params: params,
        maxTokens: maxTokens,
        imageData: imageData,
        audioData: audioData
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
    maxTokens: Int
) async throws {
    print("\nKrillLM Interactive Mode")
    print("Type your message and press Enter. Type /quit to exit.\n")

    while true {
        print("> ", terminator: "")
        fflush(stdout)

        guard let line = readLine(), !line.isEmpty else {
            continue
        }

        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed == "/quit" || trimmed == "/exit" || trimmed == "/q" {
            print("Goodbye.")
            break
        }

        try await generateAndPrint(
            engine: engine, prompt: trimmed, system: system,
            params: params, maxTokens: maxTokens
        )
        print()
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
