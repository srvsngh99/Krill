import ArgumentParser
import Foundation
import KLMEngine
import KLMCore
import KLMSampler
import KLMRegistry

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

    @Option(name: .long, help: "Image file path for Gemma 4 via mlx-vlm")
    var image: String?

    @Option(name: .long, help: "Audio file path for Gemma 4 via mlx-vlm")
    var audio: String?

    @Option(name: .long, help: "Tools JSON file for function calling (not yet supported)")
    var tools: String?

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

        // Check if this is a Gemma 4 model
        let configURL = modelDir.appendingPathComponent("config.json")
        let isGemma4 = {
            guard let configData = try? Data(contentsOf: configURL),
                  let configJSON = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
                return false
            }
            return isGemma4Config(configJSON)
        }()

        if isGemma4 && (image != nil || audio != nil) {
            // Native multimodal path for Gemma 4
            print("Loading Gemma 4 (native multimodal)...")
        } else if isGemma4 {
            print("Loading Gemma 4 (native)...")
        }

        // Image/audio are only supported for Gemma 4
        if (image != nil || audio != nil) && !isGemma4 {
            print("Error: --image/--audio is only supported for Gemma 4 models.")
            throw ExitCode.failure
        }

        // Load model (native Swift path for Llama, Qwen, Mistral, etc.)
        print("Loading model from \(modelPath)...")
        let engine = InferenceEngine(modelDirectory: modelDir)

        let loadStart = CFAbsoluteTimeGetCurrent()
        try await engine.load()
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        print(String(format: "Ready (%.1fs load time)", loadTime))

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

    // Stream tokens to stdout
    for await event in stream {
        if event.isEnd { break }
        print(event.text, terminator: "")
        fflush(stdout)
    }
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
}
