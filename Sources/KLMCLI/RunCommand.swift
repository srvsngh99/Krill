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

    @Option(name: .long, help: "Image file path for vision models")
    var image: String?

    @Option(name: .long, help: "Audio file path for audio models")
    var audio: String?

    @Option(name: .long, help: "Tools JSON file for function calling")
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

        // Load model
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

        if let prompt {
            // Single-shot mode
            try await generateAndPrint(
                engine: engine, prompt: prompt, system: system,
                params: params, maxTokens: maxTokens
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

// MARK: - Generation

private func generateAndPrint(
    engine: InferenceEngine,
    prompt: String,
    system: String?,
    params: SamplingParams,
    maxTokens: Int
) async throws {
    let (stream, getStats) = engine.generate(
        prompt: prompt,
        systemPrompt: system,
        params: params,
        maxTokens: maxTokens
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
