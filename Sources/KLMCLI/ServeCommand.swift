import ArgumentParser
import Foundation
import KLMEngine
import KLMServer
import KLMRegistry

struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start the HTTP API server (OpenAI + Ollama compatible)"
    )

    @Option(name: .long, help: "Model to pre-load (name or path)")
    var model: String?

    @Option(name: .long, help: "Port to listen on")
    var port: Int = 11435

    @Option(name: .long, help: "Host to bind to")
    var host: String = "127.0.0.1"

    @Option(name: .long, help: "Client compat surface: ollama | openai | both (default both). For an Ollama drop-in also pass --port 11434.")
    var compat: String = "both"

    func run() async throws {
        let registry = Registry()
        let config = KrillConfig.load()

        let engine: InferenceEngine

        if let model {
            // Resolve model path
            let modelDir: URL
            if registry.hasModel(model) {
                modelDir = registry.modelPath(model)
            } else if FileManager.default.fileExists(atPath: model) {
                modelDir = URL(fileURLWithPath: model)
            } else {
                print("Error: model '\(model)' not found.")
                print("Install with: krillm pull \(model)")
                throw ExitCode.failure
            }

            print("Loading model from \(modelDir.path)...")
            engine = InferenceEngine(modelDirectory: modelDir)
            try await engine.load()
            print("Model loaded.")
        } else {
            // Start without a model — callers use POST /v1/models/load to load one.
            // We pick the first installed model's directory as a placeholder base
            // (only the directory is stored; no weights are loaded until load() is called).
            let base: URL
            if let first = registry.listModels().first {
                base = registry.modelPath(first.name)
            } else {
                base = URL(fileURLWithPath: FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".krillm").path)
            }
            engine = InferenceEngine(modelDirectory: base)
            print("Server starting in API-only mode (no model pre-loaded).")
            print("Load a model via:  POST http://\(host):\(port)/v1/models/load  {\"model\": \"<name>\"}")
            let installed = registry.listModels().map(\.name)
            if !installed.isEmpty {
                print("Installed models: \(installed.joined(separator: ", "))")
            }
        }

        guard let compatMode = CompatMode(label: compat) else {
            print("Error: invalid --compat '\(compat)'. Use: ollama | openai | both")
            throw ExitCode.failure
        }

        let server = KLMServer(host: host, port: port, compat: compatMode,
                               engine: engine, registry: registry,
                               corsOrigins: config.origins,
                               keepAliveDefaultSeconds:
                                KeepAliveParse.duration(config.keepAlive) ?? 300,
                               defaultContextLimit: config.contextLength)
        try await server.start()
    }
}
