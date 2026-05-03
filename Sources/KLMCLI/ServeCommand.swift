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

    func run() async throws {
        let registry = Registry()

        // Resolve model path
        let modelDir: URL
        if let model {
            // Check if it's an installed model name
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
            let engine = InferenceEngine(modelDirectory: modelDir)
            try await engine.load()
            print("Model loaded.")

            let server = KLMServer(host: host, port: port, engine: engine, registry: registry)
            try await server.start()
        } else {
            print("Starting server without pre-loaded model.")
            print("Models will be loaded on first request.")
            // Create engine with a placeholder - in production we'd have lazy loading
            let engine = InferenceEngine(modelDirectory: URL(fileURLWithPath: "/tmp"))
            let server = KLMServer(host: host, port: port, engine: engine, registry: registry)
            try await server.start()
        }
    }
}
