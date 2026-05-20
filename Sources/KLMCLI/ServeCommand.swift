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

    @Option(name: .long, help: "Port to listen on (default: $OLLAMA_HOST port / config / 11435)")
    var port: Int?

    @Option(name: .long, help: "Host to bind to (default: $OLLAMA_HOST / config / 127.0.0.1)")
    var host: String?

    @Option(name: .long, help: "Client compat surface: ollama | openai | both (default both). For an Ollama drop-in also pass --port 11434.")
    var compat: String = "both"

    @Option(name: .long, help: "Draft model for speculative decoding (alias, path, or 'auto'). Also reads KRILL_DRAFT_MODEL.")
    var draftModel: String?

    func run() async throws {
        // Precedence (CLI flag > env > config.toml > default): KrillConfig.load()
        // already folds OLLAMA_HOST/OLLAMA_MODELS (and KRILL_* which win over
        // them) into serverHost/serverPort/modelsDir; an explicit CLI flag,
        // when present, still overrides everything.
        let config = KrillConfig.load()
        let host = self.host ?? config.serverHost
        let port = self.port ?? config.serverPort
        let registry: Registry
        if let md = config.modelsDir, !md.isEmpty {
            registry = Registry(modelsDir: URL(fileURLWithPath: md))
        } else {
            registry = Registry()
        }

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

            let draftSpec = self.draftModel
                ?? ProcessInfo.processInfo.environment["KRILL_DRAFT_MODEL"]
            if let spec = draftSpec, !spec.isEmpty {
                do {
                    try DraftModelResolver.load(
                        draftSpec: spec, target: model,
                        registry: registry, engine: engine)
                } catch {
                    print("warning: draft model '\(spec)' did not load: \(error). Continuing without speculative decoding.")
                }
            }
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

        // Bridge-backed sidecars (VLM + MoE). Both are lazy-loaded
        // on first request; instantiated up front so a signal
        // handler can shut them down on SIGINT (otherwise Python
        // children become orphans on Ctrl+C, holding the GPU
        // and the loaded model weights resident).
        let vlmEngine = Qwen25VLEngine()
        let moeEngine = MoEEngine()
        installVLMSidecarSignalHandler(vlmEngine, moeEngine)

        let server = KLMServer(host: host, port: port, compat: compatMode,
                               engine: engine, registry: registry,
                               vlmEngine: vlmEngine,
                               moeEngine: moeEngine,
                               corsOrigins: config.origins,
                               keepAliveDefaultSeconds:
                                KeepAliveParse.duration(config.keepAlive) ?? 300,
                               defaultContextLimit: config.contextLength,
                               numParallel: config.numParallel,
                               maxQueue: config.maxQueue)
        try await server.start()
    }
}

/// Hook SIGINT / SIGTERM so the VLM Python sidecar (if any) is
/// terminated before the krillm process exits. Without this the
/// child Python process becomes an orphan on Ctrl+C, holding the
/// GPU and the mlx-vlm-loaded model in memory until manually
/// killed. The sidecar's own stdin EOF would also tear it down,
/// but only AFTER mlx-vlm's blocking generate returns - which can
/// be many seconds for a large prompt.
///
/// Uses DispatchSource so the handler runs on a dedicated queue
/// (signal handlers cannot acquire NSLock safely from the signal
/// context). The default SIGINT behavior (terminate the process)
/// is preserved via `exit(0)` after shutdown.
private nonisolated(unsafe) var vlmShutdownHandler: (() -> Void)?

private func installVLMSidecarSignalHandler(
    _ vlm: Qwen25VLEngine, _ moe: MoEEngine
) {
    vlmShutdownHandler = { [weak vlm, weak moe] in
        try? vlm?.shutdown()
        try? moe?.shutdown()
    }
    for sig in [SIGINT, SIGTERM] {
        signal(sig, SIG_IGN)
        let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
        src.setEventHandler {
            vlmShutdownHandler?()
            exit(0)
        }
        src.resume()
        // Hold the source for the lifetime of the process via a
        // singleton box; otherwise DispatchSource cancels itself
        // when the local goes out of scope.
        VLMSignalSourceBox.shared.sources.append(src)
    }
}

/// Singleton box that retains the DispatchSource instances for the
/// lifetime of the process. DispatchSource cancels itself when
/// released, so we cannot let the locals fall out of scope at the
/// end of `installVLMSidecarSignalHandler`.
private final class VLMSignalSourceBox: @unchecked Sendable {
    static let shared = VLMSignalSourceBox()
    var sources: [DispatchSourceSignal] = []
    private init() {}
}
