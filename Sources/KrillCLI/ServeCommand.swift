import ArgumentParser
import Foundation
import KrillEngine
import KrillCache
import KrillServer
import KrillRegistry

struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start the HTTP API server (OpenAI + Ollama compatible)"
    )

    @Option(name: .long, help: "Model to pre-load (name or path)")
    var model: String?

    @Option(name: .long, help: "Port to listen on (default: $OLLAMA_HOST port / config / 57455). 57455 is \"KRILL\" on a keypad and coexists with Ollama on 11434.")
    var port: Int?

    @Option(name: .long, help: "Host to bind to (default: $OLLAMA_HOST / config / 127.0.0.1)")
    var host: String?

    @Option(name: .long, help: "Client compat surface: ollama | openai | both (default both). Krill's default port is 57455; for a drop-in Ollama replacement pass --port 11434.")
    var compat: String = "both"

    @Option(name: .long, help: "Draft model for speculative decoding (alias, path, or 'auto'). Also reads KRILL_DRAFT_MODEL.")
    var draftModel: String?

    @Flag(name: .long, help: "Explicitly enable n-gram (prompt-lookup) speculative decode (single-stream AND concurrent batcher). On by default already; this flag just pins it on regardless of config. Disable with KRILL_NGRAM_SPEC=0 / ngram_spec=false. Wins on repetitive workloads (RAG, code, structured output).")
    var ngramSpec: Bool = false

    func run() async throws {
        // n-gram spec is on by default (single-stream + batcher, each self-gated
        // and adaptive). The flag remains as an explicit pin: set the env EVERY
        // engine reads at init before any engine is constructed, so a config
        // `ngram_spec = false` cannot disable it when the operator asked for it.
        if ngramSpec { setenv("KRILL_NGRAM_SPEC", "1", 1) }

        // Precedence (CLI flag > env > config.toml > default): KrillConfig.load()
        // already folds OLLAMA_HOST/OLLAMA_MODELS (and KRILL_* which win over
        // them) into serverHost/serverPort/modelsDir; an explicit CLI flag,
        // when present, still overrides everything.
        let config = KrillConfig.load()
        // Bridge the decode-pipeline toggle to the env the engine reads, unless an
        // explicit env value is already set (env wins over config.toml). Default
        // is on, so we only act when config disables it.
        if ProcessInfo.processInfo.environment["KRILL_DECODE_PIPELINE"] == nil,
           !config.decodePipeline {
            setenv("KRILL_DECODE_PIPELINE", "0", 1)
        }
        // Same bridge for n-gram speculative decode (default on; only act when
        // config disables it and neither the env nor the --ngram-spec flag spoke).
        if ProcessInfo.processInfo.environment["KRILL_NGRAM_SPEC"] == nil,
           !config.ngramSpec {
            setenv("KRILL_NGRAM_SPEC", "0", 1)
        }
        let host = self.host ?? config.serverHost
        let port = self.port ?? config.serverPort
        let registry: Registry
        if let md = config.modelsDir, !md.isEmpty {
            registry = Registry(modelsDir: URL(fileURLWithPath: md))
        } else {
            registry = Registry()
        }

        // One PrefixCache shared by every resident engine (decision: shared
        // by default; its keys already namespace by model id, so models never
        // read each other's prefixes, and the GB budget stays singular rather
        // than multiplying by MAX_LOADED_MODELS).
        let sharedPrefix = PrefixCache(diskBudgetGB: config.prefixCacheSizeGB,
                                       maxEntryGB: config.prefixCacheMaxEntryGB)
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
                print("Install with: krill pull \(model)")
                throw ExitCode.failure
            }

            print("Loading model from \(modelDir.path)...")
            engine = InferenceEngine(modelDirectory: modelDir,
                                     prefixCache: sharedPrefix,
                                     kvCacheDtype: config.kvCacheDtype)
            try await engine.load()
            // Pre-warm compile caches + kernel JIT so the first
            // request does not pay one-time costs. Opt-out via
            // `KRILL_SKIP_WARMUP=1`. Best-effort: warmup errors
            // never block the server from accepting requests.
            await engine.warmup()
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
                base = URL(fileURLWithPath: FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".krill").path)
            }
            engine = InferenceEngine(modelDirectory: base,
                                     prefixCache: sharedPrefix,
                                     kvCacheDtype: config.kvCacheDtype)
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

        // Stage A: an LRU pool of resident engines (MAX_LOADED_MODELS). The
        // pre-loaded `--model` engine (if any) is registered as the initial
        // active model; further models are routed-or-loaded on demand. The
        // base engine doubles as the display fallback when nothing is loaded.
        let activeRef = ActiveEngineRef()
        let engines = EngineRegistry(
            preloaded: self.model != nil ? engine : nil,
            preloadedName: self.model,
            maxLoaded: config.maxLoadedModels,
            registry: registry,
            prefixCache: sharedPrefix,
            kvCacheDtype: config.kvCacheDtype,
            defaultKeepAliveSeconds: KeepAliveParse.duration(config.keepAlive) ?? 300,
            numParallel: config.numParallel,
            activeRef: activeRef)

        let server = KrillServer(host: host, port: port, compat: compatMode,
                               engines: engines, activeRef: activeRef,
                               fallbackEngine: engine, registry: registry,
                               corsOrigins: config.origins,
                               defaultContextLimit: config.contextLength,
                               numParallel: config.numParallel,
                               maxQueue: config.maxQueue)
        try await server.start()
    }
}

