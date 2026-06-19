import Foundation

/// Krill configuration loaded from ~/.krill/config.toml.
///
/// Precedence: CLI flags > environment variables (KRILL_*) > config.toml > defaults.
public struct KrillConfig: Sendable {
    /// Default model to use when none specified.
    public var defaultModel: String?

    /// Default quantization for new conversions.
    public var defaultQuant: Int

    /// KV cache dtype: "fp16" (default) or "int8" (quantized).
    public var kvCacheDtype: String

    /// Maximum size in GB for prefix cache memory tier.
    public var prefixCacheSizeGB: Double

    /// Per-entry KV cap in GB for the prefix cache: a prefill whose KV state
    /// exceeds this is not cached (skips memory + disk), so one huge
    /// full-attention prefix cannot spike memory into swap. `<= 0` disables the
    /// cap. `KRILL_PREFIX_CACHE_MAX_ENTRY_GB`.
    public var prefixCacheMaxEntryGB: Double

    /// Enable speculative decoding by default.
    public var speculativeDecoding: Bool

    /// Overlap CPU forward-graph construction with GPU compute on the single
    /// stream greedy decode path (the +13% double-buffer; byte-identical output).
    /// On by default; set `decode_pipeline = false` to fall back to the serial
    /// loop. Bridged to `KRILL_DECODE_PIPELINE` by the CLI.
    public var decodePipeline: Bool

    /// N-gram (prompt-lookup) speculative decode on the single stream. On by
    /// default with a per-generation stall monitor that hands off to the plain
    /// pipeline loop on non-echo workloads, so the floor holds while echo-heavy
    /// workloads (RAG, code, structured output) keep the win. Set
    /// `ngram_spec = false` to force the plain loop everywhere. Bridged to
    /// `KRILL_NGRAM_SPEC` by the CLI.
    public var ngramSpec: Bool

    /// Models directory override.
    public var modelsDir: String?

    /// HTTP server port.
    public var serverPort: Int

    /// HTTP server host.
    public var serverHost: String

    /// Idle timeout for models in serve mode (seconds).
    public var idleTimeout: Int

    /// Context-length override (tokens). nil = use the model's own max.
    /// `KRILL_CONTEXT_LENGTH` / `OLLAMA_CONTEXT_LENGTH`. (WS-D D4 / T1-3)
    public var contextLength: Int?

    /// Default keep-alive (duration string, e.g. "5m", "0", "-1").
    /// `KRILL_KEEP_ALIVE` / `OLLAMA_KEEP_ALIVE`. (WS-E / T1-4)
    public var keepAlive: String

    /// Max in-flight requests per loaded model. (WS-E / T1-5)
    public var numParallel: Int
    /// Max simultaneously-loaded models. (WS-E / T1-5)
    public var maxLoadedModels: Int
    /// Max queued requests before 503. (WS-E / T1-5)
    public var maxQueue: Int

    /// CORS allowlist. `KRILL_ORIGINS` / `OLLAMA_ORIGINS`. `*` = any.
    /// (WS-G / T3-1)
    public var origins: [String]

    /// Flash-attention toggle (advisory, WS-G / T3-2).
    public var flashAttention: Bool

    /// Default voice posture for the interactive TUI: "off"/"text" (default),
    /// "dictate", or "handsfree". Voice is opt-in; off keeps the chat text-only.
    public var voiceMode: String

    /// Read model replies aloud in the interactive TUI (text-to-speech). Opt-in,
    /// default false. Pairs with the hands-free voice posture for a full
    /// talk/listen loop. `speak_replies` in config; `KRILL_SPEAK_REPLIES` env.
    public var speakReplies: Bool

    /// Which surface the interactive TUI launches in: "chat" (default, pure
    /// inference) or "agent" (tools + file edits, the unified coding mode).
    /// `default_mode` in config. `krill code` always starts in agent mode
    /// regardless of this default.
    public var defaultMode: String

    /// Which permission posture agent mode opens on: "plan" (default, read-only),
    /// "ask" (confirm each mutating tool), "accept-edits" (auto-apply edits, ask
    /// for commands), or "auto"/"accept-all" (run everything). Shift+Tab cycles
    /// it live in the TUI. `default_agent_posture` in config.
    public var defaultAgentPosture: String

    /// Default reasoning ("thinking") state for new sessions: when true and the
    /// model has a thinking channel, the engine turns it on so the model reasons
    /// before answering. ON by default (it is a no-op for models with no thinking
    /// channel). `thinking` in config; `KRILL_ENABLE_THINKING` env. The TUI can
    /// toggle it per session; this is the starting value.
    public var thinking: Bool

    public init() {
        self.defaultModel = nil
        self.defaultQuant = 4
        self.kvCacheDtype = "fp16"
        self.prefixCacheSizeGB = 2.0
        self.prefixCacheMaxEntryGB = 4.0
        self.speculativeDecoding = false
        self.decodePipeline = true
        self.ngramSpec = true
        self.modelsDir = nil
        self.serverPort = 57455   // "KRILL" on a phone keypad; unique vs Ollama's 11434
        self.serverHost = "127.0.0.1"
        self.idleTimeout = 300
        self.contextLength = nil
        self.keepAlive = "5m"
        self.numParallel = 1
        self.maxLoadedModels = 1
        self.maxQueue = 512
        self.origins = ["http://localhost", "http://127.0.0.1", "https://localhost"]
        self.flashAttention = false
        self.voiceMode = "off"
        self.speakReplies = false
        self.defaultMode = "chat"
        self.defaultAgentPosture = "plan"
        self.thinking = true
    }

    /// Load configuration with full precedence chain.
    public static func load() -> KrillConfig {
        var config = KrillConfig()

        // Load from config.toml if it exists
        let configPath = Registry.defaultBaseDir().appendingPathComponent("config.toml")
        if let contents = try? String(contentsOf: configPath, encoding: .utf8) {
            config.mergeFromTOML(contents)
        }

        // Override from environment variables
        config.mergeFromEnvironment()

        return config
    }

    /// Merge values from a TOML string (simplified parser).
    mutating func mergeFromTOML(_ toml: String) {
        let lines = toml.components(separatedBy: .newlines)
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#"), !trimmed.hasPrefix("[") else {
                continue
            }

            let parts = trimmed.split(separator: "=", maxSplits: 1)
            guard parts.count == 2 else { continue }

            let key = parts[0].trimmingCharacters(in: .whitespaces)
            let value = parts[1].trimmingCharacters(in: .whitespaces)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\""))

            switch key {
            case "default_model":
                defaultModel = value.isEmpty ? nil : value
            case "default_quant":
                if let v = Int(value) { defaultQuant = v }
            case "kv_cache_dtype":
                kvCacheDtype = value
            case "prefix_cache_size_gb":
                if let v = Double(value) { prefixCacheSizeGB = v }
            case "prefix_cache_max_entry_gb":
                if let v = Double(value) { prefixCacheMaxEntryGB = v }
            case "speculative_decoding":
                speculativeDecoding = value == "true" || value == "1"
            case "decode_pipeline":
                decodePipeline = value == "true" || value == "1"
            case "ngram_spec":
                ngramSpec = value == "true" || value == "1"
            case "models_dir":
                modelsDir = value.isEmpty ? nil : value
            case "server_port", "port":
                if let v = Int(value) { serverPort = v }
            case "server_host", "host":
                serverHost = value
            case "idle_timeout":
                if let v = Int(value) { idleTimeout = v }
            case "max_loaded_models":
                // How many models stay resident simultaneously (route-or-load by
                // the request's `model` field). Set to 2+ to keep e.g. an
                // embedding model and a generation model both warm on one port.
                if let v = Int(value) { maxLoadedModels = v }
            case "voice_mode":
                voiceMode = value
            case "speak_replies":
                speakReplies = value == "true" || value == "1"
            case "default_mode":
                defaultMode = value
            case "default_agent_posture":
                defaultAgentPosture = value
            case "thinking", "enable_thinking":
                thinking = value == "true" || value == "1" || value == "on" || value == "yes"
            case "keep_alive":
                keepAlive = value
            case "num_parallel":
                if let v = Int(value) { numParallel = v }
            case "max_queue":
                if let v = Int(value) { maxQueue = v }
            default:
                break
            }
        }
    }

    private static func parseOrigins(_ raw: String) -> [String] {
        raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// Override from environment. `OLLAMA_*` aliases are applied first so a
    /// native `KRILL_*` of the same setting always wins (KRILL_* is the
    /// canonical surface; OLLAMA_* exists only for drop-in compatibility).
    mutating func mergeFromEnvironment() {
        let env = ProcessInfo.processInfo.environment

        // --- OLLAMA_* drop-in aliases (WS-G / T3-3) ---
        if let v = env["OLLAMA_HOST"] {
            // Accept "host", "host:port", or "http://host:port".
            var s = v
            if let r = s.range(of: "://") { s = String(s[r.upperBound...]) }
            if let colon = s.lastIndex(of: ":"),
               let p = Int(s[s.index(after: colon)...]) {
                serverPort = p
                s = String(s[..<colon])
            }
            if !s.isEmpty { serverHost = s }
        }
        if let v = env["OLLAMA_MODELS"] { modelsDir = v }
        if let v = env["OLLAMA_CONTEXT_LENGTH"], let i = Int(v) { contextLength = i }
        if let v = env["OLLAMA_KEEP_ALIVE"] { keepAlive = v }
        if let v = env["OLLAMA_NUM_PARALLEL"], let i = Int(v) { numParallel = i }
        if let v = env["OLLAMA_MAX_LOADED_MODELS"], let i = Int(v) { maxLoadedModels = i }
        if let v = env["OLLAMA_MAX_QUEUE"], let i = Int(v) { maxQueue = i }
        if let v = env["OLLAMA_KV_CACHE_TYPE"] { kvCacheDtype = v }
        if let v = env["OLLAMA_ORIGINS"] { origins = Self.parseOrigins(v) }
        if let v = env["OLLAMA_FLASH_ATTENTION"] {
            flashAttention = v == "1" || v.lowercased() == "true"
        }

        // --- KRILL_* native (wins over OLLAMA_*) ---
        if let v = env["KRILL_CONTEXT_LENGTH"], let i = Int(v) { contextLength = i }
        if let v = env["KRILL_KEEP_ALIVE"] { keepAlive = v }
        if let v = env["KRILL_NUM_PARALLEL"], let i = Int(v) { numParallel = i }
        if let v = env["KRILL_MAX_LOADED_MODELS"], let i = Int(v) { maxLoadedModels = i }
        if let v = env["KRILL_MAX_QUEUE"], let i = Int(v) { maxQueue = i }
        if let v = env["KRILL_ORIGINS"] { origins = Self.parseOrigins(v) }
        if let v = env["KRILL_FLASH_ATTENTION"] {
            flashAttention = v == "1" || v.lowercased() == "true"
        }
        if let v = env["KRILL_SPEAK_REPLIES"] {
            speakReplies = v == "1" || v.lowercased() == "true"
        }
        if let v = env["KRILL_ENABLE_THINKING"] {
            let s = v.lowercased()
            thinking = s == "1" || s == "true" || s == "yes" || s == "on"
        }

        if let v = ProcessInfo.processInfo.environment["KRILL_DEFAULT_MODEL"] {
            defaultModel = v
        }
        if let v = ProcessInfo.processInfo.environment["KRILL_DEFAULT_QUANT"], let i = Int(v) {
            defaultQuant = i
        }
        if let v = ProcessInfo.processInfo.environment["KRILL_KV_CACHE_DTYPE"] {
            kvCacheDtype = v
        }
        if let v = ProcessInfo.processInfo.environment["KRILL_PREFIX_CACHE_GB"], let d = Double(v) {
            prefixCacheSizeGB = d
        }
        if let v = ProcessInfo.processInfo.environment["KRILL_PREFIX_CACHE_MAX_ENTRY_GB"], let d = Double(v) {
            prefixCacheMaxEntryGB = d
        }
        if let v = ProcessInfo.processInfo.environment["KRILL_SPECULATIVE"] {
            speculativeDecoding = v == "true" || v == "1"
        }
        if let v = ProcessInfo.processInfo.environment["KRILL_MODELS_DIR"] {
            modelsDir = v
        }
        if let v = ProcessInfo.processInfo.environment["KRILL_PORT"], let i = Int(v) {
            serverPort = i
        }
        if let v = ProcessInfo.processInfo.environment["KRILL_HOST"] {
            serverHost = v
        }
    }
}
