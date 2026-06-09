import Foundation

/// KrillLM configuration loaded from ~/.krillm/config.toml.
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

    /// Enable speculative decoding by default.
    public var speculativeDecoding: Bool

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

    public init() {
        self.defaultModel = nil
        self.defaultQuant = 4
        self.kvCacheDtype = "fp16"
        self.prefixCacheSizeGB = 2.0
        self.speculativeDecoding = false
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
            case "speculative_decoding":
                speculativeDecoding = value == "true" || value == "1"
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
