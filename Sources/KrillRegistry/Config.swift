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

    /// Which web-search backend `web_search` uses: "searxng" (default,
    /// local-first, no API key). The tool is read-only and only active when a
    /// backend URL is set. `search_backend` in config; `KRILL_SEARCH_BACKEND` env.
    public var searchBackend: String

    /// Base URL of the SearXNG instance `web_search` queries (e.g.
    /// "http://localhost:8888"); nil disables web search. The instance must have
    /// `json` enabled in its search.formats. `searxng_url` in config;
    /// `KRILL_SEARXNG_URL` env.
    public var searxngURL: String?

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
        self.searchBackend = "searxng"
        self.searxngURL = nil
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
            case "search_backend":
                searchBackend = value
            case "searxng_url":
                searxngURL = value.isEmpty ? nil : value
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

    /// Path to the persisted config file (`~/.krill/config.toml`).
    public static func configPath() -> URL {
        Registry.defaultBaseDir().appendingPathComponent("config.toml")
    }

    /// The config keys a user may set via `/config key=value`. Other TOML keys
    /// the parser understands (e.g. `port` as an alias of `server_port`) are not
    /// offered here to keep the surface small and unambiguous.
    public static let writableKeys: [String] = [
        "default_model", "default_quant", "default_mode", "default_agent_posture",
        "search_backend", "searxng_url",
        "kv_cache_dtype", "context_length", "thinking",
        "voice_mode", "speak_replies",
        "prefix_cache_size_gb", "prefix_cache_max_entry_gb",
        "speculative_decoding", "decode_pipeline", "ngram_spec", "flash_attention",
        "server_port", "server_host", "idle_timeout", "keep_alive",
        "num_parallel", "max_loaded_models", "max_queue", "models_dir",
    ]

    /// Persist `key = value` to `config.toml`, upserting in place: an existing
    /// assignment line for `key` is replaced (preserving everything else,
    /// including comments and blank lines), otherwise the pair is appended.
    /// Creates the file (and base dir) if missing. Throws on an unknown key or
    /// on a write failure, so the caller can surface a clear message.
    public static func set(key: String, value: String) throws {
        guard writableKeys.contains(key) else {
            throw ConfigError.unknownKey(key)
        }
        // The writer emits `key = "value"` and the parser reads it back by
        // trimming surrounding quotes (it does not unescape). A value containing
        // a quote, backslash, or newline would therefore not round-trip and could
        // corrupt the file on the atomic rewrite, so reject it rather than write
        // something the parser cannot read. No legitimate config value needs them.
        guard !value.contains(where: { $0 == "\"" || $0 == "\\" || $0 == "\n" || $0 == "\r" }) else {
            throw ConfigError.invalidValue(key: key, value: value)
        }
        let url = configPath()
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)

        let existing = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        let updated = upsertTOML(existing, key: key, value: value)
        try updated.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Pure line-wise upsert of `key = "value"` into existing TOML text: replaces
    /// an existing assignment for `key` in place (preserving comments, blanks, and
    /// every other line), else appends. Returned text ends with a newline.
    ///
    /// All of Krill's config keys live in the top-level (pre-`[section]`) region,
    /// so the upsert is scoped to that region: it never matches or writes a key
    /// inside a `[section]` table (which could share a bare key name), and a new
    /// key is inserted just before the first section header rather than after a
    /// table it does not belong to. Exposed for testing.
    public static func upsertTOML(_ existing: String, key: String, value: String) -> String {
        var lines = existing.isEmpty ? [] : existing.components(separatedBy: "\n")
        let newLine = "\(key) = \"\(value)\""
        var replaced = false
        var firstSection: Int?
        for i in lines.indices {
            let t = lines[i].trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("[") { if firstSection == nil { firstSection = i }; continue }
            if firstSection != nil { continue }   // only upsert in the global region
            guard !t.hasPrefix("#"), let eq = t.firstIndex(of: "=") else { continue }
            if t[..<eq].trimmingCharacters(in: .whitespaces) == key {
                lines[i] = newLine
                replaced = true
                break
            }
        }
        if !replaced {
            if let fs = firstSection {
                lines.insert(newLine, at: fs)      // end of the global region, before any table
            } else if let last = lines.last, last.trimmingCharacters(in: .whitespaces).isEmpty {
                lines[lines.count - 1] = newLine   // reuse a trailing blank line
            } else {
                lines.append(newLine)
            }
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// Current key/value pairs for the writable keys, in `writableKeys` order,
    /// for display by `/config`. Reflects the fully-resolved config (file + env).
    public func displayPairs() -> [(key: String, value: String)] {
        func b(_ v: Bool) -> String { v ? "true" : "false" }
        let map: [String: String] = [
            "default_model": defaultModel ?? "(none)",
            "default_quant": "\(defaultQuant)",
            "default_mode": defaultMode,
            "default_agent_posture": defaultAgentPosture,
            "search_backend": searchBackend,
            "searxng_url": searxngURL ?? "(none)",
            "kv_cache_dtype": kvCacheDtype,
            "context_length": contextLength.map { "\($0)" } ?? "(model default)",
            "thinking": b(thinking),
            "voice_mode": voiceMode,
            "speak_replies": b(speakReplies),
            "prefix_cache_size_gb": "\(prefixCacheSizeGB)",
            "prefix_cache_max_entry_gb": "\(prefixCacheMaxEntryGB)",
            "speculative_decoding": b(speculativeDecoding),
            "decode_pipeline": b(decodePipeline),
            "ngram_spec": b(ngramSpec),
            "flash_attention": b(flashAttention),
            "server_port": "\(serverPort)",
            "server_host": serverHost,
            "idle_timeout": "\(idleTimeout)",
            "keep_alive": keepAlive,
            "num_parallel": "\(numParallel)",
            "max_loaded_models": "\(maxLoadedModels)",
            "max_queue": "\(maxQueue)",
            "models_dir": modelsDir ?? "(default)",
        ]
        return KrillConfig.writableKeys.map { ($0, map[$0] ?? "") }
    }

    public enum ConfigError: Error, CustomStringConvertible {
        case unknownKey(String)
        case invalidValue(key: String, value: String)
        public var description: String {
            switch self {
            case .unknownKey(let k):
                return "unknown config key '\(k)'. Known keys: "
                    + KrillConfig.writableKeys.joined(separator: ", ")
            case .invalidValue(let k, _):
                return "value for '\(k)' may not contain a quote, backslash, or newline."
            }
        }
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
        if let v = env["KRILL_SEARCH_BACKEND"] { searchBackend = v }
        if let v = env["KRILL_SEARXNG_URL"] { searxngURL = v.isEmpty ? nil : v }

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
