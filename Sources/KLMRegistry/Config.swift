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

    public init() {
        self.defaultModel = nil
        self.defaultQuant = 4
        self.kvCacheDtype = "fp16"
        self.prefixCacheSizeGB = 2.0
        self.speculativeDecoding = false
        self.modelsDir = nil
        self.serverPort = 11435
        self.serverHost = "127.0.0.1"
        self.idleTimeout = 300
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
            default:
                break
            }
        }
    }

    /// Override from KRILL_* environment variables.
    mutating func mergeFromEnvironment() {
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
