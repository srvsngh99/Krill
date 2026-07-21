import Foundation

/// Serialize a JSON object to a string with proper escaping, so model ids and
/// URLs are never raw-interpolated into config files. Returns "{}" only if the
/// object is somehow non-serializable (it never is here).
func jsonString(_ obj: [String: Any]) -> String {
    guard let d = try? JSONSerialization.data(withJSONObject: obj),
          let s = String(data: d, encoding: .utf8) else { return "{}" }
    return s
}

/// Per-agent launch profiles for `krill launch <agent>`. Each profile says
/// how to wire one coding agent to the local Krill server: which wire
/// protocol it speaks, what env to export, what config file(s) to write or
/// merge, any setup subcommands to run first, and the binary to exec.
///
/// Adding an agent is a one-literal edit to ``AgentProfiles/all``; the
/// ``LaunchCommand`` flow stays generic over the table.

/// The HTTP surface an agent talks to, mapped to a Krill endpoint.
enum WireProtocol: String, Sendable {
    case anthropic        // -> POST /v1/messages
    case openAIChat       // -> POST /v1/chat/completions
    case openAIResponses  // -> POST /v1/responses
}

/// A config file the launcher writes or merges before exec. `render` is given
/// the server root URL (`http://host:port`), model id, and resolved API key.
struct AgentConfigFile: Sendable {
    enum Mode: Sendable {
        case write       // create/overwrite verbatim (krill-owned paths)
        case mergeJSON   // deep-merge the rendered JSON into existing JSON
    }
    let path: String        // may start with ~, expanded at apply time
    let mode: Mode
    let render: @Sendable (_ baseURL: String, _ model: String, _ apiKey: String) -> String
}

struct AgentProfile: Sendable {
    let id: String
    let displayName: String
    let summary: String
    let wire: WireProtocol
    /// Env to export before exec (values may contain ~, expanded at apply time).
    let env: @Sendable (_ baseURL: String, _ model: String, _ apiKey: String) -> [String: String]
    let configFiles: [AgentConfigFile]
    /// Setup commands to run (and wait for) before exec, e.g. `hermes config set`.
    let preExec: @Sendable (_ baseURL: String, _ model: String, _ apiKey: String) -> [[String]]
    let binary: String
    let args: @Sendable (_ model: String) -> [String]
    let notInstalledHint: String

    init(id: String, displayName: String, summary: String, wire: WireProtocol,
         env: @escaping @Sendable (String, String, String) -> [String: String] = { _, _, _ in [:] },
         configFiles: [AgentConfigFile] = [],
         preExec: @escaping @Sendable (String, String, String) -> [[String]] = { _, _, _ in [] },
         binary: String, args: @escaping @Sendable (String) -> [String] = { _ in [] },
         notInstalledHint: String) {
        self.id = id
        self.displayName = displayName
        self.summary = summary
        self.wire = wire
        self.env = env
        self.configFiles = configFiles
        self.preExec = preExec
        self.binary = binary
        self.args = args
        self.notInstalledHint = notInstalledHint
    }
}

enum AgentProfiles {

    // Claude Code: Anthropic Messages API. Claude Code appends /v1/messages to
    // ANTHROPIC_BASE_URL, so the base is the server root (no /v1). The
    // small/fast + default model aliases all point at the one local model so
    // background "haiku" calls route to it too.
    static let claude = AgentProfile(
        id: "claude",
        displayName: "Claude Code",
        summary: "Anthropic's coding tool with subagents",
        wire: .anthropic,
        env: { base, model, apiKey in [
            "ANTHROPIC_BASE_URL": base,
            "ANTHROPIC_AUTH_TOKEN": apiKey,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_SMALL_FAST_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        ] },
        binary: "claude",
        notInstalledHint: "Install Claude Code:  npm i -g @anthropic-ai/claude-code")

    // Codex: OpenAI Responses API (it dropped Chat Completions). CODEX_HOME
    // relocates Codex's config dir, so we write a complete, isolated config.toml
    // into a krill-owned dir and never touch the user's real ~/.codex.
    static let codex = AgentProfile(
        id: "codex",
        displayName: "Codex",
        summary: "OpenAI's open-source coding agent",
        wire: .openAIResponses,
        env: { _, _, apiKey in [
            "CODEX_HOME": "~/.krill/agents/codex",
            "KRILL_API_KEY": apiKey,
        ] },
        configFiles: [AgentConfigFile(
            path: "~/.krill/agents/codex/config.toml", mode: .write,
            render: { base, model, _ in """
                model = "\(model)"
                model_provider = "krill"

                [model_providers.krill]
                name = "Krill"
                base_url = "\(base)/v1"
                wire_api = "responses"
                env_key = "KRILL_API_KEY"
                """ })],
        binary: "codex",
        notInstalledHint: "Install Codex:  npm i -g @openai/codex")

    // OpenCode: OpenAI Chat Completions via the @ai-sdk/openai-compatible
    // provider. Deep-merge only the `krill` provider + default model into the
    // user's opencode.json (a .bak is written first).
    static let opencode = AgentProfile(
        id: "opencode",
        displayName: "OpenCode",
        summary: "Anomaly's open-source coding agent",
        wire: .openAIChat,
        env: { _, _, apiKey in ["KRILL_API_KEY": apiKey] },
        configFiles: [AgentConfigFile(
            path: "~/.config/opencode/opencode.json", mode: .mergeJSON,
            render: { base, model, _ in jsonString([
                "provider": ["krill": [
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Krill",
                    "options": [
                        "baseURL": "\(base)/v1",
                        "apiKey": "{env:KRILL_API_KEY}",
                    ],
                    "models": [model: ["name": model]],
                ]],
                "model": "krill/\(model)",
            ]) })],
        binary: "opencode",
        notInstalledHint: "Install opencode:  npm i -g opencode-ai")

    // Hermes Agent (Nous Research): OpenAI-compatible custom endpoint,
    // configured via its own `hermes config set` subcommands before launch.
    static let hermes = AgentProfile(
        id: "hermes",
        displayName: "Hermes Agent",
        summary: "Self-improving AI agent built by Nous Research",
        wire: .openAIChat,
        env: { _, _, apiKey in ["OPENAI_API_KEY": apiKey] },
        preExec: { base, model, apiKey in [
            ["hermes", "config", "set", "model.provider", "custom"],
            ["hermes", "config", "set", "model.base_url", "\(base)/v1"],
            ["hermes", "config", "set", "model.default", model],
            ["hermes", "config", "set", "model.api_key", apiKey],
        ] },
        binary: "hermes",
        notInstalledHint: "Install Hermes Agent:  see https://hermes-agent.nousresearch.com")

    // Pi: minimal coding agent, OpenAI-compatible. Configured by merging a
    // provider + model into ~/.pi/agent/models.json.
    static let pi = AgentProfile(
        id: "pi",
        displayName: "Pi",
        summary: "Minimal AI agent toolkit with plugin support",
        wire: .openAIChat,
        configFiles: [AgentConfigFile(
            path: "~/.pi/agent/models.json", mode: .mergeJSON,
            render: { base, model, apiKey in jsonString([
                "providers": ["krill": [
                    "baseUrl": "\(base)/v1",
                    "api": "openai-completions",
                    "apiKey": apiKey,
                ]],
                "models": ["krill/\(model)": [
                    "provider": "krill",
                    "id": model,
                    "name": model,
                    "contextWindow": 65536,
                    "maxTokens": 8192,
                ]],
            ]) })],
        binary: "pi",
        notInstalledHint: "Install Pi:  npm i -g @mariozechner/pi-coding-agent")

    // Copilot CLI (GitHub): OpenAI-compatible BYOK via env. Copilot's system
    // prompt + tools are large, so it wants a >= 32k context model.
    static let copilot = AgentProfile(
        id: "copilot",
        displayName: "Copilot CLI",
        summary: "GitHub's AI coding agent for the terminal",
        wire: .openAIChat,
        env: { base, model, apiKey in [
            "COPILOT_PROVIDER_BASE_URL": "\(base)/v1",
            "COPILOT_PROVIDER_API_KEY": apiKey,
            "COPILOT_PROVIDER_TYPE": "openai",
            "COPILOT_MODEL": model,
        ] },
        binary: "copilot",
        notInstalledHint: "Install Copilot CLI:  npm i -g @github/copilot")

    // Droid (Factory): OpenAI-compatible BYOK. A custom_models entry is merged
    // into ~/.factory/config.json (the array concatenates, so existing custom
    // models are preserved). Select it in Droid via /model or
    // `droid --model custom:<display name>`.
    static let droid = AgentProfile(
        id: "droid",
        displayName: "Droid",
        summary: "Factory's coding agent across terminal and IDEs",
        wire: .openAIChat,
        configFiles: [AgentConfigFile(
            path: "~/.factory/config.json", mode: .mergeJSON,
            render: { base, model, apiKey in jsonString([
                "custom_models": [[
                    "model_display_name": "Krill \(model)",
                    "model": model,
                    "base_url": "\(base)/v1",
                    "api_key": apiKey,
                    "provider": "generic-chat-completion-api",
                    "max_tokens": 8192,
                ]],
            ]) })],
        binary: "droid",
        notInstalledHint: "Install Droid:  curl -fsSL https://app.factory.ai/cli | sh")

    /// The launchable roster. claude/codex/opencode are verified against each
    /// agent's documented local-endpoint setup; hermes/pi/copilot/droid follow
    /// their documented OpenAI-compatible config and may need per-version tweaks.
    /// (codex-app and openclaw need manual setup for now - see the connect-an-
    /// agent docs.)
    static let all: [AgentProfile] = [claude, codex, opencode, hermes, pi, copilot, droid]

    static func find(_ id: String) -> AgentProfile? {
        all.first { $0.id.lowercased() == id.lowercased() }
    }
}
