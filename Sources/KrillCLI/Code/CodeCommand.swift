import ArgumentParser
import Foundation
import KrillEngine
import KrillHarness
import KrillRegistry

/// `krill code` - the native in-process agentic loop. The model is given a
/// task and a small toolset (PR2: just `bash`), and the loop runs
/// generate -> parse tool calls -> execute -> feed back until it answers.
///
/// PR2 is intentionally a single-shot hello-loop to prove the in-process loop
/// end-to-end on a local model. A full-screen `code` TUI, a permission layer,
/// and the rest of the toolset (Read/Edit/Grep) arrive in later PRs.
struct CodeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "code",
        abstract: "Agentic coding loop: the model can run tools to complete a task.")

    @Argument(help: "Model name (from registry) or path. Falls back to default_model.")
    var modelPath: String?

    @Argument(help: "The task for the agent.")
    var prompt: String?

    @Option(name: .long, help: "Maximum tokens per model turn.")
    var maxTokens: Int = 1024

    @Option(name: .long, help: "Maximum agent iterations (tool-call rounds).")
    var maxIterations: Int = 12

    @Option(name: .long, help: "System prompt override.")
    var system: String?

    @Flag(name: .long, inversion: .prefixedNo,
          help: "Allow the agent to run shell commands via the bash tool.")
    var bash: Bool = true

    @Flag(name: .long, inversion: .prefixedNo,
          help: "Grammar-constrain malformed tool calls: regenerate args that miss the schema, and re-pick a tool name that is not one of the offered tools (helps small models and models trained on another tool vocabulary).")
    var constrainArgs: Bool = true

    @Flag(name: .long,
          help: "Read-only plan mode: the agent may inspect files but cannot edit them or run commands; it proposes a plan. Shorthand for --permission-mode plan.")
    var plan: Bool = false

    @Flag(name: .long,
          help: "Use the classic line renderer instead of the full-screen TUI (also the default for non-interactive output, or when tool allow/deny flags are set).")
    var classic: Bool = false

    @Option(name: .long,
            help: "Permission level: plan (read-only), adaptive (plans then self-promotes to edits), ask (confirm each mutating tool), accept-edits (auto-apply edits, ask for commands), or auto/accept-all (run every tool). Defaults to default_agent_permissions (plan if unset or invalid).")
    var permissionMode: String?

    @Option(name: .customLong("allow-tool"), parsing: .singleValue,
            help: "Always allow this tool by name (repeatable). Overrides the mode but not --deny-tool.")
    var allowTools: [String] = []

    @Option(name: .customLong("deny-tool"), parsing: .singleValue,
            help: "Always deny this tool by name (repeatable). Highest precedence.")
    var denyTools: [String] = []

    func run() async throws {
        let registry = Registry()
        let config = KrillConfig.load()

        func nonEmpty(_ s: String?) -> String? {
            guard let s, !s.trimmingCharacters(in: .whitespaces).isEmpty else { return nil }
            return s
        }
        func isModelRef(_ s: String) -> Bool {
            registry.hasModel(s) || FileManager.default.fileExists(atPath: s)
        }

        // Disambiguate the single positional: `krill code <task>` (uses the
        // default model) vs `krill code <model> <task>`.
        let defaultModel = nonEmpty(config.defaultModel)
        var resolvedModel = nonEmpty(modelPath)
        var task = nonEmpty(prompt)
        if task == nil, let only = resolvedModel, let def = defaultModel, !isModelRef(only) {
            resolvedModel = def
            task = only
        }
        guard let model = resolvedModel ?? defaultModel else {
            print("Error: no model. Pass one (krill code <model> \"<task>\") or set default_model in ~/.krill/config.toml.")
            throw ExitCode.failure
        }
        // A task is optional for the full-screen TUI (bare `krill code` opens
        // it idle, like other coding agents); the classic line renderer still
        // requires one and fails below with a clear error.

        let modelDir = registry.hasModel(model)
            ? registry.modelPath(model) : URL(fileURLWithPath: model)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("Error: model '\(model)' not found. Install with: krill pull \(model)")
            throw ExitCode.failure
        }

        // Resolve the permission mode: --plan is shorthand for plan mode and
        // wins if both are passed. With no CLI override, honor the configured
        // posture; malformed configuration fails closed to read-only plan.
        let mode: PermissionMode
        if plan {
            mode = .plan
        } else if let raw = nonEmpty(permissionMode) {
            guard let parsed = PermissionMode.parse(raw) else {
                print("Error: invalid --permission-mode '\(raw)'. Choose: "
                    + PermissionMode.allCases.map(\.rawValue).joined(separator: ", ") + " (or 'auto').")
                throw ExitCode.failure
            }
            mode = parsed
        } else {
            let configured = nonEmpty(config.defaultAgentPermissions)
            mode = PermissionMode.configuredDefault(configured)
            if let configured, PermissionMode.parse(configured) == nil {
                print("Warning: invalid default_agent_permissions '\(configured)'; using plan mode.")
            }
        }
        let permissionBox = PermissionBox(
            mode: mode, allow: Set(allowTools), deny: Set(denyTools))

        // Fail fast before the (expensive) model load when no task was given
        // and the selected surface cannot start idle: the full-screen TUI
        // launches idle (bare `krill code`, like other agents); the classic
        // line renderer needs a task up front.
        let opensIdleSurface = RawTerminal.isInteractive && !classic
            && bash && allowTools.isEmpty && denyTools.isEmpty
        if task == nil && !opensIdleSurface {
            print("Error: no task. Usage: krill code [<model>] \"<task>\"")
            print("(bare `krill code` opens the interactive TUI in a terminal)")
            throw ExitCode.failure
        }

        let engine = InferenceEngine(modelDirectory: modelDir)
        try await BrandedLoad.run(model: model, wordmark: "Krill Code") { try await engine.load() }

        // Filesystem toolset is always available; bash is opt-out. The
        // permission layer (below) governs whether mutating tools actually run.
        var tools: [any Tool] = [
            ReadTool(), ListTool(), GlobTool(), GrepTool(), WebFetchTool(), WebSearchTool(),
            EditTool(), MultiEditTool(), WriteTool(),
            NowTool(), TodoTool(), RepoMapTool(),
        ]
        if bash { tools.append(BashTool()) }
        let questionAsker: StdinQuestionAsker? = RawTerminal.isInteractive
            ? StdinQuestionAsker() : nil
        tools.append(contentsOf: AgentInteractionTools.make(
            mode: mode,
            permissionBox: permissionBox,
            questionGate: questionAsker))

        // Steer the model in plan mode; surface the posture inline. This is
        // shared by the classic renderer and the TUI so both
        // execution surfaces preserve exactly the same permission semantics.
        // Ambient facts first (date/time, cwd, platform, model), then posture
        // steering, then the user's system prompt. Since this always yields a
        // system turn, the tooling layer's fallback anti-over-calling directive
        // never fires — carry it explicitly when the user supplied no prompt.
        var systemParts = [AgentEnvironment.contextLine(modelName: model)]
        if let brief = AgentEnvironment.projectBrief() { systemParts.append(brief) }
        systemParts.append(contentsOf: AgentEnvironment.permissionDirectives(for: mode))
        if let userSystem = nonEmpty(system) {
            systemParts.append(userSystem)
        } else {
            systemParts.append(AgentEnvironment.toolDirective(for: mode))
        }
        let effectiveSystem = systemParts.joined(separator: "\n\n")

        // On an interactive terminal with the default toolset, `krill code` is
        // just the unified chat TUI launched in agent mode - same surface as
        // `krill`, hands already on. The full-screen TUI hosts every posture
        // (including ask / accept-edits via an in-TUI approval prompt). Fine-
        // grained tool flags (--no-bash / --allow-tool / --deny-tool) fall back
        // to the classic line renderer below, which honors them.
        let defaultToolset = bash && allowTools.isEmpty && denyTools.isEmpty
        if RawTerminal.isInteractive && !classic && defaultToolset {
            let tui = ChatTUI(
                engine: engine, modelName: model, system: nonEmpty(system),
                params: .greedy, maxTokens: maxTokens, registry: registry,
                initialImage: nil, initialAudio: nil,
                voiceModeSetting: config.voiceMode,
                speakRepliesSetting: config.speakReplies,
                voiceEngineSetting: config.voiceEngine,
                voiceLanguageSetting: config.voiceLanguage,
                voiceIdentifierSetting: config.voiceIdentifier,
                voiceRateSetting: config.voiceRate,
                voiceWhisperModelSetting: config.voiceWhisperModel,
                thinkingSetting: config.thinking,
                modeSetting: "agent", agentPermissionsSetting: mode.rawValue,
                initialAgentTask: task)
            await tui.run()
            return
        }

        // Classic line renderer (--classic, non-interactive, or tool flags set).
        // Surface the posture inline.
        switch mode {
        case .plan:
            print("Plan mode: read-only. The agent can inspect files and request approval to execute.")
        case .adaptive:
            print("Adaptive mode: starts read-only, then may self-promote to edits; commands still ask.")
        case .ask:
            print("Ask mode: you will be prompted to approve each file edit or shell command.")
        case .acceptEdits:
            print("Accept-edits mode: file edits apply automatically; you will be prompted before shell commands.")
        case .acceptAll:
            if bash {
                print("Note: the bash tool and file edits run with no sandbox. Use --no-bash to disable shell access, or --plan / --permission-mode ask to gate tools.")
            }
        }

        guard let task else {
            print("Error: no task. Usage: krill code [<model>] \"<task>\"")
            throw ExitCode.failure
        }

        var generator = EngineGenerator(engine: engine, maxTokens: maxTokens)
        generator.onStats = { stats in
            let rate = stats.decodeTime > 0
                ? Double(stats.generatedTokens) / stats.decodeTime : 0
            let line = String(
                format: "  [generation: %d prompt + %d output tokens · %.1f tok/s]\n",
                stats.promptTokens, stats.generatedTokens, rate)
            FileHandle.standardError.write(Data(line.utf8))
        }
        let loop = AgentLoop(
            generator: generator,
            tools: ToolRegistry(tools),
            maxIterations: maxIterations,
            constrainToolArgs: constrainArgs,
            permission: permissionBox.policy,
            permissionBox: permissionBox,
            gate: (mode != .acceptAll && RawTerminal.isInteractive) ? StdinApprover() : nil)

        print("\n> \(task)\n")
        // Render the run live as the loop emits events, instead of dumping the
        // whole transcript at the end. The renderer folds events through the
        // shared `foldAgentEvent` seam (same as the full-screen TUI and
        // background sessions), so a new `AgentEvent` case renders everywhere at
        // once rather than only in this command's hand-rolled switch.
        let renderer = LineAgentRenderer()
        let onEvent: @Sendable (AgentEvent) -> Void = { renderer.handle($0) }
        _ = await loop.run(user: task, system: effectiveSystem, onEvent: onEvent)
    }
}
