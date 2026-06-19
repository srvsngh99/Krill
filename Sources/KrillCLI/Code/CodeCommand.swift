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
          help: "Allow the agent to run shell commands via the bash tool (PR2: no permission gate yet).")
    var bash: Bool = true

    func run() async throws {
        let registry = Registry()

        func nonEmpty(_ s: String?) -> String? {
            guard let s, !s.trimmingCharacters(in: .whitespaces).isEmpty else { return nil }
            return s
        }
        func isModelRef(_ s: String) -> Bool {
            registry.hasModel(s) || FileManager.default.fileExists(atPath: s)
        }

        // Disambiguate the single positional: `krill code <task>` (uses the
        // default model) vs `krill code <model> <task>`.
        let defaultModel = nonEmpty(KrillConfig.load().defaultModel)
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
        guard let task else {
            print("Error: no task. Usage: krill code [<model>] \"<task>\"")
            throw ExitCode.failure
        }

        let modelDir = registry.hasModel(model)
            ? registry.modelPath(model) : URL(fileURLWithPath: model)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("Error: model '\(model)' not found. Install with: krill pull \(model)")
            throw ExitCode.failure
        }

        print("Loading model from \(model)...")
        let engine = InferenceEngine(modelDirectory: modelDir)
        let loadStart = CFAbsoluteTimeGetCurrent()
        try await engine.load()
        print(String(format: "Ready (%.1fs).", CFAbsoluteTimeGetCurrent() - loadStart))

        // Filesystem toolset is always available; bash is opt-out (no permission
        // layer yet - that arrives in a later PR).
        var tools: [any Tool] = [
            ReadTool(), ListTool(), GlobTool(), GrepTool(),
            EditTool(), MultiEditTool(), WriteTool(),
        ]
        if bash {
            print("Note: the bash tool and file edits run with no sandbox. Use --no-bash to disable shell access.")
            tools.append(BashTool())
        }

        let loop = AgentLoop(
            generator: EngineGenerator(engine: engine, maxTokens: maxTokens),
            tools: ToolRegistry(tools),
            maxIterations: maxIterations)

        print("\n> \(task)\n")
        let transcript = await loop.run(user: task, system: system)
        render(transcript)
    }

    private func render(_ t: AgentTranscript) {
        for step in t.steps {
            if !step.assistantText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && !step.toolCalls.isEmpty {
                print(step.assistantText)
            }
            for call in step.toolCalls {
                let marker = call.result.isError ? "x" : "*"
                print("  [\(marker)] \(call.name)(\(call.argumentsJSON))")
                let lines = call.result.content
                    .split(separator: "\n", omittingEmptySubsequences: false).prefix(20)
                for line in lines { print("      \(line)") }
            }
        }
        if t.hitIterationLimit {
            print("\n[stopped at iteration limit (\(maxIterations)) without a final answer]")
        } else {
            print("\n\(t.finalText)")
        }
    }
}
