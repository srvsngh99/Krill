import ArgumentParser
import Foundation
import KrillHarness
import KrillRegistry
import KrillTooling

extension CodeCommand {
    func runRemote(model requestedModel: String?, task: String?, config: KrillConfig) async throws {
        guard provider != .local else { return }

        let endpoint: URL
        if let raw = nonEmpty(baseURL) {
            guard let parsed = URL(string: raw),
                  ["http", "https"].contains(parsed.scheme?.lowercased() ?? ""),
                  parsed.host != nil else {
                print("Error: invalid --base-url '\(raw)'. Use an http:// or https:// URL.")
                throw ExitCode.failure
            }
            endpoint = parsed
        } else {
            endpoint = OpenCodeZen.defaultBaseURL
        }

        if listModels {
            guard provider == .opencode else {
                print("Error: --list-models is available with --provider opencode.")
                throw ExitCode.failure
            }
            do {
                let models = try await OpenCodeZen.freeModels(baseURL: endpoint)
                guard !models.isEmpty else {
                    print("No free models are currently advertised by OpenCode Zen.")
                    return
                }
                print("OpenCode Zen free models:")
                for model in models {
                    let preferred = model.id == OpenCodeZen.preferredFreeModelID ? " (default)" : ""
                    print("  \(model.id)\(preferred)")
                }
                return
            } catch {
                print("Error: could not discover OpenCode free models: \(error.localizedDescription)")
                throw ExitCode.failure
            }
        }

        guard let task = nonEmpty(task) else {
            print("Error: no task. Usage: krill code --provider \(provider.rawValue) [<model>] \"<task>\"")
            throw ExitCode.failure
        }

        let mode = try resolvedPermissionMode(config: config)
        let permissionBox = PermissionBox(
            mode: mode, allow: Set(allowTools), deny: Set(denyTools))
        let tools = remoteTools(mode: mode, permissionBox: permissionBox)

        let generator: any HarnessGenerator
        let modelName: String
        switch provider {
        case .local:
            return
        case .opencode:
            do {
                let freeModels = try await OpenCodeZen.freeModels(baseURL: endpoint)
                guard !freeModels.isEmpty else {
                    print("Error: OpenCode Zen currently advertises no free models.")
                    throw ExitCode.failure
                }
                if let requestedModel = nonEmpty(requestedModel) {
                    guard freeModels.contains(where: { $0.id == requestedModel }) else {
                        print("Error: '\(requestedModel)' is not in OpenCode Zen's current free-model catalog.")
                        print("Run: krill code --provider opencode --list-models")
                        throw ExitCode.failure
                    }
                    modelName = requestedModel
                } else {
                    guard let selected = OpenCodeZen.defaultFreeModel(from: freeModels) else {
                        throw ExitCode.failure
                    }
                    modelName = selected.id
                }
                generator = OpenAICompatibleHarnessGenerator(
                    model: modelName, baseURL: endpoint, maxTokens: maxTokens)
                print("Using OpenCode Zen: \(modelName) (keyless, live free-model catalog)")
            } catch let exit as ExitCode {
                throw exit
            } catch {
                print("Error: could not discover OpenCode free models: \(error.localizedDescription)")
                throw ExitCode.failure
            }
        case .codex:
            modelName = nonEmpty(requestedModel) ?? "Codex subscription default"
            generator = CodexCLIHarnessGenerator(
                model: nonEmpty(requestedModel), executable: codexExecutable)
            print("Using Codex CLI: \(modelName) (reuses `codex login`; no API key copied)")
        }

        let effectiveSystem = remoteSystemPrompt(
            modelName: modelName, mode: mode, userSystem: nonEmpty(system))
        printPermissionMode(mode)
        print("\n> \(task)\n")

        let loop = AgentLoop(
            generator: generator,
            tools: ToolRegistry(tools),
            maxIterations: maxIterations,
            constrainToolArgs: constrainArgs,
            permission: permissionBox.policy,
            permissionBox: permissionBox,
            gate: (mode != .acceptAll && RawTerminal.isInteractive) ? StdinApprover() : nil)
        let renderer = LineAgentRenderer()
        let onEvent: @Sendable (AgentEvent) -> Void = { renderer.handle($0) }
        _ = await loop.run(user: task, system: effectiveSystem, onEvent: onEvent)
    }

    private func resolvedPermissionMode(config: KrillConfig) throws -> PermissionMode {
        if plan { return .plan }
        if let raw = nonEmpty(permissionMode) {
            guard let parsed = PermissionMode.parse(raw) else {
                print("Error: invalid --permission-mode '\(raw)'. Choose: "
                    + PermissionMode.allCases.map(\.rawValue).joined(separator: ", ") + " (or 'auto').")
                throw ExitCode.failure
            }
            return parsed
        }
        let configured = nonEmpty(config.defaultAgentPermissions)
        let mode = PermissionMode.configuredDefault(configured)
        if let configured, PermissionMode.parse(configured) == nil {
            print("Warning: invalid default_agent_permissions '\(configured)'; using plan mode.")
        }
        return mode
    }

    private func remoteTools(mode: PermissionMode, permissionBox: PermissionBox) -> [any Tool] {
        var tools: [any Tool] = [
            ReadTool(), ListTool(), GlobTool(), GrepTool(), WebFetchTool(), WebSearchTool(),
            EditTool(), MultiEditTool(), WriteTool(),
            NowTool(), TodoTool(), RepoMapTool(),
        ]
        if bash { tools.append(BashTool()) }
        let questionAsker: StdinQuestionAsker? = RawTerminal.isInteractive
            ? StdinQuestionAsker() : nil
        tools.append(contentsOf: AgentInteractionTools.make(
            mode: mode, permissionBox: permissionBox, questionGate: questionAsker))
        return tools
    }

    private func remoteSystemPrompt(
        modelName: String, mode: PermissionMode, userSystem: String?
    ) -> String {
        var parts = [AgentEnvironment.contextLine(modelName: modelName)]
        if let brief = AgentEnvironment.projectBrief() { parts.append(brief) }
        parts.append(contentsOf: AgentEnvironment.permissionDirectives(for: mode))
        parts.append(userSystem ?? AgentEnvironment.toolDirective(for: mode))
        return parts.joined(separator: "\n\n")
    }

    private func printPermissionMode(_ mode: PermissionMode) {
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
    }

    private func nonEmpty(_ value: String?) -> String? {
        guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        return value
    }
}
