import ArgumentParser
import Foundation
import KLMRegistry
#if canImport(Darwin)
import Darwin
#endif

/// `krillm launch <agent>` - boot a coding agent (Claude Code, Codex,
/// OpenCode, Hermes, Pi, ...) pre-wired to the local KrillLM server, with no
/// manual env/config fiddling. Mirrors Ollama's `ollama launch`.
///
/// Flow: resolve the agent profile -> ensure the server is up (auto-start if
/// needed) with the chosen model loaded -> export env + write/merge the
/// agent's config + run any setup subcommands -> exec the agent so it inherits
/// the terminal.
struct LaunchCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "launch",
        abstract: "Boot a coding agent (Claude Code, Codex, OpenCode, ...) wired to KrillLM")

    @Argument(help: "Agent to launch (run `krillm launch` with no agent to list).")
    var agent: String?

    @Argument(parsing: .postTerminator,
              help: "Extra args forwarded to the agent binary (after a `--`).")
    var agentArgs: [String] = []

    @Option(name: .long, help: "Model to serve (default: first installed model).")
    var model: String?

    @Option(name: .long, help: "Server port (default: config / 57455).")
    var port: Int?

    @Option(name: .long, help: "Server host (default: config / 127.0.0.1).")
    var host: String?

    @Flag(name: .long, help: "Do not auto-start the server; require an already-running one.")
    var noServe: Bool = false

    func run() async throws {
        // No agent -> print the roster (like the Ollama launcher screen).
        guard let agentId = agent else {
            printRoster()
            return
        }
        guard let profile = AgentProfiles.find(agentId) else {
            print("Unknown agent: \(agentId)\n")
            printRoster()
            throw ExitCode.failure
        }

        let cfg = KrillConfig.load()
        let h = host ?? cfg.serverHost
        let p = port ?? cfg.serverPort
        let baseURL = "http://\(h):\(p)"
        let registry: Registry = {
            if let md = cfg.modelsDir, !md.isEmpty {
                return Registry(modelsDir: URL(fileURLWithPath: md))
            }
            return Registry()
        }()

        // Pick the model: explicit flag, else first installed.
        guard let modelName = model ?? registry.listModels().first?.name else {
            print("No models installed. Pull one first, e.g.:  krillm pull gemma-4-e2b")
            throw ExitCode.failure
        }

        // Ensure the server is up with the model loaded.
        try await ensureServer(host: h, port: p, baseURL: baseURL,
                               model: modelName, registry: registry)

        // Apply the agent's wiring: config files, then env, then setup commands.
        try applyConfigFiles(profile.configFiles, baseURL: baseURL, model: modelName)
        let env = profile.env(baseURL, modelName)
        runPreExec(profile.preExec(baseURL, modelName))
        let agentArgv = [profile.binary] + profile.args(modelName) + agentArgs

        print("""

            Launching \(profile.displayName) -> \(baseURL) (\(profile.wire.rawValue), model \(modelName))
            """)
        if profile.wire == .openAIResponses || profile.wire == .anthropic {
            print("Tip: coding agents want a large context window; prefer a model/serve context >= 64k.")
        }
        print("")

        // Export env, then replace this process with the agent so it owns the TTY.
        for (k, v) in env { setenv(k, expandTilde(v), 1) }
        try execAgent(profile, argv: agentArgv)
    }

    // MARK: - Roster

    private func printRoster() {
        print("Launch a coding agent wired to your local KrillLM server.\n")
        let idW = AgentProfiles.all.map { $0.id.count }.max() ?? 8
        let nameW = AgentProfiles.all.map { $0.displayName.count }.max() ?? 12
        for a in AgentProfiles.all {
            let id = a.id.padding(toLength: idW, withPad: " ", startingAt: 0)
            let name = a.displayName.padding(toLength: nameW, withPad: " ", startingAt: 0)
            print("  \(id)  \(name)  \(a.summary)")
        }
        print("""

            Usage:  krillm launch <agent> [--model <name>] [--port <port>] [-- <agent args>]
            """)
    }

    // MARK: - Server lifecycle

    private func ensureServer(host: String, port: Int, baseURL: String,
                              model: String, registry: Registry) async throws {
        if let health = await getHealth(baseURL: baseURL) {
            // Already running: load the requested model if it is not active.
            let loaded = (health["model_loaded"] as? Bool) ?? false
            let active = health["model"] as? String
            if !loaded || active != model {
                print("Loading '\(model)' into the running server...")
                // Fail loud (like the auto-start branch) so we never exec the
                // agent against the wrong model on a failed load.
                guard await loadModel(baseURL: baseURL, model: model) else {
                    print("Error: server could not load '\(model)'. "
                          + "Check `krillm list` and the server log.")
                    throw ExitCode.failure
                }
            }
            return
        }

        guard !noServe else {
            print("""
                No KrillLM server reachable at \(baseURL).
                Start one in another terminal:  krillm serve --model \(model) --port \(port)
                """)
            throw ExitCode.failure
        }

        // Auto-start a detached server, then wait for it to become healthy.
        print("Starting KrillLM server at \(baseURL) with '\(model)'...")
        let logPath = expandTilde("~/.krillm/agents/\(model.replacingOccurrences(of: "/", with: "_"))-serve.log")
        try? FileManager.default.createDirectory(
            atPath: (logPath as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: logPath, contents: nil)
        guard let logHandle = FileHandle(forWritingAtPath: logPath) else {
            print("Error: could not open server log at \(logPath)")
            throw ExitCode.failure
        }

        let krillm = Bundle.main.executablePath ?? CommandLine.arguments[0]
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: krillm)
        proc.arguments = ["serve", "--model", model, "--port", "\(port)", "--host", host]
        proc.standardOutput = logHandle
        proc.standardError = logHandle
        do {
            try proc.run()
        } catch {
            print("Error: could not start the server (\(error)).")
            throw ExitCode.failure
        }

        // Poll health. The server only listens after the model is loaded and
        // warmed, so a healthy response means the model is ready.
        for _ in 0..<120 {
            if await getHealth(baseURL: baseURL) != nil { print("Server ready."); return }
            if !proc.isRunning {
                let log = (try? String(contentsOfFile: logPath, encoding: .utf8)) ?? ""
                print("Server exited before becoming ready. Last log lines:\n"
                      + log.split(separator: "\n").suffix(8).joined(separator: "\n"))
                throw ExitCode.failure
            }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
        print("Server did not become ready in time; see \(logPath).")
        throw ExitCode.failure
    }

    private func getHealth(baseURL: String) async -> [String: Any]? {
        guard let url = URL(string: "\(baseURL)/healthz") else { return nil }
        var req = URLRequest(url: url)
        req.timeoutInterval = 2
        guard let (data, resp) = try? await URLSession.shared.data(for: req),
              (resp as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    /// Ask the running server to load `model`. Returns true on HTTP 200.
    private func loadModel(baseURL: String, model: String) async -> Bool {
        guard let url = URL(string: "\(baseURL)/v1/models/load") else { return false }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["model": model])
        guard let (_, resp) = try? await URLSession.shared.data(for: req) else { return false }
        return (resp as? HTTPURLResponse)?.statusCode == 200
    }

    // MARK: - Config + setup

    private func applyConfigFiles(_ files: [AgentConfigFile],
                                  baseURL: String, model: String) throws {
        for f in files {
            let path = expandTilde(f.path)
            let dir = (path as NSString).deletingLastPathComponent
            try? FileManager.default.createDirectory(
                atPath: dir, withIntermediateDirectories: true)
            let rendered = f.render(baseURL, model)
            switch f.mode {
            case .write:
                try rendered.write(toFile: path, atomically: true, encoding: .utf8)
            case .mergeJSON:
                try mergeJSON(rendered, into: path)
            }
            print("Wrote \(path)")
        }
    }

    /// Deep-merge a rendered JSON fragment into an existing JSON file (new keys
    /// win, sibling keys preserved). A .bak of any existing file is kept.
    private func mergeJSON(_ fragment: String, into path: String) throws {
        guard let overlay = try JSONSerialization.jsonObject(
            with: Data(fragment.utf8)) as? [String: Any] else {
            throw LaunchError.invalidConfigFragment(path)
        }
        var base: [String: Any] = [:]
        if FileManager.default.fileExists(atPath: path),
           let data = FileManager.default.contents(atPath: path) {
            base = ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any]) ?? [:]
            try? data.write(to: URL(fileURLWithPath: path + ".bak"))
        }
        let merged = deepMerge(base, overlay)
        let out = try JSONSerialization.data(
            withJSONObject: merged, options: [.prettyPrinted, .sortedKeys])
        try out.write(to: URL(fileURLWithPath: path))
    }

    private func deepMerge(_ base: [String: Any], _ overlay: [String: Any]) -> [String: Any] {
        var result = base
        for (k, v) in overlay {
            if let bv = result[k] as? [String: Any], let ov = v as? [String: Any] {
                result[k] = deepMerge(bv, ov)
            } else if let bv = result[k] as? [Any], let ov = v as? [Any] {
                // Arrays (e.g. Droid's custom_models) concatenate, then dedup by
                // value so re-running launch never duplicates our entry or drops
                // the user's existing entries.
                result[k] = dedup(bv + ov)
            } else {
                result[k] = v
            }
        }
        return result
    }

    private func dedup(_ items: [Any]) -> [Any] {
        var seen = Set<String>()
        var out: [Any] = []
        for item in items {
            let key: String
            if JSONSerialization.isValidJSONObject(item),
               let d = try? JSONSerialization.data(withJSONObject: item, options: [.sortedKeys]),
               let s = String(data: d, encoding: .utf8) {
                key = s
            } else {
                key = String(describing: item)
            }
            if seen.insert(key).inserted { out.append(item) }
        }
        return out
    }

    private func runPreExec(_ commands: [[String]]) {
        for cmd in commands {
            guard let bin = cmd.first else { continue }
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            proc.arguments = cmd
            do {
                try proc.run()
                proc.waitUntilExit()
                if proc.terminationStatus != 0 {
                    print("warning: setup command failed (\(proc.terminationStatus)): \(cmd.joined(separator: " "))")
                }
            } catch {
                print("warning: could not run setup command '\(bin)': \(error)")
            }
        }
    }

    // MARK: - Exec

    /// Replace this process with the agent binary so it inherits the real TTY,
    /// stdin/stdout, and signals. On success this never returns. `argv` already
    /// includes the binary as argv[0], the resolved model args, and passthrough.
    private func execAgent(_ profile: AgentProfile, argv: [String]) throws {
        var cArgs: [UnsafeMutablePointer<CChar>?] = argv.map { strdup($0) }
        cArgs.append(nil)
        execvp(profile.binary, &cArgs)
        // Only reached if exec failed.
        let err = errno
        if err == ENOENT {
            print("\(profile.displayName) is not installed or not on PATH.")
            print(profile.notInstalledHint)
        } else {
            print("Failed to launch \(profile.binary): \(String(cString: strerror(err)))")
        }
        throw ExitCode.failure
    }

    // MARK: - Helpers

    private func expandTilde(_ path: String) -> String {
        guard path == "~" || path.hasPrefix("~/") else { return path }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        if path == "~" { return home }
        return home + String(path.dropFirst(1))
    }
}

enum LaunchError: Error, CustomStringConvertible {
    case invalidConfigFragment(String)
    var description: String {
        switch self {
        case .invalidConfigFragment(let path):
            return "Internal error: could not build a valid JSON config for \(path)."
        }
    }
}
