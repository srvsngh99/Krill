import ArgumentParser
import Foundation
import KrillLaunch
import KrillRegistry
import KrillServer
#if canImport(Darwin)
import Darwin
#endif

/// `krill ui` - the one-command path to the phone/web agent UI.
///
/// Makes sure a phone-reachable `krill serve` is running (starting one
/// detached if needed, so it survives closing the terminal), guarantees an API
/// key exists (generating and saving one to `~/.krill/config.toml` on first
/// run), prints the links to open on a phone, and opens the UI in the local
/// browser. `--install` makes the server a launchd login item so it is always
/// on; `--stop` / `--uninstall` undo the two modes.
struct UICommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ui",
        abstract: "Start the phone/web agent UI: serve in the background and print the links")

    @Option(name: .long, help: "Model to pre-load (default: config default_model, else the first installed; none = load on demand).")
    var model: String?

    @Option(name: .long, help: "Server port (default: config / 57455).")
    var port: Int?

    @Option(name: .long, help: "Bind address (default: 0.0.0.0 so phones can reach it; a loopback config host is overridden).")
    var host: String?

    @Flag(name: .long, help: "Do not open the UI in this Mac's browser.")
    var noOpen: Bool = false

    @Flag(name: .long, help: "Run the server in this terminal instead of detaching (Ctrl+C stops it).")
    var foreground: Bool = false

    @Flag(name: .long, help: "Install as a launchd login item: starts at login, restarts if it exits. Replaces any detached server.")
    var install: Bool = false

    @Flag(name: .long, help: "Remove the launchd login item and stop the server it runs.")
    var uninstall: Bool = false

    @Flag(name: .long, help: "Stop the detached server started by `krill ui`.")
    var stop: Bool = false

    @Flag(name: .long, help: "Print the links and key for the running server and exit.")
    var status: Bool = false

    // MARK: - Entry

    func run() async throws {
        let cfg = KrillConfig.load()
        let port = self.port ?? cfg.serverPort
        let baseURL = "http://127.0.0.1:\(port)"

        if uninstall { try uninstallLaunchAgent(baseURL: baseURL); return }
        if stop { try stopDetached(baseURL: baseURL); return }

        // The whole point is phone access: a loopback config host is replaced
        // by the wildcard bind (a non-loopback config host is kept as-is).
        let bindHost = self.host
            ?? (ServerSecurity.isLoopbackHost(cfg.serverHost) ? "0.0.0.0" : cfg.serverHost)
        let apiKey = try resolveAPIKey(cfg)
        let tailscale = KrillServer.tailscaleIP()
        let lan = UIBootstrap.lanIPv4Addresses()

        if status {
            guard await health(baseURL: baseURL) != nil else {
                print(Ansi.yellow("No Krill server running on port \(port).")
                      + Ansi.chrome("  Start one with:  krill ui"))
                throw ExitCode.failure
            }
            printBanner(port: port, lan: lan, tailscale: tailscale, apiKey: apiKey, mode: .running)
            return
        }

        let registry: Registry = {
            if let md = cfg.modelsDir, !md.isEmpty { return Registry(modelsDir: URL(fileURLWithPath: md)) }
            return Registry()
        }()
        let modelName = resolveModel(cfg, registry: registry)

        if install {
            try installLaunchAgent(host: bindHost, port: port, model: modelName, baseURL: baseURL)
            try await waitForLaunchAgent(baseURL: baseURL, model: modelName)
            printBanner(port: port, lan: lan, tailscale: tailscale, apiKey: apiKey, mode: .installed)
            openBrowser(port: port)
            return
        }

        if foreground {
            printBanner(port: port, lan: lan, tailscale: tailscale, apiKey: apiKey, mode: .foreground)
            var serve = ServeCommand()
            serve.host = bindHost
            serve.port = port
            serve.model = modelName
            serve.apiKey = apiKey
            // Open once the socket is up; the server blocks this task.
            if !noOpen {
                Task { [baseURL, port] in
                    for _ in 0..<180 {
                        try? await Task.sleep(nanoseconds: 1_000_000_000)
                        if await Self.probe(baseURL: baseURL) != nil {
                            Self.open(port: port); return
                        }
                    }
                }
            }
            try await serve.run()
            return
        }

        // Reuse a healthy server; otherwise start one detached.
        if await health(baseURL: baseURL) != nil {
            print("  " + Ansi.chrome("A Krill server is already running on port \(port); reusing it."))
            if let ip = lan.first, await Self.probe(baseURL: "http://\(ip):\(port)") == nil {
                print("  " + Ansi.yellow("It does not answer on \(ip) — it is probably bound to loopback only.")
                      + "\n  " + Ansi.chrome("Phones will not reach it. Stop it and run `krill ui` again to rebind on 0.0.0.0."))
            }
            printBanner(port: port, lan: lan, tailscale: tailscale, apiKey: apiKey, mode: .running)
            openBrowser(port: port)
            return
        }
        try await startDetached(host: bindHost, port: port, model: modelName, apiKey: apiKey, baseURL: baseURL)
        printBanner(port: port, lan: lan, tailscale: tailscale, apiKey: apiKey, mode: .detached)
        openBrowser(port: port)
    }

    // MARK: - Key + model resolution

    /// config.toml wins (that is what `krill serve` reads); else an explicit
    /// KRILL_API_KEY in the environment; else generate one. Anything not yet
    /// in config.toml is saved there so every later `krill serve` picks it up.
    private func resolveAPIKey(_ cfg: KrillConfig) throws -> String {
        if let k = ServerSecurity.normalizedAPIKey(cfg.serverAPIKey) { return k }
        let fromEnv = ServerSecurity.normalizedAPIKey(ProcessInfo.processInfo.environment["KRILL_API_KEY"])
        let key = fromEnv ?? UIBootstrap.generateAPIKey()
        try KrillConfig.set(key: "server_api_key", value: key)
        print("  " + Ansi.chrome(fromEnv != nil
            ? "Saved KRILL_API_KEY to ~/.krill/config.toml (server_api_key)."
            : "Generated an API key and saved it to ~/.krill/config.toml (server_api_key)."))
        return key
    }

    private func resolveModel(_ cfg: KrillConfig, registry: Registry) -> String? {
        if let m = model, !m.isEmpty { return m }
        if let d = cfg.defaultModel, !d.isEmpty, registry.hasModel(d) { return d }
        return registry.listModels().first?.name
    }

    // MARK: - Detached server

    private static var uiDir: String { NSHomeDirectory() + "/.krill/ui" }
    private static var logPath: String { uiDir + "/serve.log" }
    private static var pidPath: String { uiDir + "/serve.pid" }

    private static var krillPath: String {
        let raw = Bundle.main.executablePath ?? CommandLine.arguments[0]
        return URL(fileURLWithPath: raw).resolvingSymlinksInPath().path
    }

    private func startDetached(host: String, port: Int, model: String?, apiKey: String,
                               baseURL: String) async throws {
        try FileManager.default.createDirectory(atPath: Self.uiDir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: Self.logPath, contents: nil)
        guard let log = FileHandle(forWritingAtPath: Self.logPath) else {
            print("Error: could not open \(Self.logPath)"); throw ExitCode.failure
        }
        log.seekToEndOfFile()

        // nohup: the server must outlive this terminal (SIGHUP on close).
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/nohup")
        proc.arguments = [Self.krillPath] + UIBootstrap.serveArguments(host: host, port: port, model: model)
        proc.standardInput = FileHandle.nullDevice
        proc.standardOutput = log
        proc.standardError = log
        var env = ProcessInfo.processInfo.environment
        env["KRILL_API_KEY"] = apiKey
        proc.environment = env
        do { try proc.run() } catch {
            print("Error: could not start the server (\(error))."); throw ExitCode.failure
        }
        try? "\(proc.processIdentifier)\n".write(toFile: Self.pidPath, atomically: true, encoding: .utf8)

        let what = model.map { "Loading '\($0)'" } ?? "Starting the server"
        let spinner = Spinner("\(what) (first load can take a moment)")
        spinner.start()
        for _ in 0..<300 {
            if await health(baseURL: baseURL) != nil { await spinner.stop(); return }
            if !proc.isRunning {
                await spinner.stop()
                let text = (try? String(contentsOfFile: Self.logPath, encoding: .utf8)) ?? ""
                print(Ansi.yellow("The server exited before becoming ready. Last log lines:") + "\n"
                      + text.split(separator: "\n").suffix(8).joined(separator: "\n"))
                throw ExitCode.failure
            }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
        await spinner.stop()
        print(Ansi.yellow("The server did not become ready in time; see \(Self.logPath)."))
        throw ExitCode.failure
    }

    private func stopDetached(baseURL: String) throws {
        if FileManager.default.fileExists(atPath: UIBootstrap.launchAgentPlistPath()) {
            print(Ansi.yellow("The server is installed as a login item; launchd would restart it.")
                  + "\n  " + Ansi.chrome("Use:  krill ui --uninstall"))
            throw ExitCode.failure
        }
        guard let text = try? String(contentsOfFile: Self.pidPath, encoding: .utf8),
              let pid = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            print(Ansi.chrome("No detached server recorded (\(Self.pidPath))."))
            print(Ansi.chrome("If a `krill serve` is running in a terminal, stop it there with Ctrl+C."))
            return
        }
        if kill(pid, 0) != 0 {
            try? FileManager.default.removeItem(atPath: Self.pidPath)
            print(Ansi.chrome("The recorded server (pid \(pid)) is not running."))
            return
        }
        kill(pid, SIGTERM)
        try? FileManager.default.removeItem(atPath: Self.pidPath)
        print("Stopped the Krill server (pid \(pid)).")
    }

    // MARK: - launchd login item

    private func installLaunchAgent(host: String, port: Int, model: String?, baseURL: String) throws {
        // A detached server on the same port would collide; retire it first.
        if let text = try? String(contentsOfFile: Self.pidPath, encoding: .utf8),
           let pid = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)), kill(pid, 0) == 0 {
            kill(pid, SIGTERM)
            try? FileManager.default.removeItem(atPath: Self.pidPath)
            print("  " + Ansi.chrome("Stopped the detached server (pid \(pid)); launchd takes over."))
        }
        try FileManager.default.createDirectory(atPath: Self.uiDir, withIntermediateDirectories: true)
        let plistPath = UIBootstrap.launchAgentPlistPath()
        try FileManager.default.createDirectory(
            atPath: (plistPath as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        let plist = UIBootstrap.launchAgentPlist(
            krillPath: Self.krillPath,
            arguments: UIBootstrap.serveArguments(host: host, port: port, model: model),
            logPath: Self.logPath, workingDirectory: NSHomeDirectory())
        try plist.write(toFile: plistPath, atomically: true, encoding: .utf8)

        let domain = "gui/\(getuid())"
        _ = Self.launchctl(["bootout", domain, plistPath])   // ignore: not loaded yet
        let (code, out) = Self.launchctl(["bootstrap", domain, plistPath])
        guard code == 0 else {
            print(Ansi.yellow("launchctl bootstrap failed (\(code)): \(out)")); throw ExitCode.failure
        }
        print("  " + Ansi.chrome("Installed \(plistPath)"))
    }

    /// Poll until the launchd-run server answers. launchd gives no feedback of
    /// its own, so a server that never comes up is reported here, with the one
    /// cause we have actually hit: a binary inside a privacy-protected folder.
    private func waitForLaunchAgent(baseURL: String, model: String?) async throws {
        let what = model.map { "Loading '\($0)' under launchd" } ?? "Starting the server under launchd"
        let spinner = Spinner("\(what) (first load can take a moment)")
        spinner.start()
        for _ in 0..<300 {
            if await health(baseURL: baseURL) != nil { await spinner.stop(); return }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
        await spinner.stop()
        print(Ansi.yellow("The login item is installed but the server has not answered on port \(baseURL.split(separator: ":").last ?? "?")."))
        print("  " + Ansi.chrome("Log: \(Self.logPath)   Status: launchctl print gui/\(getuid())/\(UIBootstrap.launchAgentLabel)"))
        if Self.isInPrivacyProtectedFolder(Self.krillPath) {
            print("  " + Ansi.yellow("The krill binary is at \(Self.krillPath).")
                  + "\n  " + Ansi.chrome("macOS privacy protection stops launchd from running programs inside Desktop, Documents, or Downloads.")
                  + "\n  " + Ansi.chrome("Install it elsewhere (brew install krill, or make install → /usr/local/bin) and run `krill ui --install` again."))
        }
        throw ExitCode.failure
    }

    /// Desktop / Documents / Downloads (and iCloud Drive) are TCC-protected:
    /// launchd cannot execute from them without a per-app grant.
    static func isInPrivacyProtectedFolder(_ path: String) -> Bool {
        let home = NSHomeDirectory()
        let protected = ["Desktop", "Documents", "Downloads", "Library/Mobile Documents"]
            .map { "\(home)/\($0)/" }
        return protected.contains { path.hasPrefix($0) }
    }

    private func uninstallLaunchAgent(baseURL: String) throws {
        let plistPath = UIBootstrap.launchAgentPlistPath()
        guard FileManager.default.fileExists(atPath: plistPath) else {
            print(Ansi.chrome("No login item installed (\(plistPath))."))
            return
        }
        _ = Self.launchctl(["bootout", "gui/\(getuid())", plistPath])
        try FileManager.default.removeItem(atPath: plistPath)
        print("Removed the Krill login item; the server is stopped.")
    }

    private static func launchctl(_ args: [String]) -> (Int32, String) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe; p.standardError = pipe
        guard (try? p.run()) != nil else { return (-1, "could not run launchctl") }
        p.waitUntilExit()
        let out = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        return (p.terminationStatus, out.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    // MARK: - Health + browser

    private func health(baseURL: String) async -> [String: Any]? { await Self.probe(baseURL: baseURL) }

    /// `/healthz` is exempt from bearer auth, so no key is needed to probe.
    private static func probe(baseURL: String) async -> [String: Any]? {
        guard let url = URL(string: "\(baseURL)/healthz") else { return nil }
        var req = URLRequest(url: url)
        req.timeoutInterval = 2
        guard let (data, resp) = try? await URLSession.shared.data(for: req),
              (resp as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private func openBrowser(port: Int) { if !noOpen { Self.open(port: port) } }

    private static func open(port: Int) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        p.arguments = ["http://localhost:\(port)/ui"]
        p.standardOutput = FileHandle.nullDevice; p.standardError = FileHandle.nullDevice
        try? p.run()
    }

    // MARK: - Banner

    private enum Mode { case running, detached, foreground, installed }

    private func printBanner(port: Int, lan: [String], tailscale: String?, apiKey: String, mode: Mode) {
        let rows = UIBootstrap.endpoints(port: port, lanIPs: lan, tailscaleIP: tailscale)
        let phone = rows.last(where: { $0.label == "Tailscale" }) ?? rows.last(where: { $0.label == "Same Wi-Fi" })
        print("")
        print("  " + Ansi.ember(Ansi.bold(">_ ")) + Ansi.bold("Krill UI"))
        for r in rows {
            print("  " + Ansi.chrome(r.label.padding(toLength: 12, withPad: " ", startingAt: 0)) + r.url)
        }
        if let phone {
            print("  " + Ansi.chrome("Phone link  ") + UIBootstrap.phoneLink(url: phone.url, apiKey: apiKey))
            print("  " + Ansi.hint("            (carries the key; open it on the phone, then Share → Add to Home Screen)"))
        } else {
            print("  " + Ansi.yellow("No LAN or Tailscale address found — phones cannot reach this Mac right now."))
        }
        print("  " + Ansi.chrome("API key     ") + apiKey + Ansi.hint("   (~/.krill/config.toml → server_api_key)"))
        switch mode {
        case .running:
            print("  " + Ansi.chrome("Server      ") + "already running on port \(port)")
        case .detached:
            print("  " + Ansi.chrome("Server      ") + "running in the background; log \(Self.logPath)")
            print("  " + Ansi.chrome("Stop        ") + "krill ui --stop" + Ansi.hint("      always-on at login: krill ui --install"))
        case .foreground:
            print("  " + Ansi.chrome("Server      ") + "starting in this terminal (Ctrl+C stops it)")
        case .installed:
            print("  " + Ansi.chrome("Server      ") + "login item installed; starts at login, restarts on exit; log \(Self.logPath)")
            print("  " + Ansi.chrome("Remove      ") + "krill ui --uninstall")
        }
        if tailscale == nil {
            print("  " + Ansi.hint("Tip: Tailscale on Mac + phone makes the link work from anywhere (free): https://tailscale.com"))
        }
        print("")
    }
}
