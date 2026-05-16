import ArgumentParser
import Foundation
import KLMRegistry

/// `krillm stop <model>` — ask a running server to unload the model now
/// (WS-E / T2-3). Talks to the local server's `/v1/models/unload`.
struct StopCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "stop",
        abstract: "Unload a running model from the server (frees memory)"
    )

    @Argument(help: "Model name (informational; the server unloads the active model)")
    var model: String?

    @Option(name: .long, help: "Server host")
    var host: String?

    @Option(name: .long, help: "Server port")
    var port: Int?

    func run() async throws {
        let cfg = KrillConfig.load()
        let h = host ?? cfg.serverHost
        let p = port ?? cfg.serverPort
        guard let url = URL(string: "http://\(h):\(p)/v1/models/unload") else {
            print("Error: bad server URL")
            throw ExitCode.failure
        }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = Data("{}".utf8)
        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            if code == 200 {
                print("Stopped\(model.map { " '\($0)'" } ?? "").")
            } else {
                let body = String(data: data, encoding: .utf8) ?? ""
                print("Server returned \(code): \(body)")
                throw ExitCode.failure
            }
        } catch {
            print("Error: could not reach server at \(h):\(p) — is `krillm serve` running? (\(error))")
            throw ExitCode.failure
        }
    }
}
