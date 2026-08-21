import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Pure helpers behind `krill ui` — the one-command path to a phone-reachable
/// server: key generation, the URL list the banner prints, the launchd plist
/// for the always-on install, and the LAN address discovery. No I/O beyond
/// `getifaddrs`, so `UICommand` stays a thin orchestrator and this is testable.
public enum UIBootstrap {

    /// launchd label for the always-on server (`krill ui --install`).
    public static let launchAgentLabel = "ai.souravailabs.krill.serve"

    /// Where the always-on plist lives.
    public static func launchAgentPlistPath(home: String = NSHomeDirectory()) -> String {
        "\(home)/Library/LaunchAgents/\(launchAgentLabel).plist"
    }

    /// A fresh bearer token: 24 random bytes, base64url (no padding, no
    /// whitespace) so it satisfies `ServerSecurity.normalizedAPIKey` and is
    /// safe in a URL fragment and in TOML.
    public static func generateAPIKey(byteCount: Int = 24) -> String {
        var bytes = [UInt8](repeating: 0, count: byteCount)
        for i in bytes.indices { bytes[i] = UInt8.random(in: .min ... .max) }
        return Data(bytes).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    /// One row of the startup banner.
    public struct Endpoint: Equatable, Sendable {
        public let label: String
        public let url: String
        public init(label: String, url: String) { self.label = label; self.url = url }
    }

    /// The `/ui` URLs to print, most local first: this Mac, each LAN address,
    /// then Tailscale. Addresses are deduplicated and order is preserved.
    public static func endpoints(port: Int, lanIPs: [String], tailscaleIP: String?) -> [Endpoint] {
        var out = [Endpoint(label: "This Mac", url: "http://localhost:\(port)/ui")]
        var seen: Set<String> = []
        for ip in lanIPs where seen.insert(ip).inserted {
            out.append(Endpoint(label: "Same Wi-Fi", url: "http://\(ip):\(port)/ui"))
        }
        if let ts = tailscaleIP, !ts.isEmpty, seen.insert(ts).inserted {
            out.append(Endpoint(label: "Tailscale", url: "http://\(ts):\(port)/ui"))
        }
        return out
    }

    /// The phone link: the UI reads `#k=<key>` on load, stores the key, and
    /// strips it from the address bar. Fragments never leave the browser, so
    /// the key is not sent over the wire by the link itself.
    public static func phoneLink(url: String, apiKey: String?) -> String {
        guard let apiKey, !apiKey.isEmpty else { return url }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_.~"))
        let enc = apiKey.addingPercentEncoding(withAllowedCharacters: allowed) ?? apiKey
        return "\(url)#k=\(enc)"
    }

    /// The launchd property list for an always-on `krill serve`. The API key
    /// is NOT embedded: the server reads `server_api_key` from
    /// `~/.krill/config.toml`, so there is a single copy of the secret.
    public static func launchAgentPlist(
        krillPath: String, arguments: [String], logPath: String, workingDirectory: String
    ) -> String {
        func x(_ s: String) -> String {
            s.replacingOccurrences(of: "&", with: "&amp;")
             .replacingOccurrences(of: "<", with: "&lt;")
             .replacingOccurrences(of: ">", with: "&gt;")
        }
        let argv = ([krillPath] + arguments).map { "        <string>\(x($0))</string>" }
            .joined(separator: "\n")
        return """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>\(launchAgentLabel)</string>
            <key>ProgramArguments</key>
            <array>
        \(argv)
            </array>
            <key>WorkingDirectory</key>
            <string>\(x(workingDirectory))</string>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>ProcessType</key>
            <string>Interactive</string>
            <key>StandardOutPath</key>
            <string>\(x(logPath))</string>
            <key>StandardErrorPath</key>
            <string>\(x(logPath))</string>
        </dict>
        </plist>

        """
    }

    /// The `krill serve` argv for a phone-reachable server.
    public static func serveArguments(host: String, port: Int, model: String?) -> [String] {
        var args = ["serve", "--host", host, "--port", "\(port)"]
        if let model, !model.isEmpty { args += ["--model", model] }
        return args
    }

    /// Non-loopback IPv4 addresses of real network interfaces (Wi-Fi /
    /// Ethernet). Tunnels (`utun*`, which is where Tailscale lives), AWDL and
    /// bridges are skipped — Tailscale is reported separately, and the rest
    /// are not phone-reachable.
    public static func lanIPv4Addresses() -> [String] {
        var list: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&list) == 0, let head = list else { return [] }
        defer { freeifaddrs(list) }
        var out: [String] = []
        var node: UnsafeMutablePointer<ifaddrs>? = head
        while let cur = node {
            node = cur.pointee.ifa_next
            guard let sa = cur.pointee.ifa_addr, Int32(sa.pointee.sa_family) == AF_INET else { continue }
            let flags = Int32(cur.pointee.ifa_flags)
            guard flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0 else { continue }
            let name = String(cString: cur.pointee.ifa_name)
            guard isPhoneReachableInterface(name) else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            guard getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count),
                              nil, 0, NI_NUMERICHOST) == 0 else { continue }
            let ip = String(cString: host)
            if ip.hasPrefix("169.254.") { continue }   // link-local, no DHCP
            out.append(ip)
        }
        return out
    }

    /// Interface-name filter used by `lanIPv4Addresses` (exposed for tests).
    public static func isPhoneReachableInterface(_ name: String) -> Bool {
        for prefix in ["utun", "awdl", "llw", "bridge", "gif", "stf", "ipsec", "ppp", "vmnet", "anpi", "ap"] {
            if name.hasPrefix(prefix) { return false }
        }
        return true
    }
}
