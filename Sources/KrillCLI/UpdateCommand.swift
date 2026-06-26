import ArgumentParser
import Foundation
import KrillRegistry

/// `krill update` — self-update to the latest published release.
///
/// Checks GitHub's `releases/latest`, compares it to the built-in
/// `KrillVersion`, and (if newer) re-runs the project's `install.sh`
/// pinned to the target version. The actual install logic is NOT
/// duplicated here: install.sh already handles the metallib, the
/// mlx-swift bundle, Gatekeeper quarantine, sudo escalation and PATH
/// linking, so this command only does the *version check* and then
/// delegates to it.
///
/// Homebrew installs are detected and redirected to `brew upgrade`,
/// because running install.sh over a brew-managed prefix would leave
/// the Cellar and the symlink out of sync.
struct UpdateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "update",
        abstract: "Update Krill to the latest release",
        discussion: """
            Checks for a newer Krill release and installs it in place.

            By default it self-updates via the official installer. If Krill
            was installed with Homebrew, it tells you to run `brew upgrade
            krill` instead. Use `--check` to only report whether an update
            is available, or `--version` to install a specific version.
            """
    )

    @Flag(name: .long, help: "Only check for an update; don't install anything.")
    var check = false

    @Flag(name: .long, help: "Reinstall even if already on the latest version (and bypass the Homebrew guard).")
    var force = false

    @Option(name: .long, help: "Install a specific version instead of the latest (e.g. 0.14.0).")
    var version: String?

    private static let repo = "srvsngh99/Krill"
    private static let installURL = "https://raw.githubusercontent.com/srvsngh99/Krill/main/install.sh"

    func run() async throws {
        let current = KrillVersion
        print("Current version: \(current)")

        // Resolve the target version: explicit --version, or the latest release.
        let target: String
        if let pinned = version {
            target = pinned.hasPrefix("v") ? String(pinned.dropFirst()) : pinned
        } else {
            print("Checking for updates...")
            target = try await latestReleaseVersion()
            print("Latest version:  \(target)")
        }

        // Up-to-date short-circuit (unless the user pinned a version or forced it).
        if version == nil && !force && compareSemver(current, target) >= 0 {
            print("You're on the latest version. \u{2713}")
            return
        }

        if version == nil && compareSemver(current, target) < 0 {
            print("An update is available: \(current) \u{2192} \(target)")
        }

        if check {
            // Report-only mode: never installs.
            if version != nil {
                print("Run `krill update --version \(target)` without --check to install it.")
            } else if compareSemver(current, target) < 0 {
                print("Run `krill update` to install it.")
            }
            return
        }

        // Homebrew-managed installs must update through brew, not install.sh.
        if isHomebrewManaged() && !force {
            print("")
            print("Krill was installed with Homebrew. Update it with:")
            print("    brew update && brew upgrade krill")
            print("")
            print("(Re-run with --force to override and use the installer anyway.)")
            throw ExitCode.failure
        }

        print("")
        print("Installing Krill \(target)...")
        try runInstaller(version: target)
        print("")
        print("Done. Run `krill version` to confirm.")
    }

    /// GET the latest release tag from the GitHub API and return it
    /// without the leading "v".
    private func latestReleaseVersion() async throws -> String {
        let url = URL(string: "https://api.github.com/repos/\(Self.repo)/releases/latest")!
        var req = URLRequest(url: url)
        req.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        req.setValue("krill-update/\(KrillVersion)", forHTTPHeaderField: "User-Agent")

        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw ValidationError("Could not reach GitHub to check for updates (HTTP \(code)).")
        }
        guard
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let tag = json["tag_name"] as? String
        else {
            throw ValidationError("Unexpected response from GitHub while checking for updates.")
        }
        return tag.hasPrefix("v") ? String(tag.dropFirst()) : tag
    }

    /// Re-run install.sh pinned to `version` via `curl … | sh`, reusing
    /// the same path documented in the README.
    private func runInstaller(version: String) throws {
        var env = ProcessInfo.processInfo.environment
        env["KRILL_VERSION"] = version

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/sh")
        proc.arguments = ["-c", "curl -fsSL \(Self.installURL) | sh"]
        proc.environment = env

        try proc.run()
        proc.waitUntilExit()
        if proc.terminationStatus != 0 {
            throw ExitCode(proc.terminationStatus)
        }
    }

    /// True when the running binary lives under a Homebrew Cellar, i.e.
    /// it was installed via the tap and should be updated with brew.
    private func isHomebrewManaged() -> Bool {
        guard let path = Bundle.main.executablePath else { return false }
        let resolved = URL(fileURLWithPath: path).resolvingSymlinksInPath().path
        return resolved.contains("/Cellar/")
    }
}

/// Compare two dotted version strings (e.g. "0.13.0" vs "0.14.1").
/// Returns -1 if `a < b`, 0 if equal, 1 if `a > b`. Missing components
/// are treated as 0, and any non-numeric/pre-release suffix is ignored.
func compareSemver(_ a: String, _ b: String) -> Int {
    func parts(_ s: String) -> [Int] {
        s.split(whereSeparator: { $0 == "." || $0 == "-" || $0 == "+" })
            .prefix(3)
            .map { Int($0) ?? 0 }
    }
    let pa = parts(a), pb = parts(b)
    for i in 0..<3 {
        let x = i < pa.count ? pa[i] : 0
        let y = i < pb.count ? pb[i] : 0
        if x != y { return x < y ? -1 : 1 }
    }
    return 0
}
