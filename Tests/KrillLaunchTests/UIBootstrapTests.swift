import Foundation
import XCTest
@testable import KrillLaunch

/// Pure-helper coverage for `krill ui` (UIBootstrap): key shape, banner URL
/// list, phone link, launchd plist, serve argv, interface filter.
final class UIBootstrapTests: XCTestCase {

    func testGeneratedAPIKeyIsURLAndTOMLSafeAndUnique() {
        let a = UIBootstrap.generateAPIKey()
        let b = UIBootstrap.generateAPIKey()
        XCTAssertNotEqual(a, b)
        XCTAssertEqual(a.count, 32)   // 24 bytes -> 32 base64url chars, no padding
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        XCTAssertTrue(a.unicodeScalars.allSatisfy { allowed.contains($0) }, a)
    }

    func testEndpointsOrderLocalThenLanThenTailscaleAndDedupe() {
        let rows = UIBootstrap.endpoints(
            port: 57455, lanIPs: ["192.168.1.5", "192.168.1.5", "10.0.0.2"], tailscaleIP: "100.91.59.7")
        XCTAssertEqual(rows.map(\.label), ["This Mac", "Same Wi-Fi", "Same Wi-Fi", "Tailscale"])
        XCTAssertEqual(rows.map(\.url), [
            "http://localhost:57455/ui", "http://192.168.1.5:57455/ui",
            "http://10.0.0.2:57455/ui", "http://100.91.59.7:57455/ui",
        ])
    }

    func testEndpointsWithoutTailscaleOrLanIsJustLocal() {
        let rows = UIBootstrap.endpoints(port: 8080, lanIPs: [], tailscaleIP: nil)
        XCTAssertEqual(rows, [UIBootstrap.Endpoint(label: "This Mac", url: "http://localhost:8080/ui")])
        XCTAssertEqual(UIBootstrap.endpoints(port: 8080, lanIPs: [], tailscaleIP: "").count, 1)
    }

    func testPhoneLinkCarriesKeyInFragmentPercentEncoded() {
        XCTAssertEqual(
            UIBootstrap.phoneLink(url: "http://h:1/ui", apiKey: "abc-_.~"),
            "http://h:1/ui#k=abc-_.~")
        XCTAssertEqual(
            UIBootstrap.phoneLink(url: "http://h:1/ui", apiKey: "a b&c"),
            "http://h:1/ui#k=a%20b%26c")
        XCTAssertEqual(UIBootstrap.phoneLink(url: "http://h:1/ui", apiKey: nil), "http://h:1/ui")
        XCTAssertEqual(UIBootstrap.phoneLink(url: "http://h:1/ui", apiKey: ""), "http://h:1/ui")
    }

    func testServeArgumentsIncludeModelOnlyWhenGiven() {
        XCTAssertEqual(
            UIBootstrap.serveArguments(host: "0.0.0.0", port: 57455, model: "gemma-4-e2b"),
            ["serve", "--host", "0.0.0.0", "--port", "57455", "--model", "gemma-4-e2b"])
        XCTAssertEqual(
            UIBootstrap.serveArguments(host: "0.0.0.0", port: 1, model: nil),
            ["serve", "--host", "0.0.0.0", "--port", "1"])
        XCTAssertEqual(UIBootstrap.serveArguments(host: "h", port: 1, model: "").count, 5)
    }

    func testLaunchAgentPlistIsValidAndCarriesNoSecret() throws {
        let text = UIBootstrap.launchAgentPlist(
            krillPath: "/usr/local/bin/krill",
            arguments: ["serve", "--host", "0.0.0.0", "--port", "57455", "--model", "a<b&c"],
            logPath: "/Users/me/.krill/ui/serve.log", workingDirectory: "/Users/me")
        let obj = try PropertyListSerialization.propertyList(from: Data(text.utf8), format: nil)
        let dict = try XCTUnwrap(obj as? [String: Any])
        XCTAssertEqual(dict["Label"] as? String, UIBootstrap.launchAgentLabel)
        XCTAssertEqual(dict["ProgramArguments"] as? [String],
                       ["/usr/local/bin/krill", "serve", "--host", "0.0.0.0", "--port", "57455", "--model", "a<b&c"])
        XCTAssertEqual(dict["RunAtLoad"] as? Bool, true)
        XCTAssertEqual(dict["KeepAlive"] as? Bool, true)
        XCTAssertEqual(dict["StandardOutPath"] as? String, "/Users/me/.krill/ui/serve.log")
        XCTAssertEqual(dict["WorkingDirectory"] as? String, "/Users/me")
        XCTAssertNil(dict["EnvironmentVariables"], "the key lives in config.toml only")
        XCTAssertTrue(UIBootstrap.launchAgentPlistPath(home: "/Users/me")
            .hasSuffix("/Library/LaunchAgents/\(UIBootstrap.launchAgentLabel).plist"))
    }

    func testInterfaceFilterSkipsTunnelsAndKeepsWifiEthernet() {
        XCTAssertTrue(UIBootstrap.isPhoneReachableInterface("en0"))
        XCTAssertTrue(UIBootstrap.isPhoneReachableInterface("en5"))
        for bad in ["utun4", "awdl0", "llw0", "bridge0", "vmnet8", "anpi0", "ap1"] {
            XCTAssertFalse(UIBootstrap.isPhoneReachableInterface(bad), bad)
        }
    }

    func testLanAddressesAreNonLoopbackDottedQuads() {
        for ip in UIBootstrap.lanIPv4Addresses() {
            XCTAssertFalse(ip.hasPrefix("127."), ip)
            XCTAssertFalse(ip.hasPrefix("169.254."), ip)
            XCTAssertEqual(ip.split(separator: ".").count, 4, ip)
        }
    }
}
