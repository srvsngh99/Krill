import XCTest
import Foundation
@testable import KLMRegistry

/// Pins the Swift `KrillLMVersion` constant to the value in the
/// repo-root `VERSION` file. The CLI, server, and Ollama-compat
/// payload all read from `KrillLMVersion`, so any drift between
/// the file and the constant produces an inconsistent advertised
/// version. This test runs in regular CI and fails loudly on
/// release-prep PRs that forget to update one of the two surfaces.
///
/// Failure mode: bump VERSION without updating
/// `Sources/KLMRegistry/KrillLMVersion.swift` (or vice versa) and
/// this test reports the exact mismatch.
final class KrillLMVersionTests: XCTestCase {

    /// Walk upward from the test file location to find the repo
    /// root - the directory that contains `VERSION` and
    /// `Package.swift`. Returns nil if the harness ran the test
    /// from an unexpected location (e.g. a build cache copy with
    /// no parent VERSION); the test then skips rather than fails.
    private func repoVersion(file: StaticString = #filePath) -> String? {
        let testFile = URL(fileURLWithPath: "\(file)")
        var dir = testFile.deletingLastPathComponent()
        for _ in 0 ..< 6 {
            let candidate = dir.appendingPathComponent("VERSION")
            if FileManager.default.fileExists(atPath: candidate.path),
               let data = try? Data(contentsOf: candidate),
               let raw = String(data: data, encoding: .utf8) {
                return raw.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            dir.deleteLastPathComponent()
        }
        return nil
    }

    func testKrillLMVersionMatchesVersionFile() throws {
        guard let onDisk = repoVersion() else {
            throw XCTSkip("VERSION file not reachable from test location")
        }
        XCTAssertEqual(
            KrillLMVersion, onDisk,
            "Swift `KrillLMVersion` (\(KrillLMVersion)) disagrees with "
            + "repo-root `VERSION` (\(onDisk)). Update both at the "
            + "same time when cutting a release - bump VERSION and "
            + "Sources/KLMRegistry/KrillLMVersion.swift together.")
    }

    func testKrillLMVersionTagHasVPrefix() {
        XCTAssertTrue(
            KrillLMVersionTag.hasPrefix("v"),
            "KrillLMVersionTag must be a git-tag-shaped string "
            + "(`v<version>`); got `\(KrillLMVersionTag)`.")
        XCTAssertEqual(
            KrillLMVersionTag.dropFirst(), Substring(KrillLMVersion),
            "KrillLMVersionTag must be exactly `v` + KrillLMVersion.")
    }
}
