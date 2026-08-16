import XCTest
@testable import KrillHarness

/// Coverage for `AgentWorkspace`, the task-local workspace root that lets a
/// hosting server (`krill serve`'s agent sessions) bind a per-session working
/// directory around a run, while the CLI/TUI surfaces (which never bind it)
/// keep resolving against the process cwd exactly as before.
final class AgentWorkspaceTests: XCTestCase {
    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-agent-workspace-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - FileToolSupport.resolve

    func testResolveRelativePathBoundToWorkspaceRoot() throws {
        let tempDir = try makeTempDir()
        let resolved = AgentWorkspace.$root.withValue(tempDir) {
            FileToolSupport.resolve("foo.txt")
        }
        XCTAssertEqual(
            resolved.standardizedFileURL.path,
            tempDir.appendingPathComponent("foo.txt").standardizedFileURL.path)
    }

    func testResolveRelativePathUnboundFallsBackToProcessCwd() {
        // No AgentWorkspace.$root bound: must resolve against the process cwd,
        // exactly the CLI/TUI behavior before AgentWorkspace existed.
        let resolved = FileToolSupport.resolve("foo.txt")
        let expected = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("foo.txt").standardizedFileURL
        XCTAssertEqual(resolved.standardizedFileURL.path, expected.path)
    }

    func testResolveAbsolutePathIgnoresBoundWorkspace() throws {
        let tempDir = try makeTempDir()
        let resolved = AgentWorkspace.$root.withValue(tempDir) {
            FileToolSupport.resolve("/etc/hosts")
        }
        XCTAssertEqual(resolved.path, "/etc/hosts")
    }

    // MARK: - AgentEnvironment.contextLine

    func testContextLineContainsBoundWorkspacePath() throws {
        let tempDir = try makeTempDir()
        let line = AgentWorkspace.$root.withValue(tempDir) {
            AgentEnvironment.contextLine()
        }
        XCTAssertTrue(
            line.contains(tempDir.standardizedFileURL.path),
            "context line must surface the bound workspace path, got: \(line)")
    }

    func testContextLineUnboundContainsProcessCwd() {
        let line = AgentEnvironment.contextLine()
        XCTAssertTrue(line.contains(FileManager.default.currentDirectoryPath))
    }

    // MARK: - AgentEnvironment.projectBrief

    func testProjectBriefReadsKrillMdFromBoundWorkspace() async throws {
        let tempDir = try makeTempDir()
        let brief = "# Test Project\nBuild with `swift build`."
        try brief.write(
            to: tempDir.appendingPathComponent("Krill.md"), atomically: true, encoding: .utf8)

        let result = AgentWorkspace.$root.withValue(tempDir) {
            AgentEnvironment.projectBrief()
        }
        XCTAssertNotNil(result)
        XCTAssertTrue(result?.contains("Test Project") ?? false, "got: \(result ?? "nil")")
        XCTAssertTrue(result?.contains("swift build") ?? false)
    }

    func testProjectBriefNilWhenKrillMdAbsent() throws {
        let tempDir = try makeTempDir()
        let result = AgentWorkspace.$root.withValue(tempDir) {
            AgentEnvironment.projectBrief()
        }
        XCTAssertNil(result, "no Krill.md in the bound workspace should yield nil")
    }

    // MARK: - BashTool

    func testBashToolPwdRunsInsideBoundWorkspace() async throws {
        let tempDir = try makeTempDir()
        let result = await AgentWorkspace.$root.withValue(tempDir) {
            await BashTool().run(argumentsJSON: #"{"command":"pwd"}"#)
        }
        XCTAssertFalse(result.isError)
        let output = result.content.trimmingCharacters(in: .whitespacesAndNewlines)
        // Normalize /private/tmp vs /tmp (macOS symlinks TMPDIR under /private).
        let normalizedOutput = URL(fileURLWithPath: output).resolvingSymlinksInPath().path
        let normalizedExpected = tempDir.resolvingSymlinksInPath().path
        XCTAssertEqual(normalizedOutput, normalizedExpected)
    }

    func testBashToolPwdUnboundRunsInProcessCwd() async {
        let result = await BashTool().run(argumentsJSON: #"{"command":"pwd"}"#)
        XCTAssertFalse(result.isError)
        let output = result.content.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedOutput = URL(fileURLWithPath: output).resolvingSymlinksInPath().path
        let normalizedExpected = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .resolvingSymlinksInPath().path
        XCTAssertEqual(normalizedOutput, normalizedExpected)
    }
}
