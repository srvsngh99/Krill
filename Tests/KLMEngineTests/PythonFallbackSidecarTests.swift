import XCTest
@testable import KLMEngine

/// Tests for the persistent Python sidecar. These use a small mock helper
/// script (no mlx-vlm dependency) injected via `KLM_MLX_VLM_SIDECAR`.
final class PythonFallbackSidecarTests: XCTestCase {

    private var tempDir: URL!
    private var savedEnv: String?

    override func setUpWithError() throws {
        try super.setUpWithError()
        if !pythonAvailable() {
            throw XCTSkip("python3 is not available on PATH")
        }
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("klm-sidecar-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        savedEnv = ProcessInfo.processInfo.environment["KLM_MLX_VLM_SIDECAR"]
    }

    override func tearDownWithError() throws {
        if let env = savedEnv {
            setenv("KLM_MLX_VLM_SIDECAR", env, 1)
        } else {
            unsetenv("KLM_MLX_VLM_SIDECAR")
        }
        try? FileManager.default.removeItem(at: tempDir)
        try super.tearDownWithError()
    }

    // MARK: - Mock helper scripts

    /// Echo helper: returns `{"id": <id>, "ok": true, "output": "ok:<id>"}`
    /// for every request and records each request line to a counter file so
    /// the test can verify single-spawn behavior.
    private func writeEchoHelper(counterFile: URL) throws -> URL {
        let path = tempDir.appendingPathComponent("echo_helper.py")
        let body = """
        #!/usr/bin/env python3
        import json, os, sys
        sys.stderr.write("READY\\n")
        sys.stderr.flush()
        counter_path = os.environ.get("KLM_TEST_COUNTER_FILE", "")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if counter_path:
                with open(counter_path, "a") as f:
                    f.write(line + "\\n")
            try:
                req = json.loads(line)
            except Exception as exc:
                sys.stdout.write(json.dumps({"id": "", "ok": False, "error": str(exc)}) + "\\n")
                sys.stdout.flush()
                continue
            sys.stdout.write(json.dumps({
                "id": req.get("id", ""),
                "ok": True,
                "output": "ok:" + req.get("id", ""),
            }) + "\\n")
            sys.stdout.flush()
        """
        try body.write(to: path, atomically: true, encoding: .utf8)
        // Ensure counter file exists so we can read it later.
        FileManager.default.createFile(atPath: counterFile.path, contents: Data())
        return path
    }

    /// Crash helper: handles the first request normally, then exits with
    /// status 1 instead of replying to subsequent requests.
    private func writeCrashHelper() throws -> URL {
        let path = tempDir.appendingPathComponent("crash_helper.py")
        let body = """
        #!/usr/bin/env python3
        import json, sys
        sys.stderr.write("READY\\n")
        sys.stderr.flush()
        served = 0
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception:
                sys.exit(1)
            served += 1
            if served == 1:
                sys.stdout.write(json.dumps({
                    "id": req.get("id", ""),
                    "ok": True,
                    "output": "first",
                }) + "\\n")
                sys.stdout.flush()
                continue
            # Crash on subsequent requests without responding.
            sys.exit(1)
        """
        try body.write(to: path, atomically: true, encoding: .utf8)
        return path
    }

    // MARK: - Tests

    func testHelperSpawnsOnceAndIsReusedAcrossCalls() async throws {
        let counterFile = tempDir.appendingPathComponent("counter.log")
        let script = try writeEchoHelper(counterFile: counterFile)
        setenv("KLM_MLX_VLM_SIDECAR", script.path, 1)
        setenv("KLM_TEST_COUNTER_FILE", counterFile.path, 1)
        defer { unsetenv("KLM_TEST_COUNTER_FILE") }

        let modelPath = tempDir.appendingPathComponent("model-\(UUID().uuidString)").path
        await HelperRegistry.shared.reset(modelPath: modelPath)

        let fallback = PythonFallback(modelPath: modelPath)

        for i in 0..<3 {
            let out = try await fallback.generate(prompt: "hello \(i)", maxTokens: 8)
            XCTAssertTrue(out.hasPrefix("ok:req-"), "unexpected output: \(out)")
        }

        // Counter file should have exactly 3 lines (one per request) — proving
        // the helper persisted across calls rather than respawning.
        let logged = (try? String(contentsOf: counterFile)) ?? ""
        let lineCount = logged.split(separator: "\n").count
        XCTAssertEqual(lineCount, 3, "expected 3 logged requests, got \(lineCount): \(logged)")

        await HelperRegistry.shared.reset(modelPath: modelPath)
    }

    func testConcurrentCallsAreCorrelatedById() async throws {
        let counterFile = tempDir.appendingPathComponent("counter.log")
        let script = try writeEchoHelper(counterFile: counterFile)
        setenv("KLM_MLX_VLM_SIDECAR", script.path, 1)
        setenv("KLM_TEST_COUNTER_FILE", counterFile.path, 1)
        defer { unsetenv("KLM_TEST_COUNTER_FILE") }

        let modelPath = tempDir.appendingPathComponent("model-\(UUID().uuidString)").path
        await HelperRegistry.shared.reset(modelPath: modelPath)
        let fallback = PythonFallback(modelPath: modelPath)

        let results = try await withThrowingTaskGroup(of: String.self) { group -> [String] in
            for i in 0..<5 {
                group.addTask {
                    try await fallback.generate(prompt: "concurrent \(i)", maxTokens: 4)
                }
            }
            var collected: [String] = []
            for try await r in group { collected.append(r) }
            return collected
        }

        XCTAssertEqual(results.count, 5)
        // Every result should match the id-echo pattern, and ids must be unique.
        let ids = Set(results)
        XCTAssertEqual(ids.count, 5, "expected 5 unique correlated responses, got \(results)")
        for r in results {
            XCTAssertTrue(r.hasPrefix("ok:req-"), "unexpected output: \(r)")
        }

        await HelperRegistry.shared.reset(modelPath: modelPath)
    }

    func testHelperCrashIsSurfacedAndNextCallRespawns() async throws {
        let script = try writeCrashHelper()
        setenv("KLM_MLX_VLM_SIDECAR", script.path, 1)

        let modelPath = tempDir.appendingPathComponent("model-\(UUID().uuidString)").path
        await HelperRegistry.shared.reset(modelPath: modelPath)
        let fallback = PythonFallback(modelPath: modelPath)

        let first = try await fallback.generate(prompt: "first", maxTokens: 4)
        XCTAssertEqual(first, "first")

        // Second request triggers a crash inside the crash helper. The Swift
        // side should surface it as a clear error rather than hanging.
        do {
            _ = try await fallback.generate(prompt: "boom", maxTokens: 4)
            XCTFail("expected helper crash to throw")
        } catch let err as FallbackError {
            switch err {
            case .helperCrashed, .pythonFailed:
                break
            default:
                XCTFail("unexpected error: \(err)")
            }
        }

        // Swap to the echo helper so the next call after the crash
        // demonstrates a fresh helper is spawned (the registry should have
        // dropped the dead one when its stdout closed).
        let counterFile = tempDir.appendingPathComponent("counter-after-crash.log")
        let echo = try writeEchoHelper(counterFile: counterFile)
        setenv("KLM_MLX_VLM_SIDECAR", echo.path, 1)

        let recovered = try await fallback.generate(prompt: "recover", maxTokens: 4)
        XCTAssertTrue(recovered.hasPrefix("ok:req-"),
                      "expected respawned helper to answer; got \(recovered)")

        await HelperRegistry.shared.reset(modelPath: modelPath)
    }

    func testMissingHelperScriptYieldsClearError() async throws {
        // Point at a path that doesn't exist; resolveHelperScriptPath checks
        // file existence before honoring the override, so this falls through
        // the cwd-walk. The test's cwd may legitimately resolve a real
        // sidecar.py from the repo, so we sandbox into a temp cwd that has
        // no `tools/` subtree.
        unsetenv("KLM_MLX_VLM_SIDECAR")
        let isolated = tempDir.appendingPathComponent("isolated-cwd")
        try FileManager.default.createDirectory(at: isolated, withIntermediateDirectories: true)

        let prevCwd = FileManager.default.currentDirectoryPath
        FileManager.default.changeCurrentDirectoryPath(isolated.path)
        defer { FileManager.default.changeCurrentDirectoryPath(prevCwd) }

        let modelPath = tempDir.appendingPathComponent("model-missing-\(UUID().uuidString)").path
        await HelperRegistry.shared.reset(modelPath: modelPath)
        let fallback = PythonFallback(modelPath: modelPath)
        do {
            _ = try await fallback.generate(prompt: "x", maxTokens: 1)
            XCTFail("expected missing helper to throw")
        } catch let err as FallbackError {
            switch err {
            case .helperNotFound(let msg):
                XCTAssertTrue(msg.contains("KLM_MLX_VLM_SIDECAR"),
                              "error should mention env override: \(msg)")
            default:
                XCTFail("expected helperNotFound, got \(err)")
            }
        }
    }

    // MARK: - Helpers

    private func pythonAvailable() -> Bool {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = ["python3", "-c", "print('ok')"]
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        do {
            try p.run()
            p.waitUntilExit()
            return p.terminationStatus == 0
        } catch {
            return false
        }
    }
}
