import Foundation
import XCTest
@testable import KrillHarness

final class RepoMapToolTests: XCTestCase {

    func testSwiftSymbolExtraction() {
        let source = """
        import Foundation

        public struct WebSearchTool: Tool {
            public func run() {}   // nested: not top-level, must not appear
        }

        final class EventQueue {}
        func jsonString(_ obj: [String: Any]) -> String { "" }
        private enum Hidden {}
        """
        let symbols = RepoMap.extractSymbols(source: source, fileExtension: "swift")
        XCTAssertEqual(symbols, ["WebSearchTool", "EventQueue", "jsonString"])
    }

    func testPythonAndTypeScriptExtraction() {
        let py = "class Trainer:\n    def fit(self): pass\n\ndef main():\n    pass\n"
        XCTAssertEqual(RepoMap.extractSymbols(source: py, fileExtension: "py"), ["Trainer", "main"])

        let ts = "export function render() {}\nexport const config = 1\nclass App {}\n"
        XCTAssertEqual(
            RepoMap.extractSymbols(source: ts, fileExtension: "ts"), ["render", "config", "App"])
    }

    func testSymbolCapAndDedup() {
        let source = (1...10).map { "func f\($0)() {}" }.joined(separator: "\n")
            + "\nfunc f1() {}"
        let symbols = RepoMap.extractSymbols(source: source, fileExtension: "swift")
        XCTAssertEqual(symbols.count, 6, "capped at 6, duplicates dropped")
    }

    func testGenerateWalksTreeSkipsJunkAndListsSymbols() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("repomap-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let src = root.appendingPathComponent("Sources/App")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try "struct Server {}\nfunc start() {}\n".write(
            to: src.appendingPathComponent("Main.swift"), atomically: true, encoding: .utf8)
        try "# readme".write(
            to: root.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
        let junk = root.appendingPathComponent("node_modules/dep")
        try FileManager.default.createDirectory(at: junk, withIntermediateDirectories: true)
        try "class ShouldNotAppear {}".write(
            to: junk.appendingPathComponent("dep.js"), atomically: true, encoding: .utf8)

        let map = RepoMap.generate(root: root)
        XCTAssertTrue(map.contains("Sources/"))
        XCTAssertTrue(map.contains("Main.swift — Server, start"))
        XCTAssertTrue(map.contains("README.md"))
        XCTAssertFalse(map.contains("ShouldNotAppear"))
        XCTAssertTrue(map.contains("build/dependency dirs skipped"))
    }

    func testGenerateRespectsCharBudget() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("repomap-budget-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for i in 0..<80 {
            try "func generated\(i)() {}\n".write(
                to: root.appendingPathComponent("file-with-a-long-name-\(i).swift"),
                atomically: true, encoding: .utf8)
        }
        let map = RepoMap.generate(root: root, maxChars: 1_000)
        XCTAssertLessThan(map.count, 1_600, "body respects the budget (plus header slack)")
        XCTAssertTrue(map.contains("older files elided"))
    }

    func testToolErrorsOnMissingDirectory() async {
        let r = await RepoMapTool().run(argumentsJSON: #"{"path":"/nope/definitely-missing"}"#)
        XCTAssertTrue(r.isError)
    }
}
