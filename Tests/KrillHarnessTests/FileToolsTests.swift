import XCTest
@testable import KrillHarness

final class FileToolsTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-filetools-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func path(_ name: String) -> String { dir.appendingPathComponent(name).path }
    private func write(_ name: String, _ content: String) throws {
        let url = dir.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(content.utf8).write(to: url)
    }
    private func read(_ name: String) -> String {
        (try? String(contentsOf: dir.appendingPathComponent(name), encoding: .utf8)) ?? ""
    }
    private func args(_ d: [String: Any]) -> String {
        String(decoding: try! JSONSerialization.data(withJSONObject: d), as: UTF8.self)
    }

    // MARK: read_file

    func testReadReturnsLineNumberedContent() async throws {
        try write("a.txt", "one\ntwo\nthree")
        let r = await ReadTool().run(argumentsJSON: args(["path": path("a.txt")]))
        XCTAssertFalse(r.isError)
        XCTAssertTrue(r.content.contains("1\tone"))
        XCTAssertTrue(r.content.contains("3\tthree"))
    }

    func testReadOffsetAndLimit() async throws {
        try write("a.txt", "l1\nl2\nl3\nl4\nl5")
        let r = await ReadTool().run(argumentsJSON: args(["path": path("a.txt"), "offset": 2, "limit": 2]))
        XCTAssertTrue(r.content.contains("2\tl2"))
        XCTAssertTrue(r.content.contains("3\tl3"))
        XCTAssertFalse(r.content.contains("\tl1"))
        XCTAssertFalse(r.content.contains("\tl4"))
    }

    func testReadMissingFileIsError() async {
        let r = await ReadTool().run(argumentsJSON: args(["path": path("nope.txt")]))
        XCTAssertTrue(r.isError)
    }

    func testReadDirectoryIsError() async {
        let r = await ReadTool().run(argumentsJSON: args(["path": dir.path]))
        XCTAssertTrue(r.isError)
        XCTAssertTrue(r.content.contains("directory"))
    }

    // MARK: write_file

    func testWriteCreatesThenOverwrites() async throws {
        let r1 = await WriteTool().run(argumentsJSON: args(["path": path("nested/new.txt"), "content": "hi"]))
        XCTAssertFalse(r1.isError)
        XCTAssertTrue(r1.content.contains("Created"))
        XCTAssertEqual(read("nested/new.txt"), "hi")
        let r2 = await WriteTool().run(argumentsJSON: args(["path": path("nested/new.txt"), "content": "bye"]))
        XCTAssertTrue(r2.content.contains("Overwrote"))
        XCTAssertEqual(read("nested/new.txt"), "bye")
    }

    // MARK: edit_file (pure logic)

    func testEditApplyUniqueReplace() {
        XCTAssertEqual(EditTool.apply(to: "let x = 1", old: "1", new: "2", replaceAll: false),
                       .ok(text: "let x = 2", count: 1))
    }
    func testEditApplyAmbiguousWithoutReplaceAll() {
        if case .ok = EditTool.apply(to: "a a a", old: "a", new: "b", replaceAll: false) {
            XCTFail("ambiguous edit must fail")
        }
    }
    func testEditApplyReplaceAll() {
        XCTAssertEqual(EditTool.apply(to: "a a a", old: "a", new: "b", replaceAll: true),
                       .ok(text: "b b b", count: 3))
    }
    func testEditApplyNotFound() {
        if case .ok = EditTool.apply(to: "abc", old: "z", new: "y", replaceAll: false) {
            XCTFail("not-found must fail")
        }
    }
    func testEditApplyIdenticalRejected() {
        if case .ok = EditTool.apply(to: "abc", old: "a", new: "a", replaceAll: false) {
            XCTFail("identical old/new must fail")
        }
    }

    func testEditFileEndToEnd() async throws {
        try write("c.swift", "func foo() {}")
        let r = await EditTool().run(argumentsJSON: args([
            "path": path("c.swift"), "old_string": "foo", "new_string": "bar",
        ]))
        XCTAssertFalse(r.isError)
        XCTAssertEqual(read("c.swift"), "func bar() {}")
    }

    // MARK: multi_edit (atomic)

    func testMultiEditAppliesAllInOrder() async throws {
        try write("m.txt", "alpha beta")
        let r = await MultiEditTool().run(argumentsJSON: args([
            "path": path("m.txt"),
            "edits": [["old_string": "alpha", "new_string": "A"], ["old_string": "beta", "new_string": "B"]],
        ]))
        XCTAssertFalse(r.isError)
        XCTAssertEqual(read("m.txt"), "A B")
    }

    func testMultiEditIsAtomicOnFailure() async throws {
        try write("m.txt", "alpha beta")
        let r = await MultiEditTool().run(argumentsJSON: args([
            "path": path("m.txt"),
            "edits": [["old_string": "alpha", "new_string": "A"], ["old_string": "ZZZ", "new_string": "B"]],
        ]))
        XCTAssertTrue(r.isError)
        XCTAssertEqual(read("m.txt"), "alpha beta", "a failing edit must leave the file untouched")
    }

    // MARK: list_dir

    func testListShowsEntriesWithDirMarker() async throws {
        try write("file.txt", "x")
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("subdir"), withIntermediateDirectories: true)
        let r = await ListTool().run(argumentsJSON: args(["path": dir.path]))
        XCTAssertTrue(r.content.contains("file.txt"))
        XCTAssertTrue(r.content.contains("subdir/"))
    }

    // MARK: glob

    func testGlobRecursiveAndNonRecursive() async throws {
        try write("a.swift", "")
        try write("sub/b.swift", "")
        try write("c.txt", "")
        let rec = await GlobTool().run(argumentsJSON: args(["pattern": "**/*.swift", "path": dir.path]))
        XCTAssertTrue(rec.content.contains("a.swift"))
        XCTAssertTrue(rec.content.contains("sub/b.swift"))
        XCTAssertFalse(rec.content.contains("c.txt"))

        let flat = await GlobTool().run(argumentsJSON: args(["pattern": "*.swift", "path": dir.path]))
        XCTAssertTrue(flat.content.contains("a.swift"))
        XCTAssertFalse(flat.content.contains("sub/b.swift"), "* must not cross directories")
    }

    func testGlobToRegexBasics() {
        XCTAssertEqual(FileToolSupport.globToRegex("*.swift"), "^[^/]*\\.swift$")
        XCTAssertEqual(FileToolSupport.globToRegex("a?b"), "^a[^/]b$")
    }

    // MARK: grep

    func testGrepReturnsFileLineMatches() async throws {
        try write("g.txt", "hello\nworld\nhello again")
        let r = await GrepTool().run(argumentsJSON: args(["pattern": "hello", "path": dir.path]))
        XCTAssertFalse(r.isError)
        XCTAssertTrue(r.content.contains(":1: hello"))
        XCTAssertTrue(r.content.contains(":3: hello again"))
        XCTAssertFalse(r.content.contains(":2:"))
    }

    func testGrepGlobFilter() async throws {
        try write("keep.swift", "needle")
        try write("skip.txt", "needle")
        let r = await GrepTool().run(argumentsJSON: args(["pattern": "needle", "path": dir.path, "glob": "**/*.swift"]))
        XCTAssertTrue(r.content.contains("keep.swift"))
        XCTAssertFalse(r.content.contains("skip.txt"))
    }

    func testGrepInvalidRegexIsError() async {
        let r = await GrepTool().run(argumentsJSON: args(["pattern": "[unclosed", "path": dir.path]))
        XCTAssertTrue(r.isError)
    }
}
