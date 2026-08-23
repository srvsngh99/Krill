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
        guard case .some(.diff(let createdPath, let createdHunks)) = r1.display else {
            return XCTFail("write_file must attach a diff display")
        }
        XCTAssertEqual(createdPath, path("nested/new.txt"))
        XCTAssertEqual(FileToolSupport.diffstat(hunks: createdHunks), "+1 -0")

        let r2 = await WriteTool().run(argumentsJSON: args(["path": path("nested/new.txt"), "content": "bye"]))
        XCTAssertTrue(r2.content.contains("Overwrote"))
        XCTAssertEqual(read("nested/new.txt"), "bye")
        guard case .some(.diff(_, let overwrittenHunks)) = r2.display else {
            return XCTFail("overwrite must attach a diff display")
        }
        XCTAssertEqual(FileToolSupport.diffstat(hunks: overwrittenHunks), "+1 -1")
    }

    func testUnifiedDiffBuildsNumberedContextHunks() {
        let old = (1 ... 12).map { "line \($0)" }.joined(separator: "\n")
        var newLines = (1 ... 12).map { "line \($0)" }
        newLines[5] = "changed six"
        let hunks = FileToolSupport.unifiedDiff(
            old: old, new: newLines.joined(separator: "\n"), context: 2)

        XCTAssertEqual(hunks.count, 1)
        let hunk = hunks[0]
        XCTAssertEqual(hunk.oldStart, 4)
        XCTAssertEqual(hunk.oldCount, 5)
        XCTAssertEqual(hunk.newStart, 4)
        XCTAssertEqual(hunk.newCount, 5)
        XCTAssertEqual(hunk.lines.first?.oldLine, 4)
        XCTAssertEqual(hunk.lines.first?.newLine, 4)
        XCTAssertEqual(hunk.lines.map(\.kind), [.context, .context, .deletion, .addition, .context, .context])
        XCTAssertEqual(hunk.lines.first(where: { $0.kind == .deletion })?.text, "line 6")
        XCTAssertEqual(hunk.lines.first(where: { $0.kind == .addition })?.text, "changed six")
    }

    func testUnifiedDiffSeparatesDistantChangesAndBoundsModelPreview() {
        let old = (1 ... 30).map { "line \($0)" }.joined(separator: "\n")
        var newLines = (1 ... 30).map { "line \($0)" }
        newLines[1] = "changed two"
        newLines[27] = "changed twenty-eight"
        let hunks = FileToolSupport.unifiedDiff(
            old: old, new: newLines.joined(separator: "\n"), context: 1)

        XCTAssertEqual(hunks.count, 2)
        let preview = FileToolSupport.compactPreview(
            hunks: hunks, maxLines: 3, maxCharacters: 120)
        XCTAssertTrue(preview.hasPrefix("@@ "))
        XCTAssertTrue(preview.hasSuffix("..."), preview)
        XCTAssertLessThanOrEqual(preview.count, 123)
        XCTAssertFalse(preview.contains("changed twenty-eight"), "only the first hunk belongs in model content")
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
        guard case .some(.diff(let displayPath, let hunks)) = r.display else {
            return XCTFail("edit_file must attach a diff display")
        }
        XCTAssertEqual(displayPath, path("c.swift"))
        XCTAssertEqual(FileToolSupport.diffstat(hunks: hunks), "+1 -1")
        XCTAssertTrue(r.content.contains("@@ -1,1 +1,1 @@"))
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
        guard case .some(.diff(let displayPath, let hunks)) = r.display else {
            return XCTFail("multi_edit must attach a diff display")
        }
        XCTAssertEqual(displayPath, path("m.txt"))
        XCTAssertEqual(FileToolSupport.diffstat(hunks: hunks), "+1 -1")
    }

    func testWriteKeepsFullDiffOutOfModelContent() async {
        let content = (1 ... 100).map { "new line \($0)" }.joined(separator: "\n")
        let r = await WriteTool().run(argumentsJSON: args([
            "path": path("large.txt"), "content": content,
        ]))

        XCTAssertLessThan(r.content.count, 1_000, "model-facing content must remain bounded")
        guard case .some(.diff(_, let hunks)) = r.display else {
            return XCTFail("write_file must carry the full diff in display")
        }
        XCTAssertEqual(hunks.flatMap(\.lines).filter { $0.kind == .addition }.count, 100)
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

    func testGlobDoubleStarKeepsPathBoundary() {
        let rx = try! NSRegularExpression(pattern: FileToolSupport.globToRegex("**/foo.txt"))
        func matches(_ s: String) -> Bool {
            rx.firstMatch(in: s, range: NSRange(s.startIndex..<s.endIndex, in: s)) != nil
        }
        XCTAssertTrue(matches("foo.txt"))         // zero dirs
        XCTAssertTrue(matches("a/b/foo.txt"))     // nested
        XCTAssertFalse(matches("barfoo.txt"), "**/ must not match across a partial component")
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

    // MARK: result counts
    //
    // These tools exist to answer questions, and "how many" is the most common
    // one. Returning a bare list makes the model tally lines instead — a real
    // agent run over 67 files answered 64, and answered 67 once the count was
    // stated. The count is the contract, not a cosmetic prefix.

    func testGlobLeadsWithTheMatchCount() async throws {
        try write("a.swift", "")
        try write("sub/b.swift", "")
        try write("c.txt", "")
        let r = await GlobTool().run(argumentsJSON: args(["pattern": "**/*.swift", "path": dir.path]))
        XCTAssertTrue(r.content.hasPrefix("2 matches"), "count must lead: \(r.content)")
        XCTAssertFalse(r.content.contains("3 matches"), "c.txt must not be counted")
    }

    func testGlobSingleMatchIsNotPluralized() async throws {
        try write("only.swift", "")
        let r = await GlobTool().run(argumentsJSON: args(["pattern": "*.swift", "path": dir.path]))
        XCTAssertTrue(r.content.hasPrefix("1 match for"), r.content)
        XCTAssertFalse(r.content.contains("1 matches"))
    }

    /// The count must survive truncation as a lower bound rather than silently
    /// reporting the cap as if it were the total.
    func testGlobTruncationReportsCountAsALowerBound() async throws {
        for i in 0 ..< 5 { try write("f\(i).swift", "") }
        let r = await GlobTool(maxResults: 3).run(
            argumentsJSON: args(["pattern": "*.swift", "path": dir.path]))
        XCTAssertTrue(r.content.hasPrefix("3+ matches"), "must read as a floor: \(r.content)")
        XCTAssertTrue(r.content.contains("truncated"))
    }

    func testGrepReportsMatchAndFileCounts() async throws {
        try write("one.txt", "needle\nchaff\nneedle")
        try write("two.txt", "needle")
        let r = await GrepTool().run(argumentsJSON: args(["pattern": "needle", "path": dir.path]))
        // 3 occurrences across 2 files — two distinct questions, both answered.
        XCTAssertTrue(r.content.hasPrefix("3 matches in 2 files"), r.content)
    }

    func testGrepEmptyResultStillStatesTheOutcome() async throws {
        try write("x.txt", "nothing here")
        let r = await GrepTool().run(argumentsJSON: args(["pattern": "needle", "path": dir.path]))
        XCTAssertFalse(r.isError)
        XCTAssertTrue(r.content.contains("No matches"), r.content)
    }
}
