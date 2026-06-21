import XCTest
@testable import KrillHarness

/// Canned backend returning fixed hits per query.
private struct StubSearch: SearchBackend {
    let name = "stub"
    let byQuery: [String: [SearchResult]]
    let flat: [SearchResult]
    init(byQuery: [String: [SearchResult]] = [:], flat: [SearchResult] = []) {
        self.byQuery = byQuery
        self.flat = flat
    }
    func search(query: String, count: Int) async throws -> [SearchResult] {
        let hits = byQuery[query] ?? flat
        return Array(hits.prefix(count))
    }
}

final class DeepResearchTests: XCTestCase {

    // MARK: - parseQueries (pure)

    func testParseQueriesStripsMarkersAndDedupes() {
        let text = """
        Here are some queries:
        1. swift concurrency model
        2) swift concurrency model
        - structured concurrency Swift
        * "async await swift"
        """
        let q = DeepResearch.parseQueries(text, max: 5)
        XCTAssertEqual(q, ["swift concurrency model", "structured concurrency Swift", "async await swift"],
                       "markers stripped, case-insensitive dedupe, preamble line dropped")
    }

    func testParseQueriesHonorsMax() {
        let text = "a\nb\nc\nd\ne"
        XCTAssertEqual(DeepResearch.parseQueries(text, max: 2), ["a", "b"])
    }

    func testParseQueriesEmpty() {
        XCTAssertTrue(DeepResearch.parseQueries("\n  \n", max: 4).isEmpty)
    }

    // MARK: - dedupeSources (pure)

    func testDedupeByCanonicalURL() {
        let results = [
            SearchResult(title: "A", url: "https://example.com/page", snippet: ""),
            SearchResult(title: "A dup", url: "https://www.example.com/page/", snippet: ""),  // same after canon
            SearchResult(title: "B", url: "http://other.org/x", snippet: ""),
        ]
        let out = DeepResearch.dedupeSources(results, limit: 10)
        XCTAssertEqual(out.count, 2, "www + trailing slash collapse to the first")
        XCTAssertEqual(out[0].title, "A")
        XCTAssertEqual(out[1].title, "B")
    }

    func testDedupeHonorsLimit() {
        let results = (1...10).map { SearchResult(title: "t\($0)", url: "https://e/\($0)", snippet: "") }
        XCTAssertEqual(DeepResearch.dedupeSources(results, limit: 3).count, 3)
    }

    // MARK: - prompt builders (pure)

    func testSynthesisMessagesNumbersSources() {
        let findings = [
            DeepResearch.Finding(url: "https://a", title: "A", summary: "alpha"),
            DeepResearch.Finding(url: "https://b", title: "B", summary: "beta"),
        ]
        let msgs = DeepResearch.synthesisMessages(question: "Q?", findings: findings)
        let user = msgs.last!["content"]!
        XCTAssertTrue(user.contains("[1] A - https://a"))
        XCTAssertTrue(user.contains("[2] B - https://b"))
        XCTAssertTrue(user.contains("alpha"))
    }

    func testSummaryMessagesFrameUntrusted() {
        let msgs = DeepResearch.summaryMessages(question: "Q", title: "T", url: "u", page: "body")
        XCTAssertTrue(msgs.first!["content"]!.contains("UNTRUSTED"))
        XCTAssertTrue(msgs.last!["content"]!.contains("body"))
    }

    func testSourcesList() {
        let f = [DeepResearch.Finding(url: "https://a", title: "A", summary: "x")]
        XCTAssertEqual(DeepResearch.sourcesList(f), "Sources:\n[1] https://a\n")
    }

    // MARK: - full run() with stubs

    func testRunHappyPath() async {
        let backend = StubSearch(flat: [
            SearchResult(title: "Doc1", url: "https://a.example/1", snippet: ""),
            SearchResult(title: "Doc2", url: "https://b.example/2", snippet: ""),
        ])
        // complete() routes by the system prompt so each phase returns sensible text.
        let complete: DeepResearch.Complete = { msgs in
            let sys = msgs.first?["content"] ?? ""
            if sys.contains("plan web research") { return "query one\nquery two" }
            if sys.contains("extract facts") { return "Relevant fact for the question." }
            if sys.contains("research answer") { return "Final answer [1][2].\nSources:\n[1]...\n[2]..." }
            return ""
        }
        let fetch: DeepResearch.Fetch = { url, _ in "page text for \(url)" }
        let dr = DeepResearch(complete: complete, backend: backend, fetch: fetch)

        final class ProgressBox: @unchecked Sendable { var items: [DeepResearch.Progress] = [] }
        let pb = ProgressBox()
        let report = await dr.run(question: "What is X?") { pb.items.append($0) }
        let progress = pb.items

        XCTAssertFalse(report.isEmpty)
        XCTAssertEqual(report.sources.count, 2)
        XCTAssertTrue(report.text.contains("Final answer"))
        XCTAssertTrue(progress.contains(.planning))
        XCTAssertTrue(progress.contains(.synthesizing))
        XCTAssertTrue(progress.contains(.queries(["query one", "query two"])))
    }

    func testRunNoSearchResultsYieldsEmpty() async {
        let dr = DeepResearch(
            complete: { _ in "q1\nq2" },
            backend: StubSearch(flat: []),
            fetch: { _, _ in "x" })
        let report = await dr.run(question: "Q") { _ in }
        XCTAssertTrue(report.isEmpty, "no sources -> empty report, not a hollow synthesis")
        XCTAssertEqual(report.text, "")
    }

    func testRunSkipsUnfetchablePages() async {
        let backend = StubSearch(flat: [
            SearchResult(title: "ok", url: "https://ok/1", snippet: ""),
            SearchResult(title: "dead", url: "https://dead/2", snippet: ""),
        ])
        let complete: DeepResearch.Complete = { msgs in
            let sys = msgs.first?["content"] ?? ""
            if sys.contains("plan web research") { return "q" }
            if sys.contains("extract facts") { return "summary" }
            return "report"
        }
        // dead URL returns nil (fetch failed) -> skipped.
        let fetch: DeepResearch.Fetch = { url, _ in url.contains("dead") ? nil : "page" }
        let dr = DeepResearch(complete: complete, backend: backend, fetch: fetch)
        let report = await dr.run(question: "Q") { _ in }
        XCTAssertEqual(report.sources.count, 1)
        XCTAssertEqual(report.sources.first?.url, "https://ok/1")
    }

    func testRunFallsBackToRawQuestionWhenPlannerEmpty() async {
        var searched: [String] = []
        final class Box: @unchecked Sendable { var qs: [String] = [] }
        let box = Box()
        struct RecordingSearch: SearchBackend {
            let name = "rec"
            let box: Box
            func search(query: String, count: Int) async throws -> [SearchResult] {
                box.qs.append(query)
                return [SearchResult(title: "t", url: "https://e/1", snippet: "")]
            }
        }
        let complete: DeepResearch.Complete = { msgs in
            let sys = msgs.first?["content"] ?? ""
            if sys.contains("plan web research") { return "   \n  " }  // empty plan
            if sys.contains("extract facts") { return "s" }
            return "r"
        }
        let dr = DeepResearch(complete: complete, backend: RecordingSearch(box: box),
                              fetch: { _, _ in "p" })
        _ = await dr.run(question: "raw question") { _ in }
        searched = box.qs
        XCTAssertEqual(searched, ["raw question"], "empty planner -> search the raw question")
    }
}
