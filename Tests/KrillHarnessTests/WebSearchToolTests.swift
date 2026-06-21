import XCTest
@testable import KrillHarness
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Canned backend so the tool is exercised without the network.
private struct StubBackend: SearchBackend {
    let name = "stub"
    let results: [SearchResult]
    let error: Error?
    init(results: [SearchResult] = [], error: Error? = nil) {
        self.results = results
        self.error = error
    }
    func search(query: String, count: Int) async throws -> [SearchResult] {
        if let error { throw error }
        return Array(results.prefix(count))
    }
}

/// Canned fetcher returning a fixed JSON body, to drive SearxngBackend itself.
private struct StubJSONFetcher: WebFetcher {
    let body: Data
    let status: Int
    private(set) var lastURL: URLBox = URLBox()
    final class URLBox: @unchecked Sendable { var url: URL? }
    init(json: String, status: Int = 200) {
        self.body = Data(json.utf8)
        self.status = status
    }
    func fetch(_ request: URLRequest) async throws -> (Data, URLResponse) {
        lastURL.url = request.url
        let resp = HTTPURLResponse(
            url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"])!
        return (body, resp)
    }
}

final class WebSearchToolTests: XCTestCase {

    func testIsReadOnly() {
        XCTAssertTrue(WebSearchTool(backend: StubBackend()).isReadOnly,
                      "web_search is read-only so it runs without prompting")
    }

    func testMissingQueryIsError() async {
        let r = await WebSearchTool(backend: StubBackend()).run(argumentsJSON: #"{}"#)
        XCTAssertTrue(r.isError)
    }

    func testUnconfiguredBackendGivesActionableError() async {
        let r = await WebSearchTool(backend: nil).run(argumentsJSON: #"{"query":"swift"}"#)
        XCTAssertTrue(r.isError)
        XCTAssertTrue(r.content.contains("searxng_url"), "tells the user how to enable it")
    }

    func testFormatsResultsWithUntrustedFraming() async {
        let backend = StubBackend(results: [
            SearchResult(title: "Swift", url: "https://swift.org", snippet: "A general-purpose language."),
            SearchResult(title: "Docs", url: "https://docs.swift.org", snippet: ""),
        ])
        let r = await WebSearchTool(backend: backend).run(argumentsJSON: #"{"query":"swift lang"}"#)
        XCTAssertFalse(r.isError)
        XCTAssertTrue(r.content.contains("UNTRUSTED"), "snippets framed as untrusted data")
        XCTAssertTrue(r.content.contains("https://swift.org"))
        XCTAssertTrue(r.content.contains("A general-purpose language."))
        XCTAssertTrue(r.content.contains("https://docs.swift.org"))
    }

    func testCountIsClampedAndHonored() async {
        let many = (1...20).map {
            SearchResult(title: "t\($0)", url: "https://e/\($0)", snippet: "")
        }
        // count=3 honored
        let r3 = await WebSearchTool(backend: StubBackend(results: many))
            .run(argumentsJSON: #"{"query":"x","count":3}"#)
        XCTAssertTrue(r3.content.contains("https://e/3"))
        XCTAssertFalse(r3.content.contains("https://e/4"))
        // count=99 clamped to 10
        let rBig = await WebSearchTool(backend: StubBackend(results: many))
            .run(argumentsJSON: #"{"query":"x","count":99}"#)
        XCTAssertTrue(rBig.content.contains("https://e/10"))
        XCTAssertFalse(rBig.content.contains("https://e/11"))
    }

    func testEmptyResultsIsNotAnError() async {
        let r = await WebSearchTool(backend: StubBackend(results: []))
            .run(argumentsJSON: #"{"query":"nothing here"}"#)
        XCTAssertFalse(r.isError)
        XCTAssertTrue(r.content.contains("No results"))
    }

    func testBackendErrorSurfaces() async {
        let backend = StubBackend(error: SearxngBackend.SearchError.httpStatus(403))
        let r = await WebSearchTool(backend: backend).run(argumentsJSON: #"{"query":"x"}"#)
        XCTAssertTrue(r.isError)
        XCTAssertTrue(r.content.contains("403"))
    }

    // MARK: - SearXNG backend (parser + URL building)

    func testSearxngParsesResults() throws {
        let json = """
        {"query":"q","results":[
          {"url":"https://a.example","title":"A","content":"first"},
          {"url":"https://b.example","title":"B","content":"second"},
          {"title":"no url"}
        ]}
        """
        let results = try SearxngBackend.parse(Data(json.utf8), limit: 5)
        XCTAssertEqual(results.count, 2, "rows without a url are skipped")
        XCTAssertEqual(results[0].url, "https://a.example")
        XCTAssertEqual(results[0].title, "A")
        XCTAssertEqual(results[0].snippet, "first")
    }

    func testSearxngParseLimit() throws {
        let rows = (1...5).map { #"{"url":"https://e/\#($0)","title":"t","content":"c"}"# }
            .joined(separator: ",")
        let results = try SearxngBackend.parse(Data("{\"results\":[\(rows)]}".utf8), limit: 2)
        XCTAssertEqual(results.count, 2)
    }

    func testSearxngBuildsJSONQueryURL() async throws {
        let fetcher = StubJSONFetcher(json: #"{"results":[]}"#)
        let backend = SearxngBackend(baseURL: "http://localhost:8888/", fetcher: fetcher)
        _ = try await backend.search(query: "hello world", count: 5)
        let url = fetcher.lastURL.url?.absoluteString ?? ""
        XCTAssertTrue(url.hasPrefix("http://localhost:8888/search?"), "trailing slash normalized: \(url)")
        XCTAssertTrue(url.contains("format=json"))
        XCTAssertTrue(url.contains("q=hello%20world") || url.contains("q=hello+world"), url)
    }

    func testSearxngNon2xxThrows() async {
        let fetcher = StubJSONFetcher(json: "forbidden", status: 403)
        let backend = SearxngBackend(baseURL: "http://localhost:8888", fetcher: fetcher)
        do {
            _ = try await backend.search(query: "x", count: 5)
            XCTFail("expected throw on 403")
        } catch {
            XCTAssertTrue("\(error)".contains("403"))
        }
    }
}
