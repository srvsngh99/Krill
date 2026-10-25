import XCTest
@testable import KrillHarness
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

// SelfHostedIndexBackend is gated behind the SELFHOSTED_INDEX compile flag (private, dev-only), so
// its tests are too — they compile only under `KRILL_SELFHOSTED_INDEX=1` (see Package.swift).
#if SELFHOSTED_INDEX

/// Canned fetcher returning a fixed JSON body, to drive SelfHostedIndexBackend without
/// the network (mirrors the SearxngBackend test's stub).
private struct StubJSONFetcher: WebFetcher {
    let body: Data
    let status: Int
    let box = URLBox()
    final class URLBox: @unchecked Sendable { var url: URL? }
    init(json: String, status: Int = 200) {
        self.body = Data(json.utf8)
        self.status = status
    }
    func fetch(_ request: URLRequest) async throws -> (Data, URLResponse) {
        box.url = request.url
        let resp = HTTPURLResponse(
            url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"])!
        return (body, resp)
    }
}

final class SelfHostedIndexBackendTests: XCTestCase {

    func testParsesSearchResults() async throws {
        let json = """
        {"query":"diabetes","results":[
          {"title":"Diabetes","url":"https://medlineplus.gov/diabetes.html","snippet":"Diabetes is a disease.","domain":"medlineplus.gov","score":0.9},
          {"title":"CDC Diabetes","url":"https://www.cdc.gov/diabetes/","snippet":"","domain":"www.cdc.gov","score":0.8}
        ],"total":2,"took_ms":12.0}
        """
        let fetcher = StubJSONFetcher(json: json)
        let backend = SelfHostedIndexBackend(baseURL: "http://127.0.0.1:8000/", fetcher: fetcher)
        let results = try await backend.search(query: "diabetes", count: 5)

        XCTAssertEqual(backend.name, "selfhosted")
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].url, "https://medlineplus.gov/diabetes.html")
        XCTAssertEqual(results[0].title, "Diabetes")
        XCTAssertEqual(results[0].snippet, "Diabetes is a disease.")

        // Builds /search?q=&limit=, trailing slash on the base normalized away.
        let u = fetcher.box.url!.absoluteString
        XCTAssertTrue(u.contains("/search?"), "calls the index /search")
        XCTAssertTrue(u.contains("q=diabetes"))
        XCTAssertTrue(u.contains("limit=5"))
        XCTAssertFalse(u.contains("//search"), "trailing slash normalized")
    }

    func testRespectsCountLimit() async throws {
        let json = """
        {"results":[{"title":"a","url":"https://a.com","snippet":"x"},
        {"title":"b","url":"https://b.com","snippet":"y"},
        {"title":"c","url":"https://c.com","snippet":"z"}]}
        """
        let backend = SelfHostedIndexBackend(baseURL: "http://127.0.0.1:8000", fetcher: StubJSONFetcher(json: json))
        let results = try await backend.search(query: "q", count: 2)
        XCTAssertEqual(results.count, 2)
    }

    func testHTTPErrorThrows() async {
        let backend = SelfHostedIndexBackend(
            baseURL: "http://127.0.0.1:8000", fetcher: StubJSONFetcher(json: "{}", status: 500))
        do {
            _ = try await backend.search(query: "q", count: 5)
            XCTFail("a non-2xx status should throw")
        } catch {
            // expected
        }
    }

    func testConfiguredBackendSelectsSelfHosted() {
        setenv("KRILL_SEARCH_BACKEND", "selfhosted", 1)
        setenv("KRILL_SELFHOSTED_URL", "http://127.0.0.1:8000", 1)
        defer {
            unsetenv("KRILL_SEARCH_BACKEND")
            unsetenv("KRILL_SELFHOSTED_URL")
        }
        let backend = WebSearchTool.configuredBackend()
        XCTAssertEqual(backend?.name, "selfhosted", "search_backend=selfhosted selects SelfHostedIndexBackend")
    }
}

#endif // SELFHOSTED_INDEX
