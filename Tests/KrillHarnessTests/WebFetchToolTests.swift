import XCTest
@testable import KrillHarness
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Canned fetcher so the tool is exercised end to end without the network.
private struct StubFetcher: WebFetcher {
    let body: Data
    let status: Int
    let contentType: String
    let finalURL: URL?
    let error: Error?

    init(body: String = "", status: Int = 200, contentType: String = "text/html",
         finalURL: URL? = nil, error: Error? = nil) {
        self.body = Data(body.utf8)
        self.status = status
        self.contentType = contentType
        self.finalURL = finalURL
        self.error = error
    }

    func fetch(_ request: URLRequest) async throws -> (Data, URLResponse) {
        if let error { throw error }
        let url = finalURL ?? request.url!
        let resp = HTTPURLResponse(
            url: url, statusCode: status, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": contentType])!
        return (body, resp)
    }
}

final class WebFetchToolTests: XCTestCase {

    // MARK: - SSRF host policy (pure)

    func testBlocksPrivateAndLocalHosts() {
        for h in ["localhost", "127.0.0.1", "0.0.0.0", "10.1.2.3", "192.168.1.1",
                  "172.16.0.1", "172.31.255.255", "169.254.169.254", "100.64.0.1",
                  "box.local", "svc.internal", "metadata.google.internal",
                  "::1", "fe80::1", "fd00::1"] {
            XCTAssertTrue(WebFetchTool.isBlockedHost(h), "\(h) must be blocked")
        }
    }

    func testAllowsPublicHosts() {
        for h in ["example.com", "en.wikipedia.org", "8.8.8.8", "172.32.0.1", "11.0.0.1"] {
            XCTAssertFalse(WebFetchTool.isBlockedHost(h), "\(h) must be allowed")
        }
    }

    // MARK: - HTML extraction (pure)

    func testHtmlToTextStripsTagsScriptsAndEntities() {
        let html = """
        <html><head><title>t</title><style>.x{}</style></head>
        <body><script>evil()</script><h1>Title</h1><p>Hello&nbsp;&amp; welcome.</p>
        <ul><li>one</li><li>two</li></ul></body></html>
        """
        let text = WebFetchTool.htmlToText(html)
        XCTAssertTrue(text.contains("Title"))
        XCTAssertTrue(text.contains("Hello & welcome."), "entities decoded, nbsp normalized")
        XCTAssertTrue(text.contains("one"))
        XCTAssertTrue(text.contains("two"))
        XCTAssertFalse(text.contains("evil()"), "script content dropped")
        XCTAssertFalse(text.contains(".x{}"), "style content dropped")
        XCTAssertFalse(text.contains("<"), "no residual tags")
    }

    func testDecodesNumericEntities() {
        XCTAssertEqual(WebFetchTool.decodeEntities("A&#66;&#x43;"), "ABC")
    }

    // MARK: - run() paths (stubbed fetch)

    func testRejectsNonHttpScheme() async {
        let r = await WebFetchTool(fetcher: StubFetcher()).run(argumentsJSON: #"{"url":"file:///etc/passwd"}"#)
        XCTAssertTrue(r.isError)
        XCTAssertTrue(r.content.contains("valid http"))
    }

    func testRejectsPrivateURLBeforeFetching() async {
        // A stub that would fail the test if ever called - the guard must short
        // circuit before any fetch happens.
        struct Boom: WebFetcher {
            func fetch(_ request: URLRequest) async throws -> (Data, URLResponse) {
                XCTFail("must not fetch a blocked host"); return (Data(), URLResponse())
            }
        }
        let r = await WebFetchTool(fetcher: Boom()).run(argumentsJSON: #"{"url":"http://169.254.169.254/latest/meta-data"}"#)
        XCTAssertTrue(r.isError)
        XCTAssertTrue(r.content.contains("private/local"))
    }

    func testFetchesAndExtractsWithUntrustedFraming() async {
        let stub = StubFetcher(body: "<html><body><p>The capital is Paris.</p></body></html>")
        let r = await WebFetchTool(fetcher: stub).run(argumentsJSON: #"{"url":"https://example.com/x"}"#)
        XCTAssertFalse(r.isError)
        XCTAssertTrue(r.content.contains("UNTRUSTED"), "content is framed as untrusted data")
        XCTAssertTrue(r.content.contains("The capital is Paris."))
        XCTAssertFalse(r.content.contains("<p>"), "markup stripped")
    }

    func testNon2xxIsError() async {
        let stub = StubFetcher(body: "nope", status: 404, contentType: "text/plain")
        let r = await WebFetchTool(fetcher: stub).run(argumentsJSON: #"{"url":"https://example.com/missing"}"#)
        XCTAssertTrue(r.isError)
        XCTAssertTrue(r.content.contains("404"))
    }

    func testRedirectToPrivateFinalHostRejected() async {
        // Simulate URLSession having landed on a private final URL.
        let stub = StubFetcher(body: "secret", status: 200, contentType: "text/plain",
                               finalURL: URL(string: "http://127.0.0.1/admin"))
        let r = await WebFetchTool(fetcher: stub).run(argumentsJSON: #"{"url":"https://public.example/redir"}"#)
        XCTAssertTrue(r.isError)
        XCTAssertTrue(r.content.contains("private/local"))
        XCTAssertFalse(r.content.contains("secret"), "blocked body never surfaced")
    }

    func testPlainTextPassThroughAndTruncation() async {
        let long = String(repeating: "x", count: 5000)
        let stub = StubFetcher(body: long, status: 200, contentType: "text/plain")
        let r = await WebFetchTool(fetcher: stub).run(argumentsJSON: #"{"url":"https://example.com/t","max_chars":1000}"#)
        XCTAssertFalse(r.isError)
        XCTAssertTrue(r.content.contains("truncated to 1000"))
    }

    func testMissingURLIsError() async {
        let r = await WebFetchTool(fetcher: StubFetcher()).run(argumentsJSON: #"{}"#)
        XCTAssertTrue(r.isError)
    }

    func testIsReadOnly() {
        XCTAssertTrue(WebFetchTool().isReadOnly, "web_fetch is read-only so it runs without prompting")
    }
}
