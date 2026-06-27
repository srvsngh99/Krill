import XCTest
@testable import KrillHarness

/// Tests for the PUBLIC web-search backends (keyless DuckDuckGo default + BYOK
/// Brave/Tavily) and the text helpers they share. All offline: parse methods are
/// pure, and backend selection is driven via env overrides.
final class PublicSearchBackendTests: XCTestCase {

    // MARK: - Text helpers

    func testStripHTMLAndEntities() {
        XCTAssertEqual(WebSearchText.stripHTML("A <strong>large</strong> model."), "A large model.")
        XCTAssertEqual(WebSearchText.stripHTML("Tom &amp; Jerry &#x27;quoted&#x27;"), "Tom & Jerry 'quoted'")
        XCTAssertEqual(WebSearchText.stripHTML("  multi   space\n\tline "), "multi space line")
    }

    // MARK: - DuckDuckGo (HTML scrape)

    func testDuckDuckGoParsesResultsAndDecodesRedirect() {
        let html = """
        <table>
        <tr><td><a rel="nofollow" href="https://example.com/llm" class="result-link">Large Language Models — Example</a></td></tr>
        <tr><td class="result-snippet">An LLM is a neural network trained on text.</td></tr>
        <tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FLLM&rut=abc" class="result-link">LLM - Wikipedia</a></td></tr>
        <tr><td class="result-snippet">A large language <b>model</b> article.</td></tr>
        </table>
        """
        let results = DuckDuckGoBackend.parse(html, limit: 5)
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].url, "https://example.com/llm")
        XCTAssertTrue(results[0].title.contains("Large Language Models"))
        XCTAssertTrue(results[0].snippet.contains("neural network"))
        // Second result's href is a /l/?uddg= redirect → decoded to the destination.
        XCTAssertEqual(results[1].url, "https://en.wikipedia.org/wiki/LLM")
        XCTAssertTrue(results[1].snippet.contains("model article"))
    }

    func testDuckDuckGoDecodeRedirect() {
        XCTAssertEqual(
            DuckDuckGoBackend.decodeRedirect("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.test%2Fx&rut=z"),
            "https://a.test/x")
        // A plain (non-redirect) href passes through.
        XCTAssertEqual(DuckDuckGoBackend.decodeRedirect("https://b.test/y"), "https://b.test/y")
    }

    func testDuckDuckGoLimit() {
        let html = (1...8).map {
            "<a href=\"https://e.test/\($0)\" class=\"result-link\">R\($0)</a>"
            + "<td class=\"result-snippet\">s\($0)</td>"
        }.joined()
        XCTAssertEqual(DuckDuckGoBackend.parse(html, limit: 3).count, 3)
    }

    // MARK: - Brave (JSON)

    func testBraveParsesAndStripsHighlightTags() throws {
        let json = """
        {"web":{"results":[
          {"title":"<strong>LLM</strong> guide","url":"https://example.com/a","description":"A <strong>large</strong> language model."},
          {"title":"Second","url":"https://example.com/b","description":"Snippet two."}
        ]}}
        """
        let results = try BraveBackend.parse(Data(json.utf8), limit: 5)
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].title, "LLM guide")
        XCTAssertEqual(results[0].url, "https://example.com/a")
        XCTAssertEqual(results[0].snippet, "A large language model.")
    }

    func testBraveBadBodyThrows() {
        XCTAssertThrowsError(try BraveBackend.parse(Data("not json".utf8), limit: 5))
    }

    // MARK: - Tavily (JSON)

    func testTavilyParses() throws {
        let json = """
        {"results":[
          {"title":"T1","url":"https://t.test/1","content":"content one"},
          {"title":"T2","url":"https://t.test/2","content":"content two"}
        ]}
        """
        let results = try TavilyBackend.parse(Data(json.utf8), limit: 5)
        XCTAssertEqual(results.map { $0.url }, ["https://t.test/1", "https://t.test/2"])
        XCTAssertEqual(results[1].snippet, "content two")
    }

    // MARK: - Backend selection (env-driven; env overrides the config file)

    func testDefaultBackendIsDuckDuckGo() {
        setenv("KRILL_SEARCH_BACKEND", "auto", 1)
        defer { unsetenv("KRILL_SEARCH_BACKEND") }
        XCTAssertEqual(WebSearchTool.configuredBackend()?.name, "duckduckgo",
                       "the zero-config default is keyless DuckDuckGo")
    }

    func testBraveRequiresKey() {
        setenv("KRILL_SEARCH_BACKEND", "brave", 1)
        unsetenv("KRILL_BRAVE_API_KEY")
        defer { unsetenv("KRILL_SEARCH_BACKEND") }
        XCTAssertNil(WebSearchTool.configuredBackend(), "brave without an api key → nil (actionable error)")
    }

    func testBraveWithKeySelected() {
        setenv("KRILL_SEARCH_BACKEND", "brave", 1)
        setenv("KRILL_BRAVE_API_KEY", "test-key", 1)
        defer { unsetenv("KRILL_SEARCH_BACKEND"); unsetenv("KRILL_BRAVE_API_KEY") }
        XCTAssertEqual(WebSearchTool.configuredBackend()?.name, "brave")
    }

    func testTavilyWithKeySelected() {
        setenv("KRILL_SEARCH_BACKEND", "tavily", 1)
        setenv("KRILL_TAVILY_API_KEY", "test-key", 1)
        defer { unsetenv("KRILL_SEARCH_BACKEND"); unsetenv("KRILL_TAVILY_API_KEY") }
        XCTAssertEqual(WebSearchTool.configuredBackend()?.name, "tavily")
    }
}
