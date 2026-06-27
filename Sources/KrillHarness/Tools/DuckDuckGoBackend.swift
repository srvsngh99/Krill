import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// DuckDuckGo backend: the KEYLESS, zero-config default so `web_search` works the
/// instant Krill is installed — no API key, no self-hosted instance. It scrapes
/// the lightweight HTML endpoint `https://lite.duckduckgo.com/lite/` (far stabler
/// to parse than the JS-heavy main SERP).
///
/// This is BEST-EFFORT by nature: HTML scraping of an unofficial endpoint is
/// subject to layout changes and rate limiting, and is a gray area w.r.t. ToS.
/// It exists so the tool is useful out of the box; users who want robust,
/// rate-limit-free results set `search_backend = "brave"` (or `"tavily"`) with a
/// free-tier API key. See `BraveBackend` / `TavilyBackend`.
public struct DuckDuckGoBackend: SearchBackend {
    public let name = "duckduckgo"
    private let fetcher: WebFetcher
    private let timeout: TimeInterval

    public init(fetcher: WebFetcher = URLSessionWebFetcher(), timeout: TimeInterval = 15) {
        self.fetcher = fetcher
        self.timeout = timeout
    }

    public func search(query: String, count: Int) async throws -> [SearchResult] {
        guard var comps = URLComponents(string: "https://lite.duckduckgo.com/lite/") else {
            throw DDGError.badResponse
        }
        comps.queryItems = [URLQueryItem(name: "q", value: query)]
        guard let url = comps.url else { throw DDGError.badResponse }

        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        // A browser-like UA: the lite endpoint serves a stripped page to clients
        // it recognizes and may refuse an obviously-bot UA.
        request.setValue(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            + "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            forHTTPHeaderField: "User-Agent")
        request.setValue("text/html", forHTTPHeaderField: "Accept")

        let (data, response) = try await fetcher.fetch(request)
        if let http = response as? HTTPURLResponse {
            // DDG answers 202 (a challenge page, no results) when it rate-limits a
            // client — which happens quickly under automated / multi-query use such
            // as deep research. Treat it (and 403/429) as a loud, actionable
            // rate-limit rather than a silent empty 2xx success.
            if http.statusCode == 202 || http.statusCode == 429 || http.statusCode == 403 {
                throw DDGError.rateLimited(http.statusCode)
            }
            if !(200...299).contains(http.statusCode) {
                throw DDGError.httpStatus(http.statusCode)
            }
        }
        guard let html = String(data: data, encoding: .utf8) else { throw DDGError.badResponse }
        return Self.parse(html, limit: count)
    }

    /// Parse a lite.duckduckgo.com results page into the top `limit` results.
    /// The lite layout pairs a `class="result-link"` anchor (title + href) with a
    /// following `class="result-snippet"` cell. Hrefs may be a `/l/?uddg=`
    /// redirect; we decode back to the destination URL. Tolerant by design:
    /// returns whatever rows it can recognize.
    static func parse(_ html: String, limit: Int) -> [SearchResult] {
        // <a ... class="result-link" ... href="URL">TITLE</a>  (attribute order varies;
        // class may use single or double quotes). Capture href then inner text.
        let anchorRows = WebSearchText.captures(
            #"<a[^>]*class=["']?result-link["']?[^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>"#,
            in: html)
        // Some renderings put href before class; try the reverse order too and
        // merge, de-duping by URL.
        let anchorRowsAlt = WebSearchText.captures(
            #"<a[^>]*href=["']([^"']+)["'][^>]*class=["']?result-link["']?[^>]*>(.*?)</a>"#,
            in: html)
        let snippets = WebSearchText.captures(
            #"class=["']?result-snippet["']?[^>]*>(.*?)</td>"#, in: html)
            .map { WebSearchText.stripHTML($0.count > 1 ? $0[1] : "") }

        var out: [SearchResult] = []
        var seen = Set<String>()
        var idx = 0
        for row in (anchorRows.isEmpty ? anchorRowsAlt : anchorRows) {
            guard row.count >= 3 else { continue }
            let url = decodeRedirect(row[1])
            guard !url.isEmpty, url.hasPrefix("http"), !seen.contains(url) else { idx += 1; continue }
            seen.insert(url)
            let title = WebSearchText.stripHTML(row[2])
            let snippet = idx < snippets.count ? snippets[idx] : ""
            out.append(SearchResult(title: title.isEmpty ? url : title, url: url, snippet: snippet))
            idx += 1
            if out.count >= limit { break }
        }
        return out
    }

    /// DuckDuckGo redirect links look like `//duckduckgo.com/l/?uddg=<pct-encoded>`
    /// (or `https://duckduckgo.com/l/?...`). Pull the real destination out of the
    /// `uddg` query param; pass non-redirect hrefs through unchanged.
    static func decodeRedirect(_ href: String) -> String {
        var h = href
        if h.hasPrefix("//") { h = "https:" + h }
        guard h.contains("/l/"), h.contains("uddg="),
              let comps = URLComponents(string: h),
              let uddg = comps.queryItems?.first(where: { $0.name == "uddg" })?.value
        else { return WebSearchText.decodeEntities(href.hasPrefix("//") ? "https:" + href : href) }
        return uddg
    }

    public enum DDGError: Error, CustomStringConvertible, LocalizedError {
        case httpStatus(Int)
        case rateLimited(Int)
        case badResponse
        public var description: String {
            switch self {
            case .httpStatus(let c): return "DuckDuckGo returned HTTP \(c)"
            case .rateLimited(let c):
                return "DuckDuckGo rate-limited this client (HTTP \(c)). The keyless "
                    + "default is best-effort and throttles under repeated/automated use "
                    + "(e.g. deep research). Set search_backend=brave or tavily with a free "
                    + "API key for reliable results."
            case .badResponse: return "DuckDuckGo returned an unparseable response"
            }
        }
        public var errorDescription: String? { description }
    }
}
