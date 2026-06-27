import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Brave Search backend (BYOK): queries the Brave Search API
/// (`https://api.search.brave.com/res/v1/web/search`) with the user's own
/// subscription token. Brave offers a free tier, so this is the recommended
/// reliable upgrade over the keyless `DuckDuckGoBackend` default: real live-web
/// results, a documented JSON contract, and no scraping fragility. Selected with
/// `search_backend = "brave"` + `brave_api_key` (or `KRILL_BRAVE_API_KEY`).
public struct BraveBackend: SearchBackend {
    public let name = "brave"
    private let apiKey: String
    private let fetcher: WebFetcher
    private let timeout: TimeInterval

    public init(apiKey: String, fetcher: WebFetcher = URLSessionWebFetcher(), timeout: TimeInterval = 15) {
        self.apiKey = apiKey
        self.fetcher = fetcher
        self.timeout = timeout
    }

    public func search(query: String, count: Int) async throws -> [SearchResult] {
        guard var comps = URLComponents(string: "https://api.search.brave.com/res/v1/web/search") else {
            throw BraveError.badResponse
        }
        comps.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "count", value: String(max(1, min(20, count)))),
        ]
        guard let url = comps.url else { throw BraveError.badResponse }

        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("gzip", forHTTPHeaderField: "Accept-Encoding")
        request.setValue(apiKey, forHTTPHeaderField: "X-Subscription-Token")
        request.setValue("krill/web_search", forHTTPHeaderField: "User-Agent")

        let (data, response) = try await fetcher.fetch(request)
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw BraveError.httpStatus(http.statusCode)
        }
        return try Self.parse(data, limit: count)
    }

    /// Parse a Brave web-search JSON body: `{ web: { results: [{title, url,
    /// description}] } }`. `description` may contain `<strong>` highlight tags,
    /// which we strip.
    static func parse(_ data: Data, limit: Int) throws -> [SearchResult] {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let web = obj["web"] as? [String: Any],
              let rows = web["results"] as? [[String: Any]]
        else { throw BraveError.badResponse }
        var out: [SearchResult] = []
        for row in rows {
            guard let url = row["url"] as? String, !url.isEmpty else { continue }
            let title = WebSearchText.stripHTML((row["title"] as? String) ?? url)
            let snippet = WebSearchText.stripHTML((row["description"] as? String) ?? "")
            out.append(SearchResult(title: title.isEmpty ? url : title, url: url, snippet: snippet))
            if out.count >= limit { break }
        }
        return out
    }

    public enum BraveError: Error, CustomStringConvertible, LocalizedError {
        case httpStatus(Int)
        case badResponse
        public var description: String {
            switch self {
            case .httpStatus(let c):
                return "Brave Search returned HTTP \(c)"
                    + (c == 401 || c == 403 ? " (check brave_api_key)" : "")
                    + (c == 429 ? " (rate limit / quota exceeded)" : "")
            case .badResponse: return "Brave Search returned an unparseable response"
            }
        }
        public var errorDescription: String? { description }
    }
}
