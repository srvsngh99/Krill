import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Tavily backend (BYOK): an LLM-oriented search API tuned for agents
/// (`POST https://api.tavily.com/search`). Like Brave it has a free tier and a
/// clean JSON contract, so it is a solid reliable alternative to the keyless
/// `DuckDuckGoBackend` default. Selected with `search_backend = "tavily"` +
/// `tavily_api_key` (or `KRILL_TAVILY_API_KEY`).
public struct TavilyBackend: SearchBackend {
    public let name = "tavily"
    private let apiKey: String
    private let fetcher: WebFetcher
    private let timeout: TimeInterval

    public init(apiKey: String, fetcher: WebFetcher = URLSessionWebFetcher(), timeout: TimeInterval = 20) {
        self.apiKey = apiKey
        self.fetcher = fetcher
        self.timeout = timeout
    }

    public func search(query: String, count: Int) async throws -> [SearchResult] {
        guard let url = URL(string: "https://api.tavily.com/search") else { throw TavilyError.badResponse }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        // Tavily accepts the key in the body; also send it as a Bearer header,
        // which the API also honors.
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        let payload: [String: Any] = [
            "api_key": apiKey,
            "query": query,
            "max_results": max(1, min(20, count)),
            "search_depth": "basic",
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (data, response) = try await fetcher.fetch(request)
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw TavilyError.httpStatus(http.statusCode)
        }
        return try Self.parse(data, limit: count)
    }

    /// Parse a Tavily JSON body: `{ results: [{title, url, content}] }`.
    static func parse(_ data: Data, limit: Int) throws -> [SearchResult] {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = obj["results"] as? [[String: Any]]
        else { throw TavilyError.badResponse }
        var out: [SearchResult] = []
        for row in rows {
            guard let url = row["url"] as? String, !url.isEmpty else { continue }
            let title = WebSearchText.stripHTML((row["title"] as? String) ?? url)
            let snippet = WebSearchText.stripHTML((row["content"] as? String) ?? "")
            out.append(SearchResult(title: title.isEmpty ? url : title, url: url, snippet: snippet))
            if out.count >= limit { break }
        }
        return out
    }

    public enum TavilyError: Error, CustomStringConvertible, LocalizedError {
        case httpStatus(Int)
        case badResponse
        public var description: String {
            switch self {
            case .httpStatus(let c):
                return "Tavily returned HTTP \(c)"
                    + (c == 401 || c == 403 ? " (check tavily_api_key)" : "")
                    + (c == 429 ? " (rate limit / quota exceeded)" : "")
            case .badResponse: return "Tavily returned an unparseable response"
            }
        }
        public var errorDescription: String? { description }
    }
}
