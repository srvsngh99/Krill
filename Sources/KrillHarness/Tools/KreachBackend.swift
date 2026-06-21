import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Kreach backend: queries the user's own Kreach search engine
/// (`<base>/search?q=...&limit=N`) and maps its ranked results into
/// `SearchResult`s. This is the conformer the `SearchBackend` protocol was
/// designed to accept — selecting it (`search_backend = "kreach"`) is the only
/// change; the `web_search` tool and the `DeepResearch` orchestrator are
/// backend-agnostic.
///
/// The role this plays: Kreach is the *search platform* underneath KrillLM's
/// `DeepResearch` — Kreach retrieves over its owned, crawled index; the agent
/// (this process, driven by the local model) plans the queries, reads the
/// pages, and synthesizes the cited answer. Every client that talks to KrillLM
/// (deepkrill, minikrill, …) gets deep research over Kreach for free via the
/// server's `/research` route, without reimplementing the loop.
///
/// Kreach is a trusted service the user runs (typically loopback on
/// `127.0.0.1:8000`), so — exactly like `SearxngBackend` — the first hop is
/// allowed to reach a private/loopback host (the shared `URLSessionWebFetcher`
/// only vets *redirects*, not the first request, and Kreach answers 200
/// directly without redirecting).
public struct KreachBackend: SearchBackend {
    public let name = "kreach"
    private let baseURL: String
    private let fetcher: WebFetcher
    private let timeout: TimeInterval

    public init(baseURL: String, fetcher: WebFetcher = URLSessionWebFetcher(), timeout: TimeInterval = 15) {
        self.baseURL = baseURL
        self.fetcher = fetcher
        self.timeout = timeout
    }

    public func search(query: String, count: Int) async throws -> [SearchResult] {
        // Normalize: strip a trailing slash so we don't emit `//search`.
        var base = baseURL.trimmingCharacters(in: .whitespaces)
        while base.hasSuffix("/") { base = String(base.dropLast()) }
        guard var comps = URLComponents(string: base + "/search") else {
            throw KreachError.badBackendURL(baseURL)
        }
        comps.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "limit", value: String(max(1, count))),
        ]
        guard let url = comps.url else { throw KreachError.badBackendURL(baseURL) }

        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        request.setValue("krill/web_search", forHTTPHeaderField: "User-Agent")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await fetcher.fetch(request)
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw KreachError.httpStatus(http.statusCode)
        }
        return try Self.parse(data, limit: count)
    }

    /// Parse a Kreach `/search` JSON body (`{results:[{title,url,snippet,...}]}`)
    /// into the top `limit` results, in rank order.
    static func parse(_ data: Data, limit: Int) throws -> [SearchResult] {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = obj["results"] as? [[String: Any]]
        else { throw KreachError.badResponse }
        var out: [SearchResult] = []
        for row in rows {
            guard let url = (row["url"] as? String), !url.isEmpty else { continue }
            let title = (row["title"] as? String) ?? url
            let snippet = (row["snippet"] as? String) ?? ""
            out.append(SearchResult(title: title, url: url, snippet: snippet))
            if out.count >= limit { break }
        }
        return out
    }

    public enum KreachError: Error, CustomStringConvertible, LocalizedError {
        case badBackendURL(String)
        case httpStatus(Int)
        case badResponse
        public var description: String {
            switch self {
            case .badBackendURL(let u): return "invalid Kreach backend URL: \(u)"
            case .httpStatus(let c): return "Kreach returned HTTP \(c)"
            case .badResponse: return "Kreach returned an unparseable response"
            }
        }
        // So `error.localizedDescription` returns this readable text.
        public var errorDescription: String? { description }
    }
}
