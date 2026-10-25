import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

// Optional backend, gated behind the `SELFHOSTED_INDEX` compile flag (set via KRILL_SELFHOSTED_INDEX=1
// — see Package.swift / `make dev`). The entire type is compiled in ONLY for
// local development builds and is ABSENT from public release binaries. Public
// users get DuckDuckGo (default) + Brave/Tavily (BYOK) + SearXNG instead.
#if SELFHOSTED_INDEX

/// Self-hosted index backend: queries the user's own search index service
/// (`<base>/search?q=...&limit=N`) and maps its ranked results into
/// `SearchResult`s. This is the conformer the `SearchBackend` protocol was
/// designed to accept — selecting it (`search_backend = "selfhosted"`) is the only
/// change; the `web_search` tool and the `DeepResearch` orchestrator are
/// backend-agnostic.
///
/// The role this plays: the self-hosted index is the *search platform* underneath Krill's
/// `DeepResearch` — the index retrieves over its owned, crawled index; the agent
/// (this process, driven by the local model) plans the queries, reads the
/// pages, and synthesizes the cited answer. Every client that talks to Krill
/// built on Krill gets deep research over the index for free via the
/// server's `/research` route, without reimplementing the loop.
///
/// The index is a trusted service the user runs (typically loopback on
/// `127.0.0.1:8000`), so — exactly like `SearxngBackend` — the first hop is
/// allowed to reach a private/loopback host (the shared `URLSessionWebFetcher`
/// only vets *redirects*, not the first request, and the index answers 200
/// directly without redirecting).
public struct SelfHostedIndexBackend: SearchBackend {
    public let name = "selfhosted"
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
            throw SelfHostedIndexError.badBackendURL(baseURL)
        }
        comps.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "limit", value: String(max(1, count))),
        ]
        guard let url = comps.url else { throw SelfHostedIndexError.badBackendURL(baseURL) }

        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        request.setValue("krill/web_search", forHTTPHeaderField: "User-Agent")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await fetcher.fetch(request)
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw SelfHostedIndexError.httpStatus(http.statusCode)
        }
        return try Self.parse(data, limit: count)
    }

    /// Parse a self-hosted index `/search` JSON body (`{results:[{title,url,snippet,...}]}`)
    /// into the top `limit` results, in rank order.
    static func parse(_ data: Data, limit: Int) throws -> [SearchResult] {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = obj["results"] as? [[String: Any]]
        else { throw SelfHostedIndexError.badResponse }
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

    public enum SelfHostedIndexError: Error, CustomStringConvertible, LocalizedError {
        case badBackendURL(String)
        case httpStatus(Int)
        case badResponse
        public var description: String {
            switch self {
            case .badBackendURL(let u): return "invalid self-hosted index backend URL: \(u)"
            case .httpStatus(let c): return "self-hosted index returned HTTP \(c)"
            case .badResponse: return "self-hosted index returned an unparseable response"
            }
        }
        // So `error.localizedDescription` returns this readable text.
        public var errorDescription: String? { description }
    }
}

#endif // SELFHOSTED_INDEX
