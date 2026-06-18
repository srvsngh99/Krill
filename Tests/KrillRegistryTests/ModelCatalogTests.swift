import XCTest
@testable import KrillRegistry

/// Tests for the remote model catalog: the JSON shape, the on-disk
/// store (load / save / staleness), remote fetch, and the `AliasMap`
/// fallback that lets a catalog model be pulled without a rebuild.
final class ModelCatalogTests: XCTestCase {

    /// A fresh, isolated temp directory used as the registry base dir.
    private func makeTempBaseDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-catalog-test-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// A catalog whose aliases are deliberately NOT built-in aliases,
    /// so `AliasMap.resolve` without a store returns nil for them.
    private func sampleCatalog() -> ModelCatalog {
        ModelCatalog(updated: "2026-05-22", models: [
            CatalogEntry(alias: "catalog-test-alpha",
                         repo: "mlx-community/Catalog-Test-Alpha-4bit",
                         family: .qwen, params: "4B", quant: "4bit", context: 32768),
            CatalogEntry(alias: "Catalog-Test-Beta",
                         repo: "mlx-community/catalog-test-beta-4bit",
                         family: .phi, params: "3B", quant: "4bit", context: 8192),
        ])
    }

    // MARK: - Entry / catalog shape

    func testCatalogEntryResolvedMirrorsFields() {
        let entry = CatalogEntry(
            alias: "qwen3-4b", repo: "mlx-community/Qwen3-4B-4bit",
            family: .qwen, params: "4B", quant: "4bit", context: 32768)
        let resolved = entry.resolved
        XCTAssertEqual(resolved.name, "qwen3-4b")
        XCTAssertEqual(resolved.repo, "mlx-community/Qwen3-4B-4bit")
        XCTAssertEqual(resolved.family, .qwen)
        XCTAssertEqual(resolved.params, "4B")
        XCTAssertEqual(resolved.quant, "4bit")
        XCTAssertEqual(resolved.context, 32768)
    }

    func testCatalogJSONRoundTrips() throws {
        let catalog = sampleCatalog()
        let data = try JSONEncoder().encode(catalog)
        let decoded = try JSONDecoder().decode(ModelCatalog.self, from: data)
        XCTAssertEqual(decoded, catalog)
        XCTAssertEqual(decoded.schemaVersion, ModelCatalog.currentSchemaVersion)
    }

    // MARK: - Store load / save

    func testStoreSaveAndLoadRoundTrips() throws {
        let store = ModelCatalogStore(baseDir: makeTempBaseDir())
        XCTAssertNil(store.load(), "a fresh store has no cache")
        try store.save(sampleCatalog())
        XCTAssertEqual(store.load(), sampleCatalog())
    }

    func testStoreLoadRejectsUnsupportedSchema() throws {
        let store = ModelCatalogStore(baseDir: makeTempBaseDir())
        // A future schema must be treated as absent, never mis-decoded.
        let future = """
        {"schemaVersion": 999, "models": []}
        """
        try future.data(using: .utf8)!.write(to: store.catalogURL)
        XCTAssertNil(store.load(),
            "a catalog with an unknown schema version must load as nil")
    }

    func testStoreLoadRejectsMalformedJSON() throws {
        let store = ModelCatalogStore(baseDir: makeTempBaseDir())
        try "not json".data(using: .utf8)!.write(to: store.catalogURL)
        XCTAssertNil(store.load())
    }

    // MARK: - Staleness

    func testIsStaleWhenNoCache() {
        let store = ModelCatalogStore(baseDir: makeTempBaseDir())
        XCTAssertNil(store.cacheAge())
        XCTAssertTrue(store.isStale(ttl: 3600))
    }

    func testFreshCacheIsNotStale() throws {
        let store = ModelCatalogStore(baseDir: makeTempBaseDir())
        try store.save(sampleCatalog())
        XCTAssertNotNil(store.cacheAge())
        XCTAssertFalse(store.isStale(ttl: 3600),
            "a just-written cache must not be stale under a 1h TTL")
        XCTAssertTrue(store.isStale(ttl: -1),
            "a negative TTL makes any existing cache stale")
    }

    // MARK: - Catalog resolution

    func testStoreResolveIsCaseInsensitive() throws {
        let store = ModelCatalogStore(baseDir: makeTempBaseDir())
        try store.save(sampleCatalog())
        XCTAssertEqual(store.resolve("CATALOG-TEST-BETA")?.repo,
                       "mlx-community/catalog-test-beta-4bit")
        XCTAssertEqual(store.resolve("  catalog-test-alpha ")?.family, .qwen)
    }

    func testStoreResolveUnknownIsNil() throws {
        let store = ModelCatalogStore(baseDir: makeTempBaseDir())
        try store.save(sampleCatalog())
        XCTAssertNil(store.resolve("no-such-model"))
    }

    // MARK: - AliasMap fallback

    func testAliasMapResolvesCatalogModelAsFallback() throws {
        let store = ModelCatalogStore(baseDir: makeTempBaseDir())
        try store.save(sampleCatalog())
        // Without a store, a catalog-only model is unknown.
        XCTAssertNil(AliasMap.resolve("catalog-test-alpha"))
        // With the store, it resolves.
        XCTAssertEqual(
            AliasMap.resolve("catalog-test-alpha", catalog: store)?.repo,
            "mlx-community/Catalog-Test-Alpha-4bit")
    }

    func testBuiltInAliasWinsOverCatalog() throws {
        let store = ModelCatalogStore(baseDir: makeTempBaseDir())
        // Pick a real built-in alias and try to shadow it.
        guard let builtIn = AliasMap.allAliases.first else {
            return XCTFail("expected a built-in alias")
        }
        try store.save(ModelCatalog(models: [
            CatalogEntry(alias: builtIn.shortName,
                         repo: "attacker/shadow-repo", family: .llama,
                         params: "?", quant: "4bit", context: 8192),
        ]))
        XCTAssertEqual(
            AliasMap.resolve(builtIn.shortName, catalog: store)?.repo,
            builtIn.model.repo,
            "the curated built-in alias must win over a catalog entry")
    }

    func testAliasMapHFPathStillResolvesWithCatalog() {
        let store = ModelCatalogStore(baseDir: makeTempBaseDir())
        let resolved = AliasMap.resolve("some-org/some-model", catalog: store)
        XCTAssertEqual(resolved?.repo, "some-org/some-model")
    }

    // MARK: - Remote fetch

    func testFetchFromFileURLSucceedsAndCaches() async throws {
        let baseDir = makeTempBaseDir()
        let store = ModelCatalogStore(baseDir: baseDir)
        // Serve the catalog from a local file:// URL - exercises the
        // real fetch path (decode + schema check + save) without a
        // network dependency.
        let sourceURL = baseDir.appendingPathComponent("remote-catalog.json")
        try JSONEncoder().encode(sampleCatalog()).write(to: sourceURL)

        let fetched = try await store.fetch(from: sourceURL)
        XCTAssertEqual(fetched, sampleCatalog())
        XCTAssertEqual(store.load(), sampleCatalog(),
            "fetch must persist the catalog to the local cache")
    }

    func testFetchRejectsUnsupportedSchema() async throws {
        let baseDir = makeTempBaseDir()
        let store = ModelCatalogStore(baseDir: baseDir)
        let sourceURL = baseDir.appendingPathComponent("future.json")
        try #"{"schemaVersion": 999, "models": []}"#
            .data(using: .utf8)!.write(to: sourceURL)

        do {
            _ = try await store.fetch(from: sourceURL)
            XCTFail("expected CatalogError.unsupportedSchema")
        } catch let error as CatalogError {
            XCTAssertEqual(error, .unsupportedSchema(999))
        }
        XCTAssertNil(store.load(), "a rejected fetch must not write the cache")
    }

    func testFetchRejectsMalformedPayload() async throws {
        let baseDir = makeTempBaseDir()
        let store = ModelCatalogStore(baseDir: baseDir)
        let sourceURL = baseDir.appendingPathComponent("garbage.json")
        try "<<not json>>".data(using: .utf8)!.write(to: sourceURL)

        do {
            _ = try await store.fetch(from: sourceURL)
            XCTFail("expected CatalogError.malformed")
        } catch let error as CatalogError {
            guard case .malformed = error else {
                return XCTFail("expected .malformed, got \(error)")
            }
        }
    }

    func testFetchRejectsNon2xxHTTPStatus() async throws {
        let store = ModelCatalogStore(baseDir: makeTempBaseDir())
        StubURLProtocol.statusCode = 404
        StubURLProtocol.responseData = Data()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        let session = URLSession(configuration: config)

        do {
            _ = try await store.fetch(
                from: URL(string: "https://example.com/catalog.json")!,
                session: session)
            XCTFail("expected CatalogError.httpStatus")
        } catch let error as CatalogError {
            XCTAssertEqual(error, .httpStatus(404))
        }
        XCTAssertNil(store.load())
    }
}

/// Minimal `URLProtocol` stub: returns a canned HTTP status + body so
/// the catalog fetch error paths can be tested without a real server.
final class StubURLProtocol: URLProtocol {
    nonisolated(unsafe) static var statusCode = 200
    nonisolated(unsafe) static var responseData: Data?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }
    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!, statusCode: Self.statusCode,
            httpVersion: "HTTP/1.1", headerFields: nil)!
        client?.urlProtocol(self, didReceive: response,
                            cacheStoragePolicy: .notAllowed)
        if let data = Self.responseData {
            client?.urlProtocol(self, didLoad: data)
        }
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
