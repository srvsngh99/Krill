import Foundation
import XCTest
@testable import KLMRegistry

final class PullerTests: XCTestCase {
    func testListRepoFilesSendsInjectedAuthToken() async throws {
        let tempDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let client = MockPullerHTTPClient(
            tempDir: tempDir,
            dataResponses: [
                .init(
                    statusCode: 200,
                    body: Data(
                        #"{"siblings":[{"rfilename":"config.json","size":12}]}"#.utf8
                    )
                )
            ]
        )
        let puller = Puller(
            registry: Registry(baseDir: tempDir),
            httpClient: client,
            tokenProvider: { "test-token" },
            sleeper: { _ in }
        )

        let files = try await puller.listRepoFiles(repo: "org/model")

        XCTAssertEqual(files.map(\.name), ["config.json"])
        let requests = client.recordedDataRequests()
        XCTAssertEqual(requests.count, 1)
        XCTAssertEqual(
            requests[0].value(forHTTPHeaderField: "Authorization"),
            "Bearer test-token"
        )
    }

    func testDownloadRetriesTransientServerErrorWithoutRealSleepAndSendsAuth() async throws {
        let tempDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let client = MockPullerHTTPClient(
            tempDir: tempDir,
            downloadResponses: [
                .init(statusCode: 500, body: Data("server error".utf8)),
                .init(statusCode: 200, body: Data("ok".utf8)),
            ]
        )
        let sleeps = SleepRecorder()
        let puller = Puller(
            registry: Registry(baseDir: tempDir),
            httpClient: client,
            tokenProvider: { "download-token" },
            sleeper: { nanoseconds in
                sleeps.record(nanoseconds)
            }
        )

        let destination = tempDir.appendingPathComponent("model.safetensors")
        _ = try await puller.downloadFile(
            repo: "org/model",
            filename: "model.safetensors",
            destination: destination
        )

        XCTAssertEqual(try Data(contentsOf: destination), Data("ok".utf8))
        XCTAssertEqual(sleeps.recorded(), [1_000_000_000])

        let requests = client.recordedDownloadRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertTrue(requests.allSatisfy {
            $0.value(forHTTPHeaderField: "Authorization") == "Bearer download-token"
        })
    }

    func testDownloadResumesFromPartialFileAndAppendsHTTP206Body() async throws {
        let tempDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let destination = tempDir.appendingPathComponent("model.safetensors")
        let partial = destination.appendingPathExtension("partial")
        try Data("hello ".utf8).write(to: partial)

        let client = MockPullerHTTPClient(
            tempDir: tempDir,
            downloadResponses: [
                .init(statusCode: 206, body: Data("world".utf8)),
            ]
        )
        let puller = Puller(
            registry: Registry(baseDir: tempDir),
            httpClient: client,
            tokenProvider: { nil },
            sleeper: { _ in }
        )

        let sha256 = try await puller.downloadFile(
            repo: "org/model",
            filename: "model.safetensors",
            destination: destination
        )

        let requests = client.recordedDownloadRequests()
        XCTAssertEqual(requests.count, 1)
        XCTAssertEqual(requests[0].value(forHTTPHeaderField: "Range"), "bytes=6-")
        XCTAssertEqual(try Data(contentsOf: destination), Data("hello world".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path))
        XCTAssertEqual(
            sha256,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )
    }

    func testIncrementalSHA256MatchesKnownBytes() throws {
        let tempDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let file = tempDir.appendingPathComponent("known-bytes.bin")
        try Data("abc".utf8).write(to: file)

        let puller = Puller(registry: Registry(baseDir: tempDir))
        XCTAssertEqual(
            try puller.incrementalSHA256(of: file),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
    }

    func testPullRejectsTraversalNameBeforeAnyFilesystemOp() async throws {
        let tempDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let registry = Registry(baseDir: tempDir)
        try registry.ensureDirectories()
        // Sentinel that a registry-root wipe would destroy.
        let sentinel = registry.modelsDir.appendingPathComponent("sentinel.txt")
        try "keep".write(to: sentinel, atomically: true, encoding: .utf8)

        let puller = Puller(registry: registry)
        let evil = ResolvedModel(
            repo: "x/..", name: "..", family: .llama,
            params: "?", quant: "4bit", context: 8192)
        do {
            _ = try await puller.pull(evil)
            XCTFail("expected pull to reject traversal name")
        } catch {
            // expected - RegistryError.invalidModelName
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: sentinel.path),
                      "registry must not be touched by a rejected pull")
    }

    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-puller-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

private final class MockPullerHTTPClient: PullerHTTPClient, @unchecked Sendable {
    struct MockResponse {
        let statusCode: Int
        let body: Data
    }

    private let lock = NSLock()
    private let tempDir: URL
    private var dataResponses: [MockResponse]
    private var downloadResponses: [MockResponse]
    private var dataRequests: [URLRequest] = []
    private var downloadRequests: [URLRequest] = []

    init(
        tempDir: URL,
        dataResponses: [MockResponse] = [],
        downloadResponses: [MockResponse] = []
    ) {
        self.tempDir = tempDir
        self.dataResponses = dataResponses
        self.downloadResponses = downloadResponses
    }

    func data(for request: URLRequest) async throws -> (Data, URLResponse) {
        let response = try locked {
            dataRequests.append(request)
            guard !dataResponses.isEmpty else {
                throw MockPullerHTTPClientError.missingResponse
            }
            return dataResponses.removeFirst()
        }

        return (
            response.body,
            try httpResponse(for: request, statusCode: response.statusCode)
        )
    }

    func download(for request: URLRequest) async throws -> (URL, URLResponse) {
        let response = try locked {
            downloadRequests.append(request)
            guard !downloadResponses.isEmpty else {
                throw MockPullerHTTPClientError.missingResponse
            }
            return downloadResponses.removeFirst()
        }

        let tempURL = tempDir
            .appendingPathComponent("mock-download-\(UUID().uuidString).tmp")
        try response.body.write(to: tempURL)
        return (
            tempURL,
            try httpResponse(for: request, statusCode: response.statusCode)
        )
    }

    func recordedDataRequests() -> [URLRequest] {
        locked { dataRequests }
    }

    func recordedDownloadRequests() -> [URLRequest] {
        locked { downloadRequests }
    }

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    private func httpResponse(
        for request: URLRequest,
        statusCode: Int
    ) throws -> HTTPURLResponse {
        guard let url = request.url,
              let response = HTTPURLResponse(
                  url: url,
                  statusCode: statusCode,
                  httpVersion: nil,
                  headerFields: nil
              ) else {
            throw MockPullerHTTPClientError.invalidResponse
        }
        return response
    }
}

private final class SleepRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [UInt64] = []

    func record(_ nanoseconds: UInt64) {
        lock.lock()
        values.append(nanoseconds)
        lock.unlock()
    }

    func recorded() -> [UInt64] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}

private enum MockPullerHTTPClientError: Error {
    case missingResponse
    case invalidResponse
}
