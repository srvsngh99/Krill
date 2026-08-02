import Foundation
import XCTest
@testable import KrillRegistry

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

    func testForcedPullFailurePreservesExistingModelAndCleansStaging() async throws {
        let tempDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let registry = Registry(baseDir: tempDir)
        try seedInstalledModel(named: "test-model", registry: registry, contents: "old weights")

        let client = MockPullerHTTPClient(
            tempDir: tempDir,
            dataResponses: [
                .init(
                    statusCode: 200,
                    body: Data(#"{"siblings":[{"rfilename":"model.safetensors","size":11}]}"#.utf8)
                )
            ],
            downloadResponses: [
                .init(statusCode: 500, body: Data()),
                .init(statusCode: 500, body: Data()),
                .init(statusCode: 500, body: Data()),
            ]
        )
        let puller = Puller(
            registry: registry,
            httpClient: client,
            tokenProvider: { nil },
            sleeper: { _ in }
        )
        let resolved = ResolvedModel(
            repo: "org/new-model", name: "test-model", family: .llama,
            params: "1B", quant: "4bit", context: 4096
        )

        do {
            _ = try await puller.pull(resolved, force: true)
            XCTFail("expected forced pull to fail")
        } catch {
            // expected after all retry attempts fail
        }

        let weight = registry.modelPath("test-model").appendingPathComponent("model.safetensors")
        XCTAssertEqual(try String(contentsOf: weight, encoding: .utf8), "old weights")
        XCTAssertEqual(registry.getModel("test-model")?.source, "org/old-model")
        XCTAssertEqual(client.recordedDownloadRequests().count, 3)
        let entries = try FileManager.default.contentsOfDirectory(atPath: registry.modelsDir.path)
        XCTAssertFalse(entries.contains { $0.hasPrefix(".pull-") })
    }

    func testForcedPullCommitsStagedBlobAndManifestAfterSuccessfulDownload() async throws {
        let tempDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let registry = Registry(baseDir: tempDir)
        try seedInstalledModel(named: "test-model", registry: registry, contents: "old weights")

        let newWeights = Data("new weights".utf8)
        let client = MockPullerHTTPClient(
            tempDir: tempDir,
            dataResponses: [
                .init(
                    statusCode: 200,
                    body: Data(#"{"siblings":[{"rfilename":"model.safetensors","size":11}]}"#.utf8)
                )
            ],
            downloadResponses: [.init(statusCode: 200, body: newWeights)]
        )
        let puller = Puller(
            registry: registry,
            httpClient: client,
            tokenProvider: { nil },
            sleeper: { _ in }
        )
        let resolved = ResolvedModel(
            repo: "org/new-model", name: "test-model", family: .llama,
            params: "1B", quant: "4bit", context: 4096
        )

        let manifest = try await puller.pull(resolved, force: true)

        let weight = registry.modelPath("test-model").appendingPathComponent("model.safetensors")
        XCTAssertEqual(try Data(contentsOf: weight), newWeights)
        XCTAssertEqual(manifest.source, "org/new-model")
        XCTAssertEqual(registry.getModel("test-model")?.source, "org/new-model")
        XCTAssertEqual(manifest.files.first?.sha256,
                       "75f585ae1855ec1d1ba5e4fc9861e07fbdba3919a965042b2ba78b6fc3e9a18f")
        let entries = try FileManager.default.contentsOfDirectory(atPath: registry.modelsDir.path)
        XCTAssertFalse(entries.contains { $0.hasPrefix(".pull-") })
    }

    func testPullRejectsUnsafeRepoFilenameWithoutEscapingStagingDirectory() async throws {
        let tempDir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let registry = Registry(baseDir: tempDir)
        let client = MockPullerHTTPClient(
            tempDir: tempDir,
            dataResponses: [
                .init(
                    statusCode: 200,
                    body: Data(#"{"siblings":[{"rfilename":"../escape.safetensors","size":4}]}"#.utf8)
                )
            ]
        )
        let puller = Puller(
            registry: registry,
            httpClient: client,
            tokenProvider: { nil },
            sleeper: { _ in }
        )
        let resolved = ResolvedModel(
            repo: "org/model", name: "safe-name", family: .llama,
            params: "1B", quant: "4bit", context: 4096
        )

        await XCTAssertThrowsErrorAsync {
            _ = try await puller.pull(resolved)
        }

        XCTAssertFalse(FileManager.default.fileExists(
            atPath: registry.modelsDir.appendingPathComponent("escape.safetensors").path
        ))
        XCTAssertTrue(client.recordedDownloadRequests().isEmpty)
    }

    func testIsEssentialFileAcceptsCoreAndShardedWeights() {
        // Weight shards and the shard map both ride along.
        XCTAssertTrue(Puller.isEssentialFile("model.safetensors"))
        XCTAssertTrue(Puller.isEssentialFile("model-00001-of-00004.safetensors"))
        XCTAssertTrue(Puller.isEssentialFile("model.safetensors.index.json"))
        // Core model + generation config.
        XCTAssertTrue(Puller.isEssentialFile("config.json"))
        XCTAssertTrue(Puller.isEssentialFile("generation_config.json"))
    }

    func testIsEssentialFileAcceptsAllTokenizerArtifacts() {
        // Merged tokenizer form.
        XCTAssertTrue(Puller.isEssentialFile("tokenizer.json"))
        XCTAssertTrue(Puller.isEssentialFile("tokenizer_config.json"))
        XCTAssertTrue(Puller.isEssentialFile("special_tokens_map.json"))
        XCTAssertTrue(Puller.isEssentialFile("tokenizer.model"))
        // Older BPE form (Qwen, GPT-2 style) and the added-token sidecar
        // that ships tool / chat special tokens.
        XCTAssertTrue(Puller.isEssentialFile("added_tokens.json"))
        XCTAssertTrue(Puller.isEssentialFile("vocab.json"))
        XCTAssertTrue(Puller.isEssentialFile("merges.txt"))
    }

    func testIsEssentialFileAcceptsExternalChatTemplateInBothForms() {
        // Two co-existing on-disk forms today, both must ride along
        // or the tokenizer raises "chat_template is not set":
        //   .jinja: Qwen 3 (Coder, Instruct-2507), Gemma 4 (e2b, e4b).
        //   .json:  Gemma 3, Qwen 2.5-VL.
        XCTAssertTrue(Puller.isEssentialFile("chat_template.jinja"))
        XCTAssertTrue(Puller.isEssentialFile("chat_template.json"))
    }

    func testIsEssentialFileAcceptsMultimodalPreprocessorConfigs() {
        // Qwen 2.5-VL ships `preprocessor_config.json`; Gemma 3 / 4
        // ship `processor_config.json` (Gemma 3 ships both). Image-mean
        // / patch-size / dtype live there; the puller captures them
        // even when the runtime loader currently hardcodes the values.
        XCTAssertTrue(Puller.isEssentialFile("preprocessor_config.json"))
        XCTAssertTrue(Puller.isEssentialFile("processor_config.json"))
    }

    func testIsEssentialFileRejectsRepoClutter() {
        // Repo READMEs / metadata: skipped.
        XCTAssertFalse(Puller.isEssentialFile("README.md"))
        XCTAssertFalse(Puller.isEssentialFile(".gitattributes"))
        // Sample assets and license PDFs that mlx-community sometimes ships.
        XCTAssertFalse(Puller.isEssentialFile("sample.png"))
        XCTAssertFalse(Puller.isEssentialFile("LICENSE"))
        // Original PyTorch checkpoints. HF returns paths inside
        // `original/`; the suffix check rejects them since they're
        // .bin/.pth rather than .safetensors. A bare `pytorch_model.bin`
        // is the bigger risk and is also rejected.
        XCTAssertFalse(Puller.isEssentialFile("pytorch_model.bin"))
        XCTAssertFalse(Puller.isEssentialFile("original/consolidated.pth"))
        // Adjacent tool-parsing scripts (Qwen3-Coder ships one) - the
        // model serves without them.
        XCTAssertFalse(Puller.isEssentialFile("qwen3coder_tool_parser.py"))
    }

    func testIsEssentialFileMatchesCaseInsensitively() {
        // HF filenames are case-stable but the loader treats them
        // case-insensitively. Make sure both forms work.
        XCTAssertTrue(Puller.isEssentialFile("Chat_Template.JINJA"))
        XCTAssertTrue(Puller.isEssentialFile("Model.Safetensors"))
    }

    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-puller-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func seedInstalledModel(
        named name: String,
        registry: Registry,
        contents: String
    ) throws {
        try registry.ensureDirectories()
        let blob = registry.modelPath(name)
        try FileManager.default.createDirectory(at: blob, withIntermediateDirectories: true)
        try Data(contents.utf8).write(to: blob.appendingPathComponent("model.safetensors"))
        try registry.saveManifest(ModelManifest(
            name: name,
            family: .llama,
            params: "1B",
            quant: "4bit",
            source: "org/old-model",
            context: 4096,
            files: [],
            chatTemplate: "llama",
            sizeBytes: Int64(contents.utf8.count)
        ))
    }
}

private func XCTAssertThrowsErrorAsync(
    _ expression: () async throws -> Void,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        try await expression()
        XCTFail("expected expression to throw", file: file, line: line)
    } catch {
        // expected
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
