import XCTest
@testable import KrillEngine

/// `stopTokenIds` unions the tokenizer EOS with the model's declared
/// `eos_token_id`. Phi-4-mini ships no `generation_config.json` and its
/// tokenizer EOS (`<|endoftext|>`, 199999) differs from its chat turn
/// terminator (`<|end|>`, 200020, declared only in config.json) - without the
/// config.json fallback the assistant never halts and runs on.
final class StopTokenIdsTests: XCTestCase {

    private func tempDir(_ files: [String: String]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("stoptok-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for (name, body) in files {
            try body.write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
        return dir
    }

    func testConfigJsonEosUnioned() throws {
        // Phi-4-mini case: tokenizer EOS 199999, no generation_config.json,
        // config.json declares the real chat terminator 200020.
        let dir = try tempDir(["config.json": #"{"eos_token_id": 200020}"#])
        defer { try? FileManager.default.removeItem(at: dir) }
        let ids = InferenceEngine.stopTokenIds(modelDirectory: dir, tokenizerEOS: 199999)
        XCTAssertTrue(ids.contains(199999), "tokenizer EOS is always a stop token")
        XCTAssertTrue(ids.contains(200020), "config.json eos_token_id must be unioned in")
    }

    func testGenerationConfigArrayStillHonored() throws {
        // Gemma 4 style: generation_config lists multiple stop ids; both files
        // present, all stop ids unioned.
        let dir = try tempDir([
            "generation_config.json": #"{"eos_token_id": [1, 106, 50]}"#,
            "config.json": #"{"eos_token_id": 1}"#,
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        let ids = InferenceEngine.stopTokenIds(modelDirectory: dir, tokenizerEOS: 1)
        XCTAssertEqual(ids, [1, 50, 106])
    }

    func testFallsBackToTokenizerEosWhenNoFiles() throws {
        let dir = try tempDir([:])
        defer { try? FileManager.default.removeItem(at: dir) }
        let ids = InferenceEngine.stopTokenIds(modelDirectory: dir, tokenizerEOS: 128001)
        XCTAssertEqual(ids, [128001])
    }
}
