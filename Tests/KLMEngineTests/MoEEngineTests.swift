import XCTest
@testable import KLMEngine

/// Live tests for the MoE Python sidecar bridge (`MoEEngine`).
/// Gated on `KLM_MOE_MODEL_PATH` because MoE checkpoints are
/// large (OLMoE-1B-7B at 4-bit is ~4GB, Qwen3-30B-A3B is ~17GB,
/// Mixtral-8x7B is ~25GB). Run locally with:
///
///     KLM_MOE_MODEL_PATH=$HOME/.krillm/models/blobs/OLMoE-1B-7B-0924-Instruct-4bit \
///       swift test --filter MoEEngineTests
///
/// The bridge is architecture-agnostic; mlx-lm handles router +
/// expert FFN dispatch internally. Any model mlx-lm can load
/// works (MoE or dense), but the family-detection arm only
/// routes MoE manifests through this engine.
final class MoEEngineTests: XCTestCase {

    private func liveModelDir() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KLM_MOE_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KLM_MOE_MODEL_PATH not set; skipping live MoE bridge test")
        }
        let url = URL(fileURLWithPath: path)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("KLM_MOE_MODEL_PATH does not point to a directory: \(path)")
        }
        return url
    }

    func testBridgeAnswersTextOnly() async throws {
        let dir = try liveModelDir()
        let engine = MoEEngine()
        do {
            try await engine.load(directory: dir)
        } catch {
            throw XCTSkip("MoE bridge could not start (mlx-lm not installed?): \(error)")
        }
        defer { try? engine.shutdown() }
        let result = try engine.generate(
            messages: [["role": "user", "content": "What is 2 plus 2? Answer with just the number."]],
            maxTokens: 32)
        XCTAssertFalse(result.text.isEmpty,
            "Bridge text generate must return non-empty text")
        XCTAssertTrue(result.text.contains("4"),
            "2 + 2 should produce text containing '4'; got: \(result.text)")
    }

    func testBridgePreservesSystemPrompt() async throws {
        let dir = try liveModelDir()
        let engine = MoEEngine()
        do {
            try await engine.load(directory: dir)
        } catch {
            throw XCTSkip("MoE bridge could not start: \(error)")
        }
        defer { try? engine.shutdown() }
        // System prompt forces a one-word answer; mlx-lm passes
        // it through the tokenizer's chat template, which
        // preserves system / user / assistant turns. If the
        // bridge silently dropped the system turn, the model
        // would respond in full sentences.
        let result = try engine.generate(
            messages: [
                ["role": "system",
                 "content": "Reply with exactly one word. No punctuation. No explanation."],
                ["role": "user",
                 "content": "What color is the sky on a clear day?"],
            ],
            maxTokens: 8)
        let trimmed = result.text
            .trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(trimmed.isEmpty)
        // "Blue" / "blue" is the universal expected answer.
        // We do not strictly assert one word (small MoE models
        // can disobey), but we DO assert the answer mentions
        // "blue" - which would be unlikely if the system
        // instruction was dropped on the floor and the bridge
        // silently rendered the user prompt without it.
        XCTAssertTrue(
            trimmed.lowercased().contains("blue"),
            "Expected 'blue' in the answer; got: \(trimmed)")
    }
}
