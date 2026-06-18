import XCTest
@testable import KrillTokenizer

/// Live-gated golden for the phi chat-prompt tokenization fix
/// (`InferenceEngine` routes `family == "phi"` through string-render +
/// `encodeWithoutExtraBOS` instead of the direct `applyChatTemplateTokens`,
/// because swift-transformers' direct path mis-tokenizes phi-4-mini's o200k
/// BPE body and the model degenerates).
///
/// Runs only when `KRILL_PHI_MODEL_PATH` points at a phi-4-mini checkpoint
/// directory (CI without the weights skips it, matching the repo's other
/// live-gated smokes). It locks the two properties the fix relies on:
/// the rendered chat string carries the native `<|user|>`/`<|assistant|>`
/// markers, and re-encoding it is the canonical (idempotent) tokenization
/// with the special tokens as single ids - not byte-fragmented.
final class PhiPromptTokenizationTests: XCTestCase {

    private func requireTokenizer() async throws -> KrillTokenizer {
        guard let path = ProcessInfo.processInfo.environment["KRILL_PHI_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KRILL_PHI_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("KRILL_PHI_MODEL_PATH is not a directory: \(path)")
        }
        return try await KrillTokenizer(from: URL(fileURLWithPath: path))
    }

    func testPhiChatPromptIsCanonicalAndKeepsSpecialTokens() async throws {
        let tok = try await requireTokenizer()
        let messages = [["role": "user", "content": "Hi"]]

        let rendered = tok.applyChatTemplate(messages: messages)
        XCTAssertTrue(rendered.contains("<|user|>"),
            "phi template must frame the user turn natively")
        XCTAssertTrue(rendered.contains("<|assistant|>"),
            "phi template must open the assistant turn for generation")

        // This is exactly what InferenceEngine's phi branch builds.
        let tokens = tok.encodeWithoutExtraBOS(rendered)
        XCTAssertFalse(tokens.isEmpty)

        // Special tokens collapse to single ids: `<|user|>Hi<|end|><|assistant|>`
        // is a handful of tokens, not a byte-fragmented blow-up (a fragmented
        // prompt is exactly what garbled the model).
        XCTAssertLessThan(tokens.count, 16,
            "special tokens must be single ids, not byte-fragmented (got \(tokens.count))")

        // Canonical = a fixed point: re-encoding the decoded prompt yields the
        // same ids. The broken direct path produced non-canonical boundaries
        // that fail this round-trip.
        let roundTrip = tok.encodeWithoutExtraBOS(tok.decode(tokens))
        XCTAssertEqual(tokens, roundTrip,
            "phi chat prompt tokenization must be canonical (idempotent)")
    }
}
