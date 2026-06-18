import XCTest
@testable import KLMTokenizer

/// Structural Gemma-4 media markers (e.g. `<end_of_image>` = `<image|>`) are
/// special tokens that demarcate a media run in the prompt. The model
/// occasionally emits one during plain text decode, where the byte-level
/// detokenizer would otherwise render the literal into the visible answer.
/// `decodeForOutput` must suppress those, while leaving `decode` (used by the
/// grammar piece table and parity paths) and every ordinary token untouched.
///
/// Resolution is vocab-specific, so the assertions need the real Gemma-4
/// tokenizer. The test loads it from `KLM_GEMMA_TOKENIZER_DIR` or the default
/// model-store path, and SKIPS (not fails) when no checkpoint is present so CI
/// without the weights stays green.
final class OutputSuppressionTests: XCTestCase {
    private func gemmaTokenizerDirIfPresent() -> URL? {
        let fm = FileManager.default
        var candidates: [URL] = []
        if let env = ProcessInfo.processInfo.environment["KLM_GEMMA_TOKENIZER_DIR"] {
            candidates.append(URL(fileURLWithPath: env))
        }
        let home = NSHomeDirectory()
        for name in ["gemma-4-12b", "gemma-4-12b-nvfp4", "gemma-4-e4b", "gemma-4-e2b"] {
            candidates.append(URL(fileURLWithPath: "\(home)/.krillm/models/blobs/\(name)"))
        }
        return candidates.first {
            fm.fileExists(atPath: $0.appendingPathComponent("tokenizer.json").path)
        }
    }

    func testGemmaMediaMarkersAreSuppressedInOutputOnly() async throws {
        guard let dir = gemmaTokenizerDirIfPresent() else {
            throw XCTSkip("No Gemma-4 tokenizer present; set KLM_GEMMA_TOKENIZER_DIR to run.")
        }
        let tok = try await KLMTokenizer(from: dir)

        // The reported leak: <end_of_image> rendered as the literal `<image|>`.
        let endImageIDs = tok.encode("<image|>").filter { $0 != tok.bosTokenId }
        try XCTSkipUnless(endImageIDs.count == 1,
                          "Tokenizer did not map <image|> to a single special token.")
        let endImageID = endImageIDs[0]

        XCTAssertTrue(tok.outputSuppressedTokenIDs.contains(endImageID),
                      "end_of_image id should be in the output-suppression set")
        XCTAssertEqual(tok.decode(token: endImageID), "<image|>",
                       "raw decode must still render the literal (grammar/parity paths)")
        XCTAssertEqual(tok.decodeForOutput(token: endImageID), "",
                       "decodeForOutput must suppress the structural media marker")

        // Every other media marker that exists as a single token is suppressed too.
        for literal in ["<|image|>", "<|audio|>", "<|image>", "<|audio>", "<audio|>"] {
            let ids = tok.encode(literal).filter { $0 != tok.bosTokenId }
            guard ids.count == 1 else { continue }
            XCTAssertEqual(tok.decodeForOutput(token: ids[0]), "",
                           "marker \(literal) should be suppressed in output")
        }

        // Ordinary prose tokens must round-trip identically through both paths.
        for word in ["Hello", " world", "image", "The"] {
            for id in tok.encode(word).filter({ $0 != tok.bosTokenId }) {
                XCTAssertFalse(tok.outputSuppressedTokenIDs.contains(id),
                               "ordinary token \(id) (from \(word.debugDescription)) must not be suppressed")
                XCTAssertEqual(tok.decodeForOutput(token: id), tok.decode(token: id),
                               "decodeForOutput must equal decode for ordinary tokens")
            }
        }
    }
}
