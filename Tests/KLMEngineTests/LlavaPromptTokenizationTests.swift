import XCTest
@testable import KLMTokenizer

/// Live-gated checks for `formatLlavaTokenIds`, the vicuna prompt builder the
/// engine uses for native LLaVA-1.5 image serving. Runs only when
/// `KLM_LLAVA_MODEL_PATH` points at a llava-1.5 checkpoint directory (CI
/// without the weights skips it, matching the repo's other live-gated smokes).
///
/// The load-bearing invariant: the prompt must contain EXACTLY `imagePadCount`
/// image-token ids (one per CLIP patch), contiguous, so
/// `LlavaForCausalLM`'s forward finds the right number of positions to splice
/// the projected vision features into. A miscount trips the model's
/// `imagePositions.count == n` precondition.
final class LlavaPromptTokenizationTests: XCTestCase {

    private func requireTokenizer() async throws -> KLMTokenizer {
        guard let path = ProcessInfo.processInfo.environment["KLM_LLAVA_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KLM_LLAVA_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("KLM_LLAVA_MODEL_PATH is not a directory: \(path)")
        }
        return try await KLMTokenizer(from: URL(fileURLWithPath: path))
    }

    func testImageTokenRunMatchesPadCount() async throws {
        let tok = try await requireTokenizer()
        let imageTokenId = 32_000   // llava-1.5 image_token_index
        let padCount = 576          // (336/14)^2
        let tokens = tok.formatLlavaTokenIds(
            messages: [["role": "user", "content": "What is in this image?"]],
            imageTokenId: imageTokenId,
            imagePadCount: padCount)

        let imageTokenCount = tokens.filter { $0 == imageTokenId }.count
        XCTAssertEqual(imageTokenCount, padCount,
            "the prompt must carry exactly imagePadCount image tokens")

        // The image tokens must be contiguous (a single run), as the model
        // splices one contiguous feature block.
        let first = tokens.firstIndex(of: imageTokenId)!
        let last = tokens.lastIndex(of: imageTokenId)!
        XCTAssertEqual(last - first + 1, padCount,
            "image tokens must form one contiguous run")
        // BOS leads the prompt; the run starts after the system + " USER: ".
        XCTAssertGreaterThan(first, 0, "BOS + system preamble must precede the image run")
    }

    func testSystemOnlyImageRequestStillPlacesImageRun() async throws {
        // Regression: a request whose only message is a system turn but which
        // carries an image must still emit the image-token run -- otherwise the
        // engine forwards pixels with zero image positions and the model's
        // `imagePositions.count == features` precondition aborts the process.
        let tok = try await requireTokenizer()
        let tokens = tok.formatLlavaTokenIds(
            messages: [["role": "system", "content": "Describe the image."]],
            imageTokenId: 32_000,
            imagePadCount: 576)
        XCTAssertEqual(tokens.filter { $0 == 32_000 }.count, 576,
            "a system-only request with an image must still place the image run")
    }

    func testTextOnlyHasNoImageTokens() async throws {
        let tok = try await requireTokenizer()
        let tokens = tok.formatLlavaTokenIds(
            messages: [["role": "user", "content": "Hello"]],
            imageTokenId: 32_000,
            imagePadCount: 0)
        XCTAssertFalse(tokens.contains(32_000),
            "a text-only request (imagePadCount 0) must place no image tokens")
        XCTAssertFalse(tokens.isEmpty)
    }
}
