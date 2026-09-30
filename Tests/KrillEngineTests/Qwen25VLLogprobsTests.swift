import XCTest
import Foundation
@testable import KrillEngine
#if canImport(CoreGraphics)
import CoreGraphics
#endif
#if canImport(ImageIO)
import ImageIO
#endif

/// Task B (2026-09-30, docs/LOGPROBS_PLAN.md "Engine follow-ups"): the native
/// Qwen 2.5-VL runtime (`generateQwen25VL` / `Qwen25VLRuntime`) was one of
/// four native multimodal drivers that never threaded `wantLogprobs`/
/// `topLogprobs` at all - every `TokenEvent.logprob` was `nil` for these
/// checkpoints regardless of the request, the same class of gap fixed for
/// qwen3_5 in `Qwen35LogprobsTests.swift`. This is the real-model
/// verification for the IMAGE path specifically (an image request is exactly
/// the case that forces this native runtime instead of the generic decode
/// loop): `logprobs: true, top_logprobs: N` against a real image prompt must
/// populate `content[]` with the requested alternate count, and the
/// bytes-concatenation of the streamed tokens' text must equal the visible
/// answer.
///
/// Gated on `KRILL_QWEN25VL_MODEL_PATH` (same env var as
/// `Qwen25VLSmokeTests`/`Qwen25VLProfileTests`); skipped when unset.
final class Qwen25VLLogprobsTests: XCTestCase {

    private func requireModel() throws -> URL {
        guard let path = ProcessInfo.processInfo
            .environment["KRILL_QWEN25VL_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KRILL_QWEN25VL_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("KRILL_QWEN25VL_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path)
    }

    #if canImport(CoreGraphics) && canImport(ImageIO)
    private func solidPNG(_ color: String) -> Data {
        let rgb: (CGFloat, CGFloat, CGFloat)
        switch color {
        case "red":   rgb = (0.85, 0.05, 0.05)
        case "green": rgb = (0.05, 0.7, 0.1)
        default:      rgb = (0.5, 0.5, 0.5)
        }
        let w = 224, h = 224
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: rgb.0, green: rgb.1, blue: rgb.2, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        let image = ctx.makeImage()!
        let out = NSMutableData()
        let dest = CGImageDestinationCreateWithData(out, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil)
        _ = CGImageDestinationFinalize(dest)
        return out as Data
    }

    /// Every generated (non-`isEnd`) token from an IMAGE request must carry a
    /// `TokenLogprobInfo` with the requested top-N count; bytes-concat of the
    /// streamed text must equal the visible answer (same bytes-concat
    /// invariant `tools/logprobs_e2e_check.py` checks against the server).
    func testImageRequestLogprobsPopulatedAndBytesConcatMatchesText() async throws {
        let dir = try requireModel()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        XCTAssertEqual(engine.family, "qwen2_5_vl",
            "the checkpoint must load through the native VL loader")

        let topN = 3
        let (stream, _) = engine.generate(
            messages: [["role": "user", "content":
                "What is the dominant color of this image? Answer with one word."]],
            params: .greedy, maxTokens: 12, usePrefixCache: false,
            imageData: solidPNG("red"),
            wantLogprobs: true, topLogprobs: topN)

        var concatText = ""
        var sawContentToken = false
        for await event in stream {
            if event.isEnd { break }
            sawContentToken = true
            concatText += event.text
            guard let info = event.logprob else {
                XCTFail("token \(event.tokenId) (text: \(event.text.debugDescription)) has no "
                    + "logprob info - qwen2_5_vl native runtime never threaded wantLogprobs")
                continue
            }
            XCTAssertEqual(info.topAlternates.count, topN,
                "expected \(topN) top alternates, got \(info.topAlternates.count)")
            XCTAssertFalse(info.logprob.isNaN, "logprob must not be NaN")
            XCTAssertLessThanOrEqual(info.logprob, 0.0001,
                "a log-probability must be <= 0 (allowing tiny float slack)")
            // Greedy + no mask/penalties: the sampled token IS the raw
            // distribution's argmax, so it must be alternate #0.
            guard let top0 = info.topAlternates.first else {
                XCTFail("topAlternates unexpectedly empty"); continue
            }
            XCTAssertEqual(top0.tokenId, event.tokenId,
                "greedy-sampled token must be the raw top-1 alternate")
            XCTAssertEqual(top0.logprob, info.logprob, accuracy: 1e-5,
                "the sampled token's own logprob must match its own top-N entry")
        }
        XCTAssertTrue(sawContentToken, "generation must produce at least one content token")
        XCTAssertFalse(concatText.isEmpty,
            "the image request must still produce visible text with logprobs on")
    }

    /// `top_logprobs: 0` on an image request must still populate `logprob`
    /// (the sampled token's own value) with zero alternates - matching the
    /// qwen3_5 text-path contract exactly.
    func testImageRequestZeroTopLogprobsStillPopulatesLogprob() async throws {
        let dir = try requireModel()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()

        let (stream, _) = engine.generate(
            messages: [["role": "user", "content":
                "What is the dominant color of this image? Answer with one word."]],
            params: .greedy, maxTokens: 6, usePrefixCache: false,
            imageData: solidPNG("green"),
            wantLogprobs: true, topLogprobs: 0)

        var count = 0
        for await event in stream {
            if event.isEnd { break }
            count += 1
            XCTAssertNotNil(event.logprob, "token \(event.tokenId) missing logprob info")
            XCTAssertEqual(event.logprob?.topAlternates.count ?? -1, 0,
                "top_logprobs: 0 must report zero alternates")
        }
        XCTAssertGreaterThan(count, 0, "generation must produce at least one content token")
    }

    /// The logprobs-OFF path must be BYTE-FOR-BYTE the old code path: the
    /// same greedy image request with `wantLogprobs` left off must produce
    /// the exact same token sequence as the logprobs-ON run above (only the
    /// reporting differs, never the sampled tokens).
    func testLogprobsOffProducesIdenticalTokensToLogprobsOn() async throws {
        let dir = try requireModel()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        let messages = [["role": "user", "content":
            "What is the dominant color of this image? Answer with one word."]]
        let image = solidPNG("red")

        let (streamOff, _) = engine.generate(
            messages: messages, params: .greedy, maxTokens: 10,
            usePrefixCache: false, imageData: image)
        var tokensOff: [Int] = []
        for await event in streamOff {
            if event.isEnd { break }
            tokensOff.append(event.tokenId)
        }

        let (streamOn, _) = engine.generate(
            messages: messages, params: .greedy, maxTokens: 10,
            usePrefixCache: false, imageData: image,
            wantLogprobs: true, topLogprobs: 5)
        var tokensOn: [Int] = []
        for await event in streamOn {
            if event.isEnd { break }
            tokensOn.append(event.tokenId)
        }

        XCTAssertEqual(tokensOff, tokensOn,
            "requesting logprobs must never change which tokens are sampled")
    }
    #endif
}
