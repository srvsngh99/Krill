import XCTest
import Foundation
@testable import KrillEngine
#if canImport(CoreGraphics)
import CoreGraphics
#endif
#if canImport(ImageIO)
import ImageIO
#endif

/// Task B (2026-09-30, docs/LOGPROBS_PLAN.md "Engine follow-ups"): Gemma 4's
/// multimodal forward (image AND audio) does NOT go through any of the four
/// dedicated native VL runtimes audited elsewhere in this change - it uses
/// standard 1-D RoPE, so `InferenceEngine.generate(messages:)` serves it
/// through the GENERIC dense decode loop (the one Phase 1's `wantLogprobs`
/// plumbing was originally wired into), never intercepted by
/// `generateLlamaVision`/`generateQwen25VL`/`generateQwen35VL`/
/// `generateLocateAnything`/`generateMuseGlimmer`. This test verifies that
/// claim for real: a gemma-4 request WITH an image, and a separate request
/// WITH audio, both populate `TokenEvent.logprob` when `wantLogprobs: true`.
///
/// Both media inputs are generated in-process (a solid-color PNG via
/// CoreGraphics/ImageIO, a short synthetic sine-tone WAV via a hand-built
/// PCM16 header) rather than reused from a checkpointed fixture file.
///
/// Gated on `KRILL_GEMMA4_MODEL_PATH` (same env var as the other Gemma 4
/// tests); run for real against the on-disk `gemma-4-e2b` checkpoint.
final class Gemma4MultimodalLogprobsTests: XCTestCase {

    private func requireModel() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KRILL_GEMMA4_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KRILL_GEMMA4_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("KRILL_GEMMA4_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path)
    }

    #if canImport(CoreGraphics) && canImport(ImageIO)
    private func solidPNG(_ color: String) -> Data {
        let rgb: (CGFloat, CGFloat, CGFloat)
        switch color {
        case "blue": rgb = (0.05, 0.1, 0.85)
        default:     rgb = (0.5, 0.5, 0.5)
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
    #endif

    /// A short mono 16-bit PCM WAV: a 1 kHz sine tone, generated in-process
    /// (no checkpointed audio fixture).
    private func sineWAV(seconds: Double = 1.5, sampleRate: Int = 16_000, freqHz: Double = 1000) -> Data {
        let n = Int(Double(sampleRate) * seconds)
        var samples = [Int16](repeating: 0, count: n)
        for i in 0 ..< n {
            let t = Double(i) / Double(sampleRate)
            samples[i] = Int16((sin(2.0 * Double.pi * freqHz * t) * 0.3 * 32767.0).rounded())
        }
        var data = Data()
        func append32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func append16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let byteRate = UInt32(sampleRate * 2)
        let dataSize = UInt32(n * 2)
        data.append("RIFF".data(using: .ascii)!)
        append32(36 + dataSize)
        data.append("WAVE".data(using: .ascii)!)
        data.append("fmt ".data(using: .ascii)!)
        append32(16)                 // fmt chunk size
        append16(1)                  // PCM
        append16(1)                  // mono
        append32(UInt32(sampleRate))
        append32(byteRate)
        append16(2)                  // block align
        append16(16)                 // bits per sample
        data.append("data".data(using: .ascii)!)
        append32(dataSize)
        for s in samples { append16(UInt16(bitPattern: s)) }
        return data
    }

    #if canImport(CoreGraphics) && canImport(ImageIO)
    /// The image path: a real gemma-4 image request with `wantLogprobs: true`
    /// must go through the generic decode loop and populate every token's
    /// `TokenEvent.logprob`.
    func testImageRequestPopulatesLogprobsViaGenericLoop() async throws {
        let dir = try requireModel()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        XCTAssertTrue(engine.supportsNativeImage,
            "gemma-4-e2b must advertise native image input")

        let topN = 4
        let (stream, _) = engine.generate(
            prompt: "What is the dominant color of this image? Answer with one word.",
            maxTokens: 12, imageData: solidPNG("blue"),
            wantLogprobs: true, topLogprobs: topN)

        var count = 0
        for await event in stream {
            if event.isEnd { break }
            count += 1
            guard let info = event.logprob else {
                XCTFail("token \(event.tokenId) has no logprob info on an image request")
                continue
            }
            XCTAssertEqual(info.topAlternates.count, topN)
            XCTAssertFalse(info.logprob.isNaN)
            XCTAssertLessThanOrEqual(info.logprob, 0.0001)
        }
        XCTAssertGreaterThan(count, 0, "image generation must produce at least one content token")
    }
    #endif

    /// The audio path: a real gemma-4 audio request with `wantLogprobs: true`
    /// must go through the generic decode loop and populate every token's
    /// `TokenEvent.logprob`.
    func testAudioRequestPopulatesLogprobsViaGenericLoop() async throws {
        let dir = try requireModel()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        XCTAssertTrue(engine.canUseNativeAudio,
            "gemma-4-e2b must advertise native audio input")

        let topN = 4
        let (stream, _) = engine.generate(
            prompt: "What do you hear in this audio?",
            maxTokens: 12, audioData: sineWAV(),
            wantLogprobs: true, topLogprobs: topN)

        var count = 0
        for await event in stream {
            if event.isEnd { break }
            count += 1
            guard let info = event.logprob else {
                XCTFail("token \(event.tokenId) has no logprob info on an audio request")
                continue
            }
            XCTAssertEqual(info.topAlternates.count, topN)
            XCTAssertFalse(info.logprob.isNaN)
            XCTAssertLessThanOrEqual(info.logprob, 0.0001)
        }
        XCTAssertGreaterThan(count, 0, "audio generation must produce at least one content token")
    }
}
