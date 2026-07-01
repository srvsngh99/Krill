import XCTest
import Foundation
@testable import KrillEngine
#if canImport(CoreGraphics)
import CoreGraphics
#endif
#if canImport(ImageIO)
import ImageIO
#endif

/// Live smoke gate for the native Qwen3.5-VL (Ornith) runtime. Gated on
/// `KRILL_ORNITH_MODEL_PATH` pointing at the int4 checkpoint directory; skipped
/// when unset. Loads the real checkpoint through the native Swift+MLX path (no
/// Python bridge) and asserts the whole pipeline — preprocess → native vision
/// tower → image-feature scatter → 3D mRoPE → hybrid GatedDeltaNet/attn decoder
/// → decode loop — produces coherent, image-conditioned output.
final class Qwen35VLSmokeTests: XCTestCase {

    private func requireModel() throws -> URL {
        guard let path = ProcessInfo.processInfo
            .environment["KRILL_ORNITH_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KRILL_ORNITH_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("KRILL_ORNITH_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path)
    }

    #if canImport(CoreGraphics) && canImport(ImageIO)
    private func solidPNG(_ color: String) -> Data {
        let rgb: (CGFloat, CGFloat, CGFloat)
        switch color {
        case "red":   rgb = (0.85, 0.05, 0.05)
        case "green": rgb = (0.05, 0.7, 0.1)
        case "blue":  rgb = (0.05, 0.1, 0.85)
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
        let dest = CGImageDestinationCreateWithData(
            out, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil)
        _ = CGImageDestinationFinalize(dest)
        return out as Data
    }
    #endif

    private func generate(
        engine: InferenceEngine, prompt: String,
        imageData: Data?, maxTokens: Int
    ) async -> String {
        let (stream, _) = engine.generate(
            messages: [["role": "user", "content": prompt]],
            params: .greedy, maxTokens: maxTokens,
            usePrefixCache: false, imageData: imageData)
        var text = ""
        for await event in stream {
            text += event.text
            if event.isEnd { break }
        }
        return text
    }

    #if canImport(CoreGraphics) && canImport(ImageIO)
    func testNativeLoadAndImagePromptIsCoherent() async throws {
        let dir = try requireModel()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        XCTAssertEqual(engine.family, "qwen3_5",
            "the checkpoint must load through the native VL loader")
        XCTAssertTrue(engine.supportsNativeImage,
            "a loaded VL checkpoint must advertise native image input")

        let answer = await generate(
            engine: engine,
            prompt: "What is the dominant color of this image? "
                + "Answer with one word.",
            imageData: solidPNG("red"), maxTokens: 16)
        XCTAssertFalse(answer.isEmpty, "native VL generation must produce output")
        XCTAssertTrue(answer.lowercased().contains("red"),
            "a solid red image must yield a red-ish answer; got: \(answer)")
    }

    /// Caption the real repo logo through the native path and print it (for
    /// eyeball parity against the mlx_vlm oracle, which describes the same
    /// stacked-bars-and-chevron logo). Asserts only non-empty + coherent length.
    func testRealImageCaptionIsCoherent() async throws {
        let dir = try requireModel()
        let assetPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("assets/krill-icon.png")
        guard let imageData = try? Data(contentsOf: assetPath) else {
            throw XCTSkip("assets/krill-icon.png missing")
        }
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        let caption = await generate(
            engine: engine, prompt: "Describe this image in one sentence.",
            imageData: imageData, maxTokens: 48)
        print("=== native Ornith caption ===\n\(caption)\n=============================")
        XCTAssertFalse(caption.isEmpty, "native VL caption must be non-empty")
        XCTAssertGreaterThan(caption.split(separator: " ").count, 4,
            "caption should be a coherent sentence; got: \(caption)")
    }

    func testImageConditionsTheAnswer() async throws {
        let dir = try requireModel()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        let prompt = "What is the dominant color of this image? Answer with one word."
        let red = await generate(
            engine: engine, prompt: prompt, imageData: solidPNG("red"), maxTokens: 16)
        let green = await generate(
            engine: engine, prompt: prompt, imageData: solidPNG("green"), maxTokens: 16)
        XCTAssertNotEqual(red.lowercased(), green.lowercased(),
            "the image must condition the answer; red=\(red) green=\(green)")
        XCTAssertTrue(green.lowercased().contains("green"),
            "a solid green image must yield a green-ish answer; got: \(green)")
    }
    #endif
}
