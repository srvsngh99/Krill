import XCTest
import Foundation
@testable import KLMEngine
#if canImport(CoreGraphics)
import CoreGraphics
#endif
#if canImport(ImageIO)
import ImageIO
#endif

/// WS5 live smoke + oracle-parity gate for the native Qwen 2.5-VL
/// runtime.
///
/// Gated on `KLM_QWEN25VL_MODEL_PATH` pointing at a real
/// `Qwen2.5-VL-*-Instruct` checkpoint directory; skipped when unset
/// (mirrors `Gemma4SmokeTests`). These tests load the real
/// checkpoint through the native Swift+MLX path - no Python bridge -
/// and assert it (a) produces finite, non-empty output, (b) lets the
/// image actually condition the answer, and (c) matches the recorded
/// mlx-vlm oracle rubric (`Fixtures/ws5_oracle_baseline.json`). The
/// oracle was captured from mlx-vlm before the bridge was retired;
/// rubric-equivalence (expected/forbidden substrings), not greedy
/// token equality, is the cross-runtime contract - the same shape
/// WS6 used for Gemma 4.
final class Qwen25VLSmokeTests: XCTestCase {

    // MARK: - Gating

    private func requireModel() throws -> URL {
        guard let path = ProcessInfo.processInfo
            .environment["KLM_QWEN25VL_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KLM_QWEN25VL_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("KLM_QWEN25VL_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path)
    }

    // MARK: - Oracle baseline

    struct OracleBaseline: Decodable {
        struct Entry: Decodable {
            let name: String
            let prompt: String
            /// Solid-color asset to attach: "red" / "green" / nil.
            let color: String?
            let max_tokens: Int
            let expected_any: [String]
            let forbidden: [String]
            let oracle_output: String?
        }
        let entries: [Entry]
    }

    private func loadBaseline(file: StaticString = #filePath) -> OracleBaseline? {
        let here = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
        let url = here.appendingPathComponent("Fixtures/ws5_oracle_baseline.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(OracleBaseline.self, from: data)
    }

    // MARK: - Image fixtures

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
        ctx.setFillColor(CGColor(
            red: rgb.0, green: rgb.1, blue: rgb.2, alpha: 1))
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

    // MARK: - Generation helper

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

    // MARK: - Tests

    #if canImport(CoreGraphics) && canImport(ImageIO)
    func testNativeLoadAndImagePromptIsCoherent() async throws {
        let dir = try requireModel()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        XCTAssertEqual(engine.family, "qwen2_5_vl",
            "the checkpoint must load through the native VL loader")
        XCTAssertTrue(engine.supportsNativeImage,
            "a loaded VL checkpoint must advertise native image input")

        let answer = await generate(
            engine: engine,
            prompt: "What is the dominant color of this image? "
                + "Answer with one word.",
            imageData: solidPNG("red"), maxTokens: 16)
        XCTAssertFalse(answer.isEmpty,
            "native VL generation must produce output")
        XCTAssertTrue(answer.lowercased().contains("red"),
            "a solid red image must yield a red-ish answer; got: \(answer)")
    }

    func testImageConditionsTheAnswer() async throws {
        let dir = try requireModel()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        let prompt = "What is the dominant color of this image? "
            + "Answer with one word."
        let red = await generate(
            engine: engine, prompt: prompt,
            imageData: solidPNG("red"), maxTokens: 16)
        let green = await generate(
            engine: engine, prompt: prompt,
            imageData: solidPNG("green"), maxTokens: 16)
        XCTAssertNotEqual(red.lowercased(), green.lowercased(),
            "different images must condition different answers; "
            + "red=\(red) green=\(green)")
        XCTAssertTrue(green.lowercased().contains("green"),
            "a solid green image must yield a green-ish answer; got: \(green)")
    }

    func testMatchesOracleRubric() async throws {
        let dir = try requireModel()
        guard let baseline = loadBaseline() else {
            throw XCTSkip("ws5_oracle_baseline.json fixture not present")
        }
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()

        for entry in baseline.entries {
            let image = entry.color.map { solidPNG($0) }
            let output = await generate(
                engine: engine, prompt: entry.prompt,
                imageData: image, maxTokens: entry.max_tokens)
            let lower = output.lowercased()
            XCTAssertTrue(
                entry.expected_any.contains { lower.contains($0.lowercased()) },
                "[\(entry.name)] native output must contain one of "
                + "\(entry.expected_any); got: \(output)")
            for bad in entry.forbidden {
                XCTAssertFalse(lower.contains(bad.lowercased()),
                    "[\(entry.name)] native output must not contain "
                    + "'\(bad)'; got: \(output)")
            }
        }
    }
    #endif
}
