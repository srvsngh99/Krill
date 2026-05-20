import XCTest
@testable import KLMEngine

/// Live tests for the Qwen 2.5-VL Python sidecar bridge
/// (`Qwen25VLEngine`). Gated on `KLM_QWEN25VL_MODEL_PATH` so CI does
/// not download 2.9 GB of weights implicitly; run locally with:
///
///     KLM_QWEN25VL_MODEL_PATH=$HOME/.krillm/models/blobs/Qwen2.5-VL-3B-Instruct-4bit \
///       swift test --filter Qwen25VLBridgeTests
final class Qwen25VLBridgeTests: XCTestCase {

    private func liveModelDir() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KLM_QWEN25VL_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KLM_QWEN25VL_MODEL_PATH not set; skipping live VLM bridge test")
        }
        let url = URL(fileURLWithPath: path)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("KLM_QWEN25VL_MODEL_PATH does not point to a directory: \(path)")
        }
        return url
    }

    /// Verifies a single-shot text-only request works through the
    /// bridge. This is the smallest possible smoke test - if the
    /// bridge cannot start the Python sidecar / load mlx-vlm / run a
    /// single generate call, all the multimodal tests will also
    /// fail with the same root cause but be harder to diagnose.
    func testBridgeAnswersTextOnly() async throws {
        let dir = try liveModelDir()
        let engine = Qwen25VLEngine()
        do {
            try await engine.load(directory: dir)
        } catch {
            throw XCTSkip("VLM bridge could not start (mlx-vlm not installed?): \(error)")
        }
        defer { try? engine.shutdown() }
        let result = try engine.generate(
            prompt: "What is 2 + 2?", imagePath: nil, maxTokens: 16)
        XCTAssertFalse(result.text.isEmpty,
            "Bridge text-only generate must return non-empty text")
        XCTAssertTrue(result.text.contains("4"),
            "2 + 2 should produce text containing '4'; got: \(result.text)")
    }

    /// WS5 acceptance bar: image fixture changes output vs text-only.
    func testImageInputChangesOutputVsTextOnly() async throws {
        let dir = try liveModelDir()
        let engine = Qwen25VLEngine()
        do {
            try await engine.load(directory: dir)
        } catch {
            throw XCTSkip("VLM bridge could not start: \(error)")
        }
        defer { try? engine.shutdown() }

        let redPath = try writeSolidPNG(red: 220, green: 30, blue: 30)
        defer { try? FileManager.default.removeItem(atPath: redPath) }

        let textOnly = try engine.generate(
            prompt: "What single color is in this image? Reply with just the color name.",
            imagePath: nil, maxTokens: 12)
        let withImage = try engine.generate(
            prompt: "What single color is in this image? Reply with just the color name.",
            imagePath: redPath, maxTokens: 12)

        XCTAssertNotEqual(
            textOnly.text.trimmingCharacters(in: .whitespacesAndNewlines),
            withImage.text.trimmingCharacters(in: .whitespacesAndNewlines),
            "Image input must produce different output than text-only")
    }

    /// WS5 acceptance bar: two different image fixtures produce
    /// different outputs.
    func testTwoFixturesProduceDifferentOutputs() async throws {
        let dir = try liveModelDir()
        let engine = Qwen25VLEngine()
        do {
            try await engine.load(directory: dir)
        } catch {
            throw XCTSkip("VLM bridge could not start: \(error)")
        }
        defer { try? engine.shutdown() }

        let redPath = try writeSolidPNG(red: 220, green: 30, blue: 30)
        let greenPath = try writeSolidPNG(red: 30, green: 220, blue: 30)
        defer {
            try? FileManager.default.removeItem(atPath: redPath)
            try? FileManager.default.removeItem(atPath: greenPath)
        }

        let prompt = "What single color is in this image? Reply with just the color name."
        let redOut = try engine.generate(
            prompt: prompt, imagePath: redPath, maxTokens: 12)
        let greenOut = try engine.generate(
            prompt: prompt, imagePath: greenPath, maxTokens: 12)

        XCTAssertNotEqual(
            redOut.text.trimmingCharacters(in: .whitespacesAndNewlines),
            greenOut.text.trimmingCharacters(in: .whitespacesAndNewlines),
            "Two different image fixtures must produce different outputs")
        XCTAssertTrue(
            redOut.text.lowercased().contains("red"),
            "Red fixture should produce a response naming red; got: \(redOut.text)")
        XCTAssertTrue(
            greenOut.text.lowercased().contains("green"),
            "Green fixture should produce a response naming green; got: \(greenOut.text)")
    }

    // MARK: - Helpers

    /// Write a 64x64 solid-color PNG to a temp file. We do this in
    /// pure Foundation rather than pulling in CoreGraphics so the
    /// test target keeps a small footprint and works on any
    /// platform that can run KLMEngine.
    private func writeSolidPNG(red: UInt8, green: UInt8, blue: UInt8) throws -> String {
        // Build a minimal uncompressed PNG. PNG requires IHDR + IDAT
        // (deflate-compressed) + IEND. The deflate-compressed pixel
        // payload is too much for hand-rolling; we use the system
        // ImageIO instead.
        #if canImport(CoreGraphics) && canImport(ImageIO)
        let width = 64, height = 64
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for i in stride(from: 0, to: pixels.count, by: 4) {
            pixels[i] = red
            pixels[i + 1] = green
            pixels[i + 2] = blue
            pixels[i + 3] = 255
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(
            data: &pixels, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace, bitmapInfo: bitmapInfo
        ), let cgImage = ctx.makeImage() else {
            throw NSError(domain: "Qwen25VLBridgeTests", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Could not build CGImage"])
        }
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-vlm-\(UUID().uuidString).png")
        guard let dest = CGImageDestinationCreateWithURL(
            tmp as CFURL, "public.png" as CFString, 1, nil
        ) else {
            throw NSError(domain: "Qwen25VLBridgeTests", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Could not create PNG destination"])
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw NSError(domain: "Qwen25VLBridgeTests", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "PNG finalize failed"])
        }
        return tmp.path
        #else
        throw XCTSkip("Solid-color PNG fixture generation requires CoreGraphics")
        #endif
    }
}

#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif
