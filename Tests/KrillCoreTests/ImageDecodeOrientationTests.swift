import XCTest
import MLX
@testable import KrillCore

/// One gate for EVERY image decoder in Krill: row 0 of the decoded tensor must
/// be the TOP of the image.
///
/// This seam had no coverage, and it is exactly the kind that hides. A
/// vertically flipped image keeps its shape, its colours, its histogram, and
/// its patch layout — every other vision test still passes. It surfaces only as
/// the model confidently inverting above/below, which reads as "the model is
/// weak at spatial reasoning" rather than as a bug.
///
/// The trap is CoreGraphics: a bitmap context's DRAWING origin is bottom-left,
/// but its backing buffer is stored top row first. Several decoders here
/// flipped rows on readout "because CGContext is bottom-up" and so inverted
/// every image they touched.
///
/// The fixture is a 2x2 PNG authored OUTSIDE CoreGraphics (top row red, bottom
/// row blue), so a flip in a decoder cannot cancel against a flip in the
/// fixture.
final class ImageDecodeOrientationTests: XCTestCase {

    /// 2x2 PNG: top row pure red, bottom row pure blue.
    private var redOverBluePNG: Data {
        get throws {
            try XCTUnwrap(Data(base64Encoded:
                "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEUlEQVR4nGP8zwACjAwMIAYAERICAXrJpWEAAAAASUVORK5CYII="))
        }
    }

    /// Assert a `[..., H, W]` channel-FIRST tensor has a red top and blue
    /// bottom. Sampling the middle column avoids any edge/letterbox artifact.
    private func assertRedTopChannelFirst(
        _ arr: MLXArray, height: Int, width: Int, planeOffset: Int = 0,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        eval(arr)
        let flat = arr.asArray(Float.self)
        let plane = height * width
        let col = width / 2
        func channel(_ c: Int, row: Int) -> Float {
            flat[planeOffset + c * plane + row * width + col]
        }
        // Rows 0 and H-1 can be letterbox padding in decoders that pad; sample
        // just inside the image instead.
        let top = height / 4, bottom = height - 1 - height / 4
        XCTAssertGreaterThan(channel(0, row: top), channel(2, row: top),
            "the TOP of the image must be red — a larger blue channel here means "
            + "the decoder flipped the image vertically",
            file: file, line: line)
        XCTAssertGreaterThan(channel(2, row: bottom), channel(0, row: bottom),
            "the BOTTOM of the image must be blue", file: file, line: line)
    }

    /// Same assertion for a `[H, W, C]` channel-LAST tensor.
    private func assertRedTopChannelLast(
        _ arr: MLXArray, file: StaticString = #filePath, line: UInt = #line
    ) {
        eval(arr)
        let h = arr.dim(arr.ndim - 3), w = arr.dim(arr.ndim - 2), c = arr.dim(arr.ndim - 1)
        let flat = arr.asArray(Float.self)
        let col = w / 2
        func px(_ row: Int) -> (Float, Float) {
            let base = (row * w + col) * c
            return (flat[base], flat[base + 2])
        }
        let (topR, topB) = px(h / 4)
        let (botR, botB) = px(h - 1 - h / 4)
        XCTAssertGreaterThan(topR, topB,
            "the TOP of the image must be red — a larger blue channel here means "
            + "the decoder flipped the image vertically", file: file, line: line)
        XCTAssertGreaterThan(botB, botR,
            "the BOTTOM of the image must be blue", file: file, line: line)
    }

    // MARK: - Gemma 4 / general path

    func testPreprocessImageKeepsTopAtRowZero() throws {
        let out = try preprocessImage(try redOverBluePNG, targetSize: 336)
        // [1, 3, H, W] channel-first.
        let h = out.dim(out.ndim - 2), w = out.dim(out.ndim - 1)
        assertRedTopChannelFirst(out, height: h, width: w)
    }

    // MARK: - LLaVA / CLIP

    func testLlavaPreprocessKeepsTopAtRowZero() throws {
        let out = try LlavaImagePreprocessor.preprocess(try redOverBluePNG, imageSize: 336)
        // [1, 3, 336, 336]; CLIP normalization keeps red > blue on a red pixel.
        assertRedTopChannelFirst(out, height: 336, width: 336)
    }

    // MARK: - Qwen 2.5-VL

    func testQwen25VLDecodeKeepsTopAtRowZero() throws {
        let out = try Qwen25VLImagePreprocessor.decode(
            try redOverBluePNG, patchSize: 14, spatialMergeSize: 2)
        assertRedTopChannelLast(out)
    }

    // MARK: - Llama 3.2 Vision (mllama)

    func testMllamaPreprocessKeepsTopAtRowZero() throws {
        let inputs = try MllamaProcessing.preprocess(
            images: [try redOverBluePNG], tileSize: 224, maxTiles: 1,
            mean: [0, 0, 0], std: [1, 1, 1])
        // A single tile covers the whole image: [.., 3, 224, 224] channel-first.
        assertRedTopChannelFirst(inputs.pixelValues, height: 224, width: 224)
    }

    // MARK: - Decoders that were already correct (regression guards)

    func testLocateAnythingDecodeKeepsTopAtRowZero() throws {
        let out = try LocateAnythingImagePreprocessor.decode(
            try redOverBluePNG, patch: 14, merge: 2, tokenLimit: 1024)
        assertRedTopChannelLast(out)
    }

    /// DeepSeek UnlimitedOCR. Its decoder already reasoned this out correctly
    /// ("an extra flip here would invert the page and the model would read
    /// upside-down text") — this pins that reasoning so nobody "fixes" it into
    /// agreement with the flipped decoders.
    func testUnlimitedOCRPreprocessKeepsTopAtRowZero() throws {
        let out = try UnlimitedOCRImagePreprocessor.preprocess(try redOverBluePNG)
        // [1, S, S, 3] channel-last, normalized to [-1, 1].
        assertRedTopChannelLast(out)
    }

    /// Gemma 4 "unified" (encoder-free) packs patches itself after calling the
    /// shared `preprocessImage`, so a correct decode could still be undone by
    /// the repack. Each packed row is `[p*p*C, x, y]`, which lets the test find
    /// a patch by its GRID POSITION rather than assuming a traversal order.
    func testGemma4UnifiedPackingKeepsTopPatchesAtSmallY() throws {
        let p = 48
        let packed = try preprocessGemma4UnifiedImage(try redOverBluePNG, modelPatchSize: p)
        eval(packed)
        let n = packed.dim(1), stride = packed.dim(2)
        let flat = packed.asArray(Float.self)
        let posBase = stride - 2                      // [..., x, y]

        var maxY: Float = 0
        for i in 0 ..< n { maxY = Swift.max(maxY, flat[i * stride + posBase + 1]) }
        XCTAssertGreaterThan(maxY, 0, "expected a multi-row patch grid")

        /// First pixel (R,B) of the patch nearest the given fractional height.
        func sample(atFractionOfHeight f: Float) throws -> (Float, Float) {
            let wantY = (maxY * f).rounded()
            let i = try XCTUnwrap((0 ..< n).first {
                flat[$0 * stride + posBase + 1] == wantY
            })
            let base = i * stride
            return (flat[base], flat[base + 2])       // pixel 0: R, B
        }
        let (topR, topB) = try sample(atFractionOfHeight: 0.25)
        let (botR, botB) = try sample(atFractionOfHeight: 0.75)
        XCTAssertGreaterThan(topR, topB,
            "patches with a SMALL y must come from the red top of the image")
        XCTAssertGreaterThan(botB, botR,
            "patches with a LARGE y must come from the blue bottom")
    }
}
