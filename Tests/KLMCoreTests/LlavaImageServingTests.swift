import XCTest
import MLX
@testable import KLMCore
@testable import KLMTokenizer

/// Engine image-serving wiring for the native LLaVA-1.5 runtime (PR #129 landed
/// the model math). These pin the two family-specific seams the generic
/// multimodal path needs: CLIP image preprocessing and the vicuna prompt with
/// the image-token run placed directly.
final class LlavaImageServingTests: XCTestCase {

    #if canImport(CoreGraphics) && canImport(ImageIO)
    /// Encode a solid-color image as PNG bytes.
    private func solidPNG(width: Int, height: Int,
                          r: UInt8, g: UInt8, b: UInt8) -> Data {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(
            red: CGFloat(r) / 255, green: CGFloat(g) / 255,
            blue: CGFloat(b) / 255, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = ctx.makeImage()!
        let out = NSMutableData()
        let dest = CGImageDestinationCreateWithData(
            out, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        return out as Data
    }

    // MARK: - Preprocessing

    func testPreprocessReturnsSquareChannelFirstTensor() throws {
        // A non-square image must be resized (shortest edge -> 336) and
        // center-cropped to a square [1, 3, 336, 336].
        let png = solidPNG(width: 500, height: 280, r: 200, g: 30, b: 30)
        let pixels = try LlavaImagePreprocessor.preprocess(png, imageSize: 336)
        XCTAssertEqual(pixels.shape, [1, 3, 336, 336])
    }

    func testPreprocessNormalizesWithClipStats() throws {
        // A pure-black image (all channels 0) maps to (0 - mean) / std per
        // channel, so each channel plane is a constant equal to that value.
        let png = solidPNG(width: 336, height: 336, r: 0, g: 0, b: 0)
        let pixels = try LlavaImagePreprocessor.preprocess(png, imageSize: 336)
        eval(pixels)
        let arr = pixels.asArray(Float.self)
        let plane = 336 * 336
        for c in 0 ..< 3 {
            let expected = (0 - LlavaImagePreprocessor.imageMean[c])
                / LlavaImagePreprocessor.imageStd[c]
            XCTAssertEqual(arr[c * plane], expected, accuracy: 1e-3,
                "channel \(c) must be normalized with the CLIP mean/std")
        }
        // The R channel constant (mean 0.481, smallest std) is the most
        // negative; B (largest mean offset/std) differs -- the three planes
        // are NOT identical, confirming per-channel normalization.
        XCTAssertNotEqual(arr[0], arr[2 * plane], accuracy: 1e-4)
    }

    func testPreprocessRejectsEmptyData() {
        XCTAssertThrowsError(try LlavaImagePreprocessor.preprocess(Data(), imageSize: 336))
    }
    #endif
}
