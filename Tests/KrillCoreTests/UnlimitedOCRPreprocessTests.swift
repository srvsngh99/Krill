import XCTest
import MLX
@testable import KrillCore

#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
#endif

/// Guards the Unlimited-OCR image preprocessor's geometry — specifically the
/// row orientation. A vertical flip here makes the model read upside-down text
/// and silently garbles every OCR result (the original serving bug), so a
/// fast, model-free orientation check is worth pinning.
final class UnlimitedOCRPreprocessTests: XCTestCase {
    #if canImport(CoreGraphics) && canImport(ImageIO)
    /// Encode a 2-colour image (black TOP half, white BOTTOM half) to PNG bytes.
    private func topBlackBottomWhitePNG(_ size: Int = 64) -> Data? {
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: size, height: size, bitsPerComponent: 8,
            bytesPerRow: size * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        // CGContext origin is bottom-left: fill the visual TOP (high y) black.
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: size / 2, width: size, height: size / 2))   // top half black
        guard let img = ctx.makeImage() else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return out as Data
    }

    func testPreprocessKeepsTopAtTop() throws {
        guard let png = topBlackBottomWhitePNG() else {
            throw XCTSkip("could not build test PNG")
        }
        let t = try UnlimitedOCRImagePreprocessor.preprocess(png)
        XCTAssertEqual(t.shape, [1, 1024, 1024, 3])
        // Mean brightness of the top 1/4 of rows vs the bottom 1/4. The source's
        // top half is black (-1 after normalize), bottom half white (+1), so a
        // correctly-oriented tensor has topMean << bottomMean. A vertical flip
        // inverts this — exactly the bug that garbled OCR.
        let img = t.reshaped([1024, 1024, 3]).asType(.float32)
        let topMean = MLX.mean(img[0 ..< 256, 0..., 0...]).item(Float.self)
        let bottomMean = MLX.mean(img[768 ..< 1024, 0..., 0...]).item(Float.self)
        XCTAssertLessThan(topMean, bottomMean - 0.5,
            "preprocessor orientation flipped: top(\(topMean)) should be darker than bottom(\(bottomMean))")
    }
    #endif
}
