import AppKit
import XCTest
@testable import KrillCore

final class ClipboardImageTests: XCTestCase {
    func testPreservesValidPNGData() {
        let png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0])
        XCTAssertEqual(ClipboardImage.normalizedImageData(pngData: png, tiffData: nil), png)
    }

    func testConvertsTIFFPasteboardDataToPNG() throws {
        guard let bitmap = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: 2, pixelsHigh: 3,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
            isPlanar: false, colorSpaceName: .deviceRGB,
            bitmapFormat: [], bytesPerRow: 0, bitsPerPixel: 0
        ), let tiff = bitmap.tiffRepresentation else {
            throw XCTSkip("Could not create a test TIFF bitmap")
        }

        let data = ClipboardImage.normalizedImageData(pngData: nil, tiffData: tiff)
        XCTAssertNotNil(data)
        XCTAssertTrue(data.map(MediaAttachment.isImageData) ?? false)
        XCTAssertEqual(MediaAttachment.imageDimensions(data ?? Data())?.width, 2)
        XCTAssertEqual(MediaAttachment.imageDimensions(data ?? Data())?.height, 3)
    }

    func testRejectsNonImagePasteboardData() {
        XCTAssertNil(ClipboardImage.normalizedImageData(
            pngData: Data("not an image".utf8), tiffData: Data("also not an image".utf8)))
    }
}
