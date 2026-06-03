import XCTest
import MLX
@testable import KLMCore

/// Deterministic unit tests for the pure mllama preprocessing math: the
/// aspect-ratio tiling, id mapping, and the sparse-to-dense cross-attention
/// mask. These need no model (unlike `MllamaImageServingTests`, which is gated
/// on the parity fixture), so they run in CI and cover the tile/id/mask logic
/// the RAM-blocked real-checkpoint run cannot exercise here. Reference behaviour
/// is `transformers.image_processing_mllama` + `mlx_vlm...processing_mllama`.
final class MllamaProcessingTests: XCTestCase {

    func testSupportedAspectRatiosMatchesReferenceExample() {
        // get_all_supported_aspect_ratios(4) docstring example.
        let got = MllamaProcessing.supportedAspectRatios(maxTiles: 4)
        let expected = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (4, 1)]
        XCTAssertEqual(got.count, expected.count)
        for (a, b) in zip(got, expected) { XCTAssertTrue(a == b, "\(a) != \(b)") }
    }

    func testOptimalTiledCanvasSquareIsOneTile() {
        // A square image at exactly tile size needs a single tile.
        let (h, w) = MllamaProcessing.optimalTiledCanvas(
            imageHeight: 560, imageWidth: 560, maxTiles: 4, tileSize: 560)
        XCTAssertEqual(h, 560)
        XCTAssertEqual(w, 560)
    }

    func testOptimalTiledCanvasWideImagePicksWideCanvas() {
        // 280x1120 (very wide): the smallest >=1 upscale is the (w=2,h=1)
        // arrangement -> canvas (560, 1120) = 1 tile tall, 2 tiles wide.
        let (h, w) = MllamaProcessing.optimalTiledCanvas(
            imageHeight: 280, imageWidth: 1120, maxTiles: 4, tileSize: 560)
        XCTAssertEqual(h, 560, "tilesHeight should be 1")
        XCTAssertEqual(w, 1120, "tilesWidth should be 2")
    }

    func testAspectRatioIdMatchesReferenceLookup() {
        // (tilesHeight=1, tilesWidth=1) -> index 0 of supported -> id 1.
        XCTAssertEqual(
            MllamaProcessing.aspectRatioId(tilesHeight: 1, tilesWidth: 1, maxTiles: 4), 1)
        // (tilesHeight=1, tilesWidth=2) -> (1,2) is index 1 -> id 2.
        XCTAssertEqual(
            MllamaProcessing.aspectRatioId(tilesHeight: 1, tilesWidth: 2, maxTiles: 4), 2)
        // (tilesHeight=2, tilesWidth=1) -> (2,1) is index 4 -> id 5.
        XCTAssertEqual(
            MllamaProcessing.aspectRatioId(tilesHeight: 2, tilesWidth: 1, maxTiles: 4), 5)
    }

    func testCrossAttentionTokenMaskSingleImage() {
        // One <|image|> at position 0 -> attend from 0 to end (-1).
        let m = MllamaProcessing.crossAttentionTokenMask(
            inputIds: [128256, 11, 12], imageTokenId: 128256)
        XCTAssertEqual(m, [[0, -1]])
    }

    func testCrossAttentionTokenMaskTwoImages() {
        // Two images: first attends [0,3), second attends [3, len).
        let tokens = [128256, 11, 12, 128256, 13, 14, 15]
        let m = MllamaProcessing.crossAttentionTokenMask(
            inputIds: tokens, imageTokenId: 128256)
        XCTAssertEqual(m, [[0, 3], [3, 7]])
    }

    func testCrossAttentionTokenMaskNoImage() {
        XCTAssertEqual(
            MllamaProcessing.crossAttentionTokenMask(inputIds: [11, 12], imageTokenId: 128256),
            [])
    }

    func testDenseCrossMaskMarksValidTilesOverSpan() {
        // Single image, full length, 2 valid tiles, maxTiles=4 -> every row
        // gets its first 2 tiles set for image 0.
        let tokenMask = [[0, -1]]
        let dense = MllamaProcessing.denseCrossMask(
            tokenMask: tokenMask, numTiles: [2], maxImages: 1, maxTiles: 4, length: 3)
        XCTAssertEqual(dense.count, 3 * 1 * 4)
        for row in 0 ..< 3 {
            XCTAssertEqual(dense[row * 4 + 0], 1)
            XCTAssertEqual(dense[row * 4 + 1], 1)
            XCTAssertEqual(dense[row * 4 + 2], 0, "tile 2 is padding (only 2 valid)")
            XCTAssertEqual(dense[row * 4 + 3], 0)
        }
    }

    func testPrepareCrossMaskAdditiveValuesAndShape() {
        // 2 rows, 1 image, 2 tiles, 1 vision token. Row 0 attends (dense 1),
        // row 1 attends nothing (dense 0) -> dead row zeroed.
        // dense layout [length, maxImages, maxTiles] = [2,1,2].
        let dense = [/* row0 */ 1, 0, /* row1 */ 0, 0]
        let (cross, fullRow) = MllamaProcessing.prepareCrossMask(
            dense: dense, length: 2, maxImages: 1, maxTiles: 2, numVisionTokens: 1)
        // S = maxImages*maxTiles*numVisionTokens = 2.
        XCTAssertEqual(cross.shape, [1, 1, 2, 2])
        XCTAssertEqual(fullRow.shape, [1, 2, 1])
        eval(cross, fullRow)
        let c = cross.reshaped(4).asArray(Float.self)
        // Row 0: tile0 attended -> 1.0 ; tile1 masked -> -1e9.
        XCTAssertEqual(c[0], 1.0, accuracy: 1e-3)
        XCTAssertLessThan(c[1], -1e8)
        // Row 1: dead row -> all zeroed.
        XCTAssertEqual(c[2], 0.0, accuracy: 1e-6)
        XCTAssertEqual(c[3], 0.0, accuracy: 1e-6)
        let fr = fullRow.reshaped(2).asArray(Float.self)
        XCTAssertEqual(fr[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(fr[1], 0.0, accuracy: 1e-6)
    }
}
