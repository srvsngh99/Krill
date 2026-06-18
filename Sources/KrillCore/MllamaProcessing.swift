import Foundation
import MLX

#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

/// Image preprocessing + cross-attention-mask construction for the native
/// Llama-3.2-Vision (mllama) runtime. Ports `transformers`
/// `image_processing_mllama.py` (tile / aspect-ratio canvas) and
/// `mlx_vlm.models.mllama.processing_mllama` + `Model._prepare_cross_attention_mask`
/// (the sparse-to-dense additive cross mask), so a chat request with one or more
/// `<|image|>` tokens drives the gated cross-attention exactly as the reference.
///
/// The vision tower forward is gated by synthetic mlx-vlm logit parity
/// (`MllamaParityTests`); the image-decode + resize geometry here cannot be
/// validated against a real Llama-3.2-11B-Vision checkpoint on this 24GB host
/// (RAM-blocked), so the pure tiling/id/mask math is unit-tested deterministically
/// (`MllamaProcessingTests`) against the reference's worked examples instead.
public enum MllamaProcessing {

    // MARK: - Aspect-ratio tiling (image_processing_mllama.py)

    /// All `(width, height)` tile arrangements with `width*height <= maxTiles`,
    /// width-major. Mirrors `get_all_supported_aspect_ratios`.
    public static func supportedAspectRatios(maxTiles: Int) -> [(Int, Int)] {
        var out: [(Int, Int)] = []
        for w in 1 ... maxTiles {
            for h in 1 ... maxTiles where w * h <= maxTiles {
                out.append((w, h))
            }
        }
        return out
    }

    /// Best `(canvasHeight, canvasWidth)` (in pixels) for an image, chosen from
    /// the supported tile arrangements. Mirrors `get_optimal_tiled_canvas`:
    /// prefer the smallest upscale >= 1; else the largest downscale < 1; ties
    /// broken by smallest canvas area.
    public static func optimalTiledCanvas(
        imageHeight: Int, imageWidth: Int, maxTiles: Int, tileSize: Int
    ) -> (Int, Int) {
        let arrangements = supportedAspectRatios(maxTiles: maxTiles)
        // canvas (h, w) = (heightTiles*tile, widthTiles*tile); arrangement is (w, h).
        let canvases = arrangements.map { (w, h) in (h * tileSize, w * tileSize) }
        let scales: [Double] = canvases.map { (ch, cw) in
            let sh = Double(ch) / Double(imageHeight)
            let sw = Double(cw) / Double(imageWidth)
            return sw > sh ? sh : sw
        }
        let upscales = scales.filter { $0 >= 1.0 }
        let selected: Double
        if !upscales.isEmpty { selected = upscales.min()! }
        else { selected = scales.filter { $0 < 1.0 }.max()! }
        // Among canvases at the selected scale, pick the smallest area.
        var bestIdx = -1
        var bestArea = Int.max
        for i in 0 ..< canvases.count where scales[i] == selected {
            let area = canvases[i].0 * canvases[i].1
            if area < bestArea { bestArea = area; bestIdx = i }
        }
        return canvases[bestIdx]
    }

    /// New `(height, width)` to fit an image into a canvas preserving aspect,
    /// clamped to `[tileSize, canvas]`. Mirrors `get_image_size_fit_to_canvas`.
    public static func imageSizeFitToCanvas(
        imageHeight: Int, imageWidth: Int, canvasHeight: Int, canvasWidth: Int, tileSize: Int
    ) -> (Int, Int) {
        let targetW = min(max(imageWidth, tileSize), canvasWidth)
        let targetH = min(max(imageHeight, tileSize), canvasHeight)
        let scaleH = Double(targetH) / Double(imageHeight)
        let scaleW = Double(targetW) / Double(imageWidth)
        if scaleW < scaleH {
            let newH = min(max(Int((Double(imageHeight) * scaleW).rounded(.down)), 1), targetH)
            return (newH, targetW)
        } else {
            let newW = min(max(Int((Double(imageWidth) * scaleH).rounded(.down)), 1), targetW)
            return (targetH, newW)
        }
    }

    /// The aspect-ratio id for a `(tilesHeight, tilesWidth)` arrangement: the
    /// 1-based index of `(tilesHeight, tilesWidth)` in the supported `(w, h)`
    /// list (0 is the batch-pad id). This reproduces the reference's
    /// `convert_aspect_ratios_to_ids` lookup (which indexes a `(h, w)` tuple into
    /// the `(w, h)` table — a quirk we match so the embedding lookup agrees).
    public static func aspectRatioId(
        tilesHeight: Int, tilesWidth: Int, maxTiles: Int
    ) -> Int {
        let supported = supportedAspectRatios(maxTiles: maxTiles)
        for (i, pair) in supported.enumerated() where pair == (tilesHeight, tilesWidth) {
            return i + 1
        }
        return 0
    }

    // MARK: - Cross-attention token mask (processing_mllama.py)

    /// Per-image `[start, end]` text-token ranges that attend to each image,
    /// derived from `<|image|>` token positions. Mirrors
    /// `get_cross_attention_token_mask` (an `end` of `-1` means "to sequence end").
    public static func crossAttentionTokenMask(
        inputIds: [Int], imageTokenId: Int
    ) -> [[Int]] {
        let locs = inputIds.enumerated().compactMap { $0.element == imageTokenId ? $0.offset : nil }
        if locs.isEmpty { return [] }
        if locs.count == 1 { return [[locs[0], -1]] }
        var masks: [[Int]] = []
        for i in 0 ..< locs.count - 1 { masks.append([locs[i], locs[i + 1]]) }
        masks.append([locs[locs.count - 1], inputIds.count])
        // Consecutive image tokens both attend to the subsequent text.
        var lastEnd = masks[masks.count - 1][1]
        for i in stride(from: masks.count - 1, through: 0, by: -1) {
            if masks[i][0] == masks[i][1] - 1 { masks[i][1] = lastEnd }
            lastEnd = masks[i][1]
        }
        return masks
    }

    /// Dense `[length, maxImages, maxTiles]` 0/1 mask (batch 1), flattened
    /// row-major. Mirrors `convert_sparse_cross_attention_mask_to_dense`.
    public static func denseCrossMask(
        tokenMask: [[Int]], numTiles: [Int], maxImages: Int, maxTiles: Int, length: Int
    ) -> [Int] {
        var out = [Int](repeating: 0, count: length * maxImages * maxTiles)
        for (img, loc) in tokenMask.enumerated() where loc.count == 2 {
            let start = loc[0]
            var end = loc[1] == -1 ? length : min(loc[1], length)
            end = min(end, length)
            let nt = min(numTiles[img], maxTiles)
            var t = start
            while t < end {
                let base = (t * maxImages + img) * maxTiles
                for tile in 0 ..< nt { out[base + tile] = 1 }
                t += 1
            }
        }
        return out
    }

    /// Build the additive cross-attention mask `[1, 1, length, S]` and the
    /// full-text-row mask `[1, length, 1]` (S = maxImages*maxTiles*numVisionTokens),
    /// where attended vision columns carry `1.0`, masked ones `-1e9`, and rows
    /// that attend to no image are zeroed. Mirrors `_prepare_cross_attention_mask`.
    public static func prepareCrossMask(
        dense: [Int], length: Int, maxImages: Int, maxTiles: Int, numVisionTokens: Int
    ) -> (cross: MLXArray, fullRow: MLXArray) {
        let S = maxImages * maxTiles * numVisionTokens
        let big: Float = -1e9
        var crossF = [Float](repeating: 0, count: length * S)
        var fullRow = [Float](repeating: 0, count: length)
        for row in 0 ..< length {
            var anyAttend = false
            for img in 0 ..< maxImages {
                for tile in 0 ..< maxTiles {
                    let attend = dense[(row * maxImages + img) * maxTiles + tile] == 1
                    if attend { anyAttend = true }
                    let val: Float = attend ? 1.0 : big
                    let colBase = (img * maxTiles + tile) * numVisionTokens
                    for p in 0 ..< numVisionTokens { crossF[row * S + colBase + p] = val }
                }
            }
            fullRow[row] = anyAttend ? 1.0 : 0.0
        }
        // Dead rows (attend to nothing) are zeroed entirely (reference `*=`).
        for row in 0 ..< length where fullRow[row] == 0 {
            for c in 0 ..< S { crossF[row * S + c] = 0 }
        }
        return (MLXArray(crossF, [1, 1, length, S]), MLXArray(fullRow, [1, length, 1]))
    }

    // MARK: - Full image preprocessing

    /// Preprocessed vision inputs for the model forward.
    public struct VisionInputs {
        public let pixelValues: MLXArray       // [1, maxImages, maxTiles, 3, tile, tile]
        public let aspectRatioIds: MLXArray    // [1, maxImages] int32
        public let aspectRatioMask: MLXArray   // [1, maxImages, maxTiles] int32
        public let numTiles: [Int]             // actual tile count per image
    }

    /// Preprocess one or more images into the model's `[1, maxImages, maxTiles,
    /// 3, tile, tile]` tensor plus the aspect-ratio ids/mask and per-image tile
    /// counts. `tileSize` is the vision tower's `image_size`; `mean`/`std` come
    /// from the model's `preprocessor_config.json` (OpenAI-CLIP stats for the
    /// real checkpoint). Resampling uses CoreGraphics high-quality interpolation
    /// (the codebase's other vision preprocessors do the same), a close but not
    /// bit-exact stand-in for torchvision bilinear.
    public static func preprocess(
        images: [Data], tileSize: Int, maxTiles: Int, mean: [Float], std: [Float]
    ) throws -> VisionInputs {
        guard !images.isEmpty else { throw MultimodalPreprocessingError.emptyImageData }
        #if canImport(CoreGraphics) && canImport(ImageIO)
        var perImageTiles: [[Float]] = []         // each flat [nTiles, 3, tile, tile]
        var perImageNumTiles: [Int] = []
        var perImageArrangement: [(Int, Int)] = []  // (tilesHeight, tilesWidth)

        for data in images {
            guard !data.isEmpty,
                  let source = CGImageSourceCreateWithData(data as CFData, nil),
                  let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw MultimodalPreprocessingError.emptyImageData
            }
            let origW = cg.width, origH = cg.height
            guard origW > 0, origH > 0 else { throw MultimodalPreprocessingError.emptyImageData }

            let (canvasH, canvasW) = optimalTiledCanvas(
                imageHeight: origH, imageWidth: origW, maxTiles: maxTiles, tileSize: tileSize)
            let tilesH = canvasH / tileSize, tilesW = canvasW / tileSize
            let (newH, newW) = imageSizeFitToCanvas(
                imageHeight: origH, imageWidth: origW,
                canvasHeight: canvasH, canvasWidth: canvasW, tileSize: tileSize)

            // Draw the resized image into the top-left of a zero-padded canvas
            // (CG origin is bottom-left, so the image occupies the high-y rows).
            let bytesPerRow = canvasW * 4
            guard let ctx = CGContext(
                data: nil, width: canvasW, height: canvasH, bitsPerComponent: 8,
                bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                throw MultimodalPreprocessingError.emptyImageData
            }
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: canvasH - newH, width: newW, height: newH))
            guard let buf = ctx.data else { throw MultimodalPreprocessingError.emptyImageData }
            let ptr = buf.bindMemory(to: UInt8.self, capacity: canvasW * canvasH * 4)

            // Split the canvas into row-major tiles, normalizing each pixel.
            let nTiles = tilesH * tilesW
            let plane = tileSize * tileSize
            var tiles = [Float](repeating: 0, count: nTiles * 3 * plane)
            for th in 0 ..< tilesH {
                for tw in 0 ..< tilesW {
                    let tileIdx = th * tilesW + tw
                    let tileBase = tileIdx * 3 * plane
                    for row in 0 ..< tileSize {
                        let canvasRow = th * tileSize + row
                        let srcRow = canvasH - 1 - canvasRow      // flip CG bottom-up
                        for col in 0 ..< tileSize {
                            let canvasCol = tw * tileSize + col
                            let srcIdx = (srcRow * canvasW + canvasCol) * 4
                            let dst = row * tileSize + col
                            let r = Float(ptr[srcIdx]) / 255.0
                            let g = Float(ptr[srcIdx + 1]) / 255.0
                            let b = Float(ptr[srcIdx + 2]) / 255.0
                            tiles[tileBase + dst] = (r - mean[0]) / std[0]
                            tiles[tileBase + plane + dst] = (g - mean[1]) / std[1]
                            tiles[tileBase + 2 * plane + dst] = (b - mean[2]) / std[2]
                        }
                    }
                }
            }
            perImageTiles.append(tiles)
            perImageNumTiles.append(nTiles)
            perImageArrangement.append((tilesH, tilesW))
        }

        // Stack into [1, maxImages, maxTiles, 3, tile, tile], zero-padding the
        // unused image/tile slots.
        let maxImages = images.count
        let plane = tileSize * tileSize
        let perTile = 3 * plane
        var pixels = [Float](repeating: 0, count: maxImages * maxTiles * perTile)
        var idsFlat = [Int32](repeating: 0, count: maxImages)
        var maskFlat = [Int32](repeating: 0, count: maxImages * maxTiles)
        for img in 0 ..< maxImages {
            let nTiles = perImageNumTiles[img]
            let tiles = perImageTiles[img]
            for t in 0 ..< nTiles {
                let dstBase = ((img * maxTiles) + t) * perTile
                let srcBase = t * perTile
                for k in 0 ..< perTile { pixels[dstBase + k] = tiles[srcBase + k] }
                maskFlat[img * maxTiles + t] = 1
            }
            maskFlat[img * maxTiles + 0] = 1   // tile 0 always valid (reference default)
            let (tilesH, tilesW) = perImageArrangement[img]
            idsFlat[img] = Int32(aspectRatioId(
                tilesHeight: tilesH, tilesWidth: tilesW, maxTiles: maxTiles))
        }
        let pixelValues = MLXArray(pixels, [1, maxImages, maxTiles, 3, tileSize, tileSize])
        return VisionInputs(
            pixelValues: pixelValues,
            aspectRatioIds: MLXArray(idsFlat, [1, maxImages]),
            aspectRatioMask: MLXArray(maskFlat, [1, maxImages, maxTiles]),
            numTiles: perImageNumTiles)
        #else
        throw MultimodalPreprocessingError.imagePreprocessingUnavailable
        #endif
    }
}
