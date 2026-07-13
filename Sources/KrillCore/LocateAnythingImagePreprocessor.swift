import Foundation
import MLX
#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

// Image preprocessing for NVIDIA LocateAnything-3B, porting
// `image_processing_locateanything.py` (the Kimi-VL / MoonViT native-resolution
// pipeline). Differences from the Qwen2.5-VL preprocessor: mean/std are a flat
// 0.5 (not CLIP), the size cap is a patch-count limit (`in_token_limit`, not a
// pixel budget), and patches are emitted in plain RASTER order — the 2×2 merge
// is done inside `MoonViTVisionModel.patch_merger`, NOT here (unlike Qwen, which
// pre-groups patches into merge blocks).
//
// Output contract for `MoonViTVisionModel`: `patches` is `[N, C*ph*pw]` with
// each patch flattened channel-major `(c, ph, pw)` and patches in raster order;
// the pre-merge grid is `(gridH, gridW) = (H/patch, W/patch)`. The number of
// merged vision tokens the prompt must reserve is `(gridH*gridW)/(mergeH*mergeW)`.
public enum LocateAnythingImagePreprocessor {
    /// LocateAnything normalizes with mean=std=0.5 → `x*2 - 1`.
    public static func normalize(_ pixels: MLXArray) -> MLXArray {
        pixels * 2.0 - 1.0
    }

    /// Reference `rescale`: if the patch count `(w/p)*(h/p)` exceeds
    /// `tokenLimit`, scale the image down by `sqrt(limit / patchCount)`; then
    /// pad (resize) each side UP to a multiple of `factor = patch*merge` so the
    /// grid divides evenly by the merge kernel. Returns the target `(h, w)`.
    static func targetSize(
        h: Int, w: Int, patch: Int, factor: Int, tokenLimit: Int
    ) -> (h: Int, w: Int) {
        var newW = w, newH = h
        let patchCount = (w / patch) * (h / patch)
        if patchCount > tokenLimit {
            let scale = (Double(tokenLimit) / Double(patchCount)).squareRoot()
            newW = max(1, Int(Double(w) * scale))
            newH = max(1, Int(Double(h) * scale))
        }
        func ceilTo(_ v: Int) -> Int { max(factor, ((v + factor - 1) / factor) * factor) }
        return (ceilTo(newH), ceilTo(newW))
    }

    /// Decode PNG/JPEG data into a normalized-to-`[0,1]` `[H, W, 3]` float32
    /// tensor, resized to the LocateAnything target size. CoreGraphics-backed.
    public static func decode(
        _ imageData: Data, patch: Int, merge: Int, tokenLimit: Int
    ) throws -> MLXArray {
        guard !imageData.isEmpty else { throw MultimodalPreprocessingError.emptyImageData }
        #if canImport(CoreGraphics) && canImport(ImageIO)
        let factor = patch * merge
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw MultimodalPreprocessingError.emptyImageData
        }
        let (newH, newW) = targetSize(
            h: cgImage.height, w: cgImage.width, patch: patch,
            factor: factor, tokenLimit: tokenLimit)
        precondition(newW / patch < 512 && newH / patch < 512,
            "LocateAnything: image grid exceeds the 512 position-embedding bound")

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: newW, height: newH,
            bitsPerComponent: 8, bytesPerRow: newW * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw MultimodalPreprocessingError.emptyImageData
        }
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: newW, height: newH))
        guard let data = context.data else {
            throw MultimodalPreprocessingError.emptyImageData
        }

        // RGBA -> [H, W, 3] in [0, 1]. This CGBitmapContext lays row 0 at the
        // TOP (verified end-to-end: a top-half object grounds to the top half),
        // matching the reference's PIL/torchvision top-origin convention, so
        // read rows directly — do NOT vertically flip (that inverts the Y axis
        // of every predicted box).
        let pixelCount = newH * newW
        var floats = [Float](repeating: 0, count: pixelCount * 3)
        let ptr = data.bindMemory(to: UInt8.self, capacity: pixelCount * 4)
        for row in 0 ..< newH {
            for col in 0 ..< newW {
                let srcIdx = (row * newW + col) * 4
                let dstIdx = (row * newW + col) * 3
                floats[dstIdx] = Float(ptr[srcIdx]) / 255.0
                floats[dstIdx + 1] = Float(ptr[srcIdx + 1]) / 255.0
                floats[dstIdx + 2] = Float(ptr[srcIdx + 2]) / 255.0
            }
        }
        return MLXArray(floats, [newH, newW, 3])
        #else
        throw MultimodalPreprocessingError.imagePreprocessingUnavailable
        #endif
    }

    /// Patchify a normalized `[H, W, 3]` tensor into `[N, 3*patch*patch]` in
    /// raster order, each patch flattened channel-major `(c, ph, pw)` (matching
    /// the reference `patchify` → `[N, C, p, p]` and the parity oracle's
    /// `reshape(N, -1)`).
    static func patchify(_ pixels: MLXArray, patch: Int) -> (MLXArray, Int, Int) {
        let H = pixels.dim(0), W = pixels.dim(1), C = pixels.dim(2)
        let gridH = H / patch, gridW = W / patch
        // [gridH, p, gridW, p, C] -> [gridH, gridW, C, p, p] -> [N, C*p*p]
        let x = pixels.reshaped(gridH, patch, gridW, patch, C)
            .transposed(0, 2, 4, 1, 3)
            .reshaped(gridH * gridW, C * patch * patch)
        return (x, gridH, gridW)
    }

    /// Full pipeline: decode + resize, normalize (mean/std 0.5), patchify.
    /// Returns the flattened patch batch and the pre-merge `(gridH, gridW)`.
    public static func preprocess(
        _ imageData: Data, config: MoonViTVisionConfig, tokenLimit: Int = 4096
    ) throws -> (patches: MLXArray, gridH: Int, gridW: Int) {
        let pixels = try decode(
            imageData, patch: config.patchSize, merge: config.mergeH, tokenLimit: tokenLimit)
        let (patches, gridH, gridW) = patchify(normalize(pixels), patch: config.patchSize)
        return (patches, gridH, gridW)
    }
}
