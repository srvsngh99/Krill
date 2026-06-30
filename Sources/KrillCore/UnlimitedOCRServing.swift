import Foundation
import MLX

#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

// MARK: - Unlimited-OCR (DeepSeek-OCR) serving helpers

/// Base-view geometry (single 1024 global view, no gundam tiling). The native
/// runtime ships the parity-validated base view: a 16x16 feature grid plus one
/// `image_newline` column per row plus one `view_seperator`, so the LM sees
/// `16 * (16 + 1) + 1 = 273` `<image>` placeholder tokens, spliced with the
/// DeepEncoder features. (`num_queries_base = ceil((1024 / 16) / 4) = 16`.)
public enum UnlimitedOCR {
    public static let imageTokenId = 128815
    public static let baseSize = 1024
    /// `num_queries_base` = ceil((baseSize / patch_size) / downsample_ratio).
    public static let numQueriesBase = 16
    /// 273 placeholder `<image>` tokens for the base view.
    public static let baseImageTokenCount = numQueriesBase * (numQueriesBase + 1) + 1

    /// Build the prompt token ids for a base-view OCR request, mirroring
    /// `modeling_unlimitedocr.infer` with the `plain` conversation template:
    ///   `[BOS] + encode(text_before_<image>) + [img]*273 + encode(text_after)`.
    /// The model card's canonical prompt is `<image>document parsing.` — the
    /// `<image>` placeholder is immediately followed by the instruction with NO
    /// separator, and the instruction lands *after* the image block. A bare user
    /// prompt is therefore rendered as `<image>{prompt}` (no newline) and split
    /// on the placeholder. `encodeNoSpecial` is the model tokenizer's
    /// `encode(_, add_special_tokens: false)`; the reference prepends bos_id=0
    /// once at the front (bos=False/eos=False per piece).
    public static func promptTokens(
        userText: String, encodeNoSpecial: (String) -> [Int]
    ) -> [Int] {
        // If the caller already embedded `<image>`, honor its placement;
        // otherwise prepend the placeholder directly before the instruction.
        let rendered = userText.contains("<image>") ? userText : "<image>" + userText
        let parts = rendered.components(separatedBy: "<image>")
        let before = parts.first ?? ""
        let after = parts.dropFirst().joined(separator: "<image>")
        var toks = [0]                                   // bos_id = 0
        toks += encodeNoSpecial(before)
        toks += Array(repeating: imageTokenId, count: baseImageTokenCount)
        toks += encodeNoSpecial(after)
        return toks
    }

    /// Splice the assembled base-view vision embeddings into `embeds` at the
    /// first contiguous run of `<image>` placeholder positions. `embeds` is
    /// `[1, L, H]`; `vision` is `[nImg, H]` (273 rows). Returns `[1, L, H]`.
    public static func spliceBaseView(
        embeds: MLXArray, vision: MLXArray, tokens: MLXArray
    ) -> MLXArray {
        let L = embeds.dim(1)
        let H = embeds.dim(2)
        let nImg = vision.dim(0)
        let ids = tokens.reshaped([-1]).asArray(Int32.self)
        guard let start = ids.firstIndex(of: Int32(imageTokenId)) else {
            return embeds   // no image block (text-only) — leave embeddings as-is
        }
        let flat = embeds.reshaped(L, H)
        let head = flat[0 ..< start, 0...]
        let tail = flat[(start + nImg) ..< L, 0...]
        return concatenated([head, vision, tail], axis: 0).reshaped(1, L, H)
    }
}

/// Image preprocessing for the native Unlimited-OCR base view, matching
/// `modeling_unlimitedocr`'s `ImageOps.pad(image, (1024,1024), color=127)` +
/// `BasicImageTransform(mean=std=0.5)`: scale to fit within `baseSize x
/// baseSize` preserving aspect ratio, center-paste onto a mid-gray (127)
/// canvas, then normalize to `[-1, 1]` via `(x/255 - 0.5) / 0.5`.
///
/// Returns a **channels-last** `[1, baseSize, baseSize, 3]` float32 tensor —
/// the layout `DeepEncoder(image:)` consumes directly (no transpose).
public enum UnlimitedOCRImagePreprocessor {
    public static func preprocess(
        _ imageData: Data, baseSize: Int = UnlimitedOCR.baseSize
    ) throws -> MLXArray {
        guard !imageData.isEmpty else {
            throw MultimodalPreprocessingError.emptyImageData
        }
        #if canImport(CoreGraphics) && canImport(ImageIO)
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw MultimodalPreprocessingError.emptyImageData
        }
        let origW = cgImage.width, origH = cgImage.height
        guard origW > 0, origH > 0 else {
            throw MultimodalPreprocessingError.emptyImageData
        }

        // Contain-fit: largest scale that keeps both dims within baseSize.
        let scale = min(Float(baseSize) / Float(origW), Float(baseSize) / Float(origH))
        let drawW = max(1, Int((Float(origW) * scale).rounded()))
        let drawH = max(1, Int((Float(origH) * scale).rounded()))
        let offX = (baseSize - drawW) / 2
        let offY = (baseSize - drawH) / 2

        let bytesPerRow = baseSize * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: baseSize, height: baseSize,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw MultimodalPreprocessingError.emptyImageData
        }
        // Mid-gray pad (127/255), matching `color=int(mean*255)=127`.
        context.setFillColor(CGColor(red: 127.0/255, green: 127.0/255, blue: 127.0/255, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: baseSize, height: baseSize))
        context.interpolationQuality = .high
        // CGContext is bottom-up; offY from the top maps to (baseSize-drawH-offY).
        context.draw(cgImage, in: CGRect(x: offX, y: baseSize - drawH - offY,
                                         width: drawW, height: drawH))

        guard let data = context.data else {
            throw MultimodalPreprocessingError.emptyImageData
        }
        let ptr = data.bindMemory(to: UInt8.self, capacity: baseSize * baseSize * 4)

        // Channels-LAST [H, W, 3], normalized to [-1,1]. This CGContext buffer
        // is already top-to-bottom (row 0 = top of the canvas), matching PIL —
        // an extra flip here would invert the page and the model would read
        // upside-down text. `(x/255 - 0.5)/0.5 = x/127.5 - 1`.
        let n = baseSize * baseSize
        var floats = [Float](repeating: 0, count: n * 3)
        for row in 0 ..< baseSize {
            for col in 0 ..< baseSize {
                let s = (row * baseSize + col) * 4
                let d = (row * baseSize + col) * 3
                floats[d]     = Float(ptr[s])     / 127.5 - 1.0
                floats[d + 1] = Float(ptr[s + 1]) / 127.5 - 1.0
                floats[d + 2] = Float(ptr[s + 2]) / 127.5 - 1.0
            }
        }
        return MLXArray(floats, [1, baseSize, baseSize, 3])
        #else
        throw MultimodalPreprocessingError.imagePreprocessingUnavailable
        #endif
    }
}
