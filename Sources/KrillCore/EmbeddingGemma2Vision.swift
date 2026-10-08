import Foundation
import MLX
import MLXNN
#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

// MARK: - EmbeddingGemma 2: vision (images and, later, video frames)
//
// The checkpoint's `vision_tower.*` / `embed_vision.*` tensors are the Gemma 4
// vision encoder (`gemma4_vision`: 16 layers, d 768, 12 heads, 2-D axial RoPE,
// factored 2-D position table) with clipping OFF and no standardisation, so
// Krill's existing `VisionEncoder` runs it as is. This file adds only what is
// specific to embeddings:
//
//   * `EG2ImagePreprocessor`: the HF `Gemma4ImageProcessor` recipe
//     (aspect-preserving resize to a patch budget, bicubic with antialiasing,
//     rescale 1/255, NO normalisation, 16x16 patches). Krill's chat
//     `preprocessImage` is a different recipe (longest side 672 + white pad) and
//     is untouched.
//   * `EG2VisionTower`: `vision_tower` + `embed_vision` as ONE module whose
//     parameter keys equal the checkpoint keys, so loading is a strict
//     key-for-key bind (`loadEG2VisionTower`).
//   * an UNPADDED forward: the reference pads every image to 2520 patches and
//     masks the padding out; padding keys never influence real tokens (they are
//     masked and zeroed before pooling), so running only the real patches is the
//     same function and much cheaper (verified by parity).

// MARK: Config

struct EG2VisionConfig: Decodable {
    let hiddenSize: Int
    let intermediateSize: Int
    let numLayers: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let patchSize: Int
    let poolingKernelSize: Int
    let defaultOutputLength: Int
    let positionEmbeddingSize: Int
    let ropeTheta: Float
    let eps: Float
    let useClippedLinears: Bool
    let standardize: Bool

    private enum Root: String, CodingKey { case vision = "vision_config" }
    private enum K: String, CodingKey {
        case hiddenSize = "hidden_size", intermediateSize = "intermediate_size"
        case numLayers = "num_hidden_layers", numHeads = "num_attention_heads"
        case numKVHeads = "num_key_value_heads", headDim = "head_dim", patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size", defaultOutputLength = "default_output_length"
        case positionEmbeddingSize = "position_embedding_size", ropeParameters = "rope_parameters"
        case eps = "rms_norm_eps", useClippedLinears = "use_clipped_linears", standardize
    }
    private struct Rope: Decodable {
        let ropeTheta: Float?
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: Root.self)
        let c = try root.nestedContainer(keyedBy: K.self, forKey: .vision)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numLayers = try c.decode(Int.self, forKey: .numLayers)
        numHeads = try c.decode(Int.self, forKey: .numHeads)
        numKVHeads = try c.decodeIfPresent(Int.self, forKey: .numKVHeads) ?? numHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        defaultOutputLength = try c.decodeIfPresent(Int.self, forKey: .defaultOutputLength) ?? 280
        positionEmbeddingSize = try c.decodeIfPresent(Int.self, forKey: .positionEmbeddingSize) ?? 10240
        ropeTheta = (try c.decodeIfPresent(Rope.self, forKey: .ropeParameters))?.ropeTheta ?? 100
        eps = try c.decodeIfPresent(Float.self, forKey: .eps) ?? 1e-6
        useClippedLinears = try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? false
        standardize = try c.decodeIfPresent(Bool.self, forKey: .standardize) ?? false
    }
}

// MARK: Preprocessor

public enum EG2ImageError: Error, CustomStringConvertible, Equatable {
    case undecodable
    case unsupportedSoftTokenBudget(Int)
    case tooSmall(width: Int, height: Int)
    case platformUnavailable

    public var description: String {
        switch self {
        case .undecodable: return "image could not be decoded (supported: PNG, JPEG, GIF, TIFF, HEIC, BMP, WebP)"
        case .unsupportedSoftTokenBudget(let n): return "max_soft_tokens \(n) is not one of 70, 140, 280, 560, 1120"
        case .tooSmall(let w, let h): return "image \(w)x\(h) is too small to patchify"
        case .platformUnavailable: return "image decoding is not available on this platform"
        }
    }
}

/// A decoded image: 8-bit RGB, row-major `[height][width][3]` (alpha dropped
/// like PIL `convert("RGB")`, colour profiles ignored like PIL).
public struct EG2RGBImage: Sendable {
    public let pixels: [UInt8]
    public let width: Int
    public let height: Int
    public init(pixels: [UInt8], width: Int, height: Int) {
        precondition(pixels.count == width * height * 3)
        self.pixels = pixels; self.width = width; self.height = height
    }
}

/// An image ready for the tower: patches `[numPatches, 3*16*16]` in [0, 1],
/// row-major over the (resized) patch grid, each patch flattened (py, px, c).
public struct EG2PreparedImage: Sendable {
    public let patches: [Float]
    public let gridW: Int
    public let gridH: Int
    public let patchSize: Int
    public let poolingKernel: Int
    public var numPatches: Int { gridW * gridH }
    /// Soft tokens the image occupies in the prompt (patches / k^2).
    public var softTokens: Int { numPatches / (poolingKernel * poolingKernel) }
    public var resizedWidth: Int { gridW * patchSize }
    public var resizedHeight: Int { gridH * patchSize }
}

public enum EG2ImagePreprocessor {
    public static let patchSize = 16
    public static let poolingKernel = 3
    public static let supportedSoftTokenBudgets = [70, 140, 280, 560, 1120]
    /// Default image budget (`processor_config.image_processor.max_soft_tokens`).
    public static let imageSoftTokens = 280
    /// Per-frame video budget (`video_processor.max_soft_tokens`).
    public static let videoFrameSoftTokens = 140

    /// Port of `get_aspect_ratio_preserving_size` (transformers gemma4): the
    /// largest (height, width), each a multiple of `pooling * patch`, whose patch
    /// count is at most `maxPatches` and that keeps the aspect ratio.
    public static func targetSize(height: Int, width: Int, patchSize: Int = 16,
                                  maxPatches: Int, poolingKernel: Int = 3) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else { throw EG2ImageError.tooSmall(width: width, height: height) }
        let totalPx = Double(height * width)
        let targetPx = Double(maxPatches * patchSize * patchSize)
        let factor = (targetPx / totalPx).squareRoot()
        let sideMult = poolingKernel * patchSize
        var th = Int((factor * Double(height) / Double(sideMult)).rounded(.down)) * sideMult
        var tw = Int((factor * Double(width) / Double(sideMult)).rounded(.down)) * sideMult
        if th == 0 && tw == 0 { throw EG2ImageError.tooSmall(width: width, height: height) }
        let maxSide = (maxPatches / (poolingKernel * poolingKernel)) * sideMult
        if th == 0 {
            th = sideMult
            tw = min(Int((Double(width) / Double(height)).rounded(.down)) * sideMult, maxSide)
        } else if tw == 0 {
            tw = sideMult
            th = min(Int((Double(height) / Double(width)).rounded(.down)) * sideMult, maxSide)
        }
        precondition(Double(th * tw) <= targetPx, "resize exceeds the patch budget")
        return (th, tw)
    }

    /// Decode bytes (PNG/JPEG/...) to 8-bit RGB. EXIF orientation is NOT applied
    /// (PIL `Image.open` does not either).
    public static func decode(_ data: Data) throws -> EG2RGBImage {
        #if canImport(CoreGraphics) && canImport(ImageIO)
        guard !data.isEmpty,
              let src = CGImageSourceCreateWithData(data as CFData, nil),
              CGImageSourceGetCount(src) > 0,
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil),
              cg.width > 0, cg.height > 0 else { throw EG2ImageError.undecodable }
        if let raw = rawRGB(cg) { return raw }
        return try drawnRGB(cg)
        #else
        throw EG2ImageError.platformUnavailable
        #endif
    }

    #if canImport(CoreGraphics) && canImport(ImageIO)
    /// Read the decoder's own 8-bit samples (no colour management, so values
    /// equal what PIL sees). Handles gray, gray+alpha, RGB, RGBA in either byte
    /// order; anything else returns nil and takes the drawn path.
    private static func rawRGB(_ cg: CGImage) -> EG2RGBImage? {
        guard cg.bitsPerComponent == 8, let model = cg.colorSpace?.model,
              model == .monochrome || model == .rgb,
              let provider = cg.dataProvider, let cfData = provider.data else { return nil }
        let bpp = cg.bitsPerPixel
        let comps = bpp / 8
        let w = cg.width, h = cg.height, bpr = cg.bytesPerRow
        guard bpp % 8 == 0, bpr >= w * comps, CFDataGetLength(cfData) >= bpr * h else { return nil }
        let alpha = cg.alphaInfo
        let little = cg.bitmapInfo.contains(.byteOrder32Little) && comps == 4
        // Logical channel order within a pixel, then reversed for little-endian words.
        var order: [Character]
        switch (model, comps) {
        case (.monochrome, 1): order = ["g"]
        case (.monochrome, 2): order = (alpha == .first || alpha == .premultipliedFirst) ? ["a", "g"] : ["g", "a"]
        case (.rgb, 3): order = ["r", "g", "b"]
        case (.rgb, 4):
            switch alpha {
            case .first, .premultipliedFirst, .noneSkipFirst: order = ["a", "r", "g", "b"]
            default: order = ["r", "g", "b", "a"]
            }
        default: return nil
        }
        if little { order.reverse() }
        let premultiplied = alpha == .premultipliedFirst || alpha == .premultipliedLast
        let hasAlpha = order.contains("a") && !(alpha == .noneSkipFirst || alpha == .noneSkipLast || alpha == .none)
        let src = CFDataGetBytePtr(cfData)!
        var out = [UInt8](repeating: 0, count: w * h * 3)
        let ri = order.firstIndex(of: "r"), gi = order.firstIndex(of: "g"), bi = order.firstIndex(of: "b")
        let ai = order.firstIndex(of: "a")
        out.withUnsafeMutableBufferPointer { o in
            for y in 0 ..< h {
                let row = src + y * bpr
                for x in 0 ..< w {
                    let p = row + x * comps
                    var r: Int, g: Int, b: Int
                    if let ri, let gi, let bi { r = Int(p[ri]); g = Int(p[gi]); b = Int(p[bi]) }
                    else { let v = Int(p[order.firstIndex(of: "g")!]); r = v; g = v; b = v }
                    if premultiplied, hasAlpha, let ai {
                        let a = Int(p[ai])
                        if a > 0 && a < 255 {
                            r = min(255, (r * 255 + a / 2) / a)
                            g = min(255, (g * 255 + a / 2) / a)
                            b = min(255, (b * 255 + a / 2) / a)
                        }
                    }
                    let k = (y * w + x) * 3
                    o[k] = UInt8(r); o[k + 1] = UInt8(g); o[k + 2] = UInt8(b)
                }
            }
        }
        return EG2RGBImage(pixels: out, width: w, height: h)
    }

    /// Fallback for exotic layouts (16-bit, palette, CMYK, ...): let CoreGraphics
    /// convert to 8-bit RGB. Colour management may shift values slightly.
    private static func drawnRGB(_ cg: CGImage) throws -> EG2RGBImage {
        let w = cg.width, h = cg.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ok = rgba.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(
                data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { throw EG2ImageError.undecodable }
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0 ..< w * h {
            let a = Int(rgba[i * 4 + 3])
            for c in 0 ..< 3 {
                let v = Int(rgba[i * 4 + c])
                out[i * 3 + c] = UInt8(a > 0 && a < 255 ? min(255, (v * 255 + a / 2) / a) : v)
            }
        }
        return EG2RGBImage(pixels: out, width: w, height: h)
    }
    #endif

    // MARK: Resize

    /// PIL-style bicubic (a = -0.5) with antialiasing: the support widens by the
    /// down-scale factor, weights are normalised per output pixel. Horizontal
    /// pass first, 8-bit rounding between passes (as PIL / torchvision uint8).
    /// Measured against the HF processor's torchvision resize: max 2 levels on
    /// < 0.5% of pixels, mean abs error <= 0.006 (of 255).
    static func resizeCoefficients(inSize: Int, outSize: Int) -> (start: [Int], weights: [[Float]]) {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1.0)
        let support = 2.0 * filterScale
        var starts = [Int](repeating: 0, count: outSize)
        var weights = [[Float]](repeating: [], count: outSize)
        func cubic(_ xIn: Double) -> Double {
            let x = abs(xIn), a = -0.5
            if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
            if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
            return 0
        }
        for i in 0 ..< outSize {
            let center = (Double(i) + 0.5) * scale
            let xmin = max(Int(center - support + 0.5), 0)
            let xmax = min(Int(center + support + 0.5), inSize)
            var ws = [Double]()
            ws.reserveCapacity(max(xmax - xmin, 0))
            var sum = 0.0
            for x in xmin ..< max(xmax, xmin) {
                let w = cubic((Double(x) - center + 0.5) / filterScale)
                ws.append(w); sum += w
            }
            starts[i] = xmin
            weights[i] = sum != 0 ? ws.map { Float($0 / sum) } : ws.map { _ in Float(0) }
        }
        return (starts, weights)
    }

    public static func resize(_ img: EG2RGBImage, toWidth tw: Int, height th: Int) -> EG2RGBImage {
        if tw == img.width && th == img.height { return img }
        let sw = img.width, sh = img.height
        let hx = resizeCoefficients(inSize: sw, outSize: tw)
        let vy = resizeCoefficients(inSize: sh, outSize: th)
        // horizontal: [sh][tw][3]
        var mid = [UInt8](repeating: 0, count: sh * tw * 3)
        img.pixels.withUnsafeBufferPointer { src in
            mid.withUnsafeMutableBufferPointer { dst in
                for y in 0 ..< sh {
                    for ox in 0 ..< tw {
                        let st = hx.start[ox]
                        let ws = hx.weights[ox]
                        var r: Float = 0, g: Float = 0, b: Float = 0
                        for (k, w) in ws.enumerated() {
                            let p = ((y * sw) + st + k) * 3
                            r += w * Float(src[p]); g += w * Float(src[p + 1]); b += w * Float(src[p + 2])
                        }
                        let o = (y * tw + ox) * 3
                        dst[o] = clamp8(r); dst[o + 1] = clamp8(g); dst[o + 2] = clamp8(b)
                    }
                }
            }
        }
        // vertical: [th][tw][3]
        var out = [UInt8](repeating: 0, count: th * tw * 3)
        mid.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for oy in 0 ..< th {
                    let st = vy.start[oy]
                    let ws = vy.weights[oy]
                    for x in 0 ..< tw {
                        var r: Float = 0, g: Float = 0, b: Float = 0
                        for (k, w) in ws.enumerated() {
                            let p = (((st + k) * tw) + x) * 3
                            r += w * Float(src[p]); g += w * Float(src[p + 1]); b += w * Float(src[p + 2])
                        }
                        let o = (oy * tw + x) * 3
                        dst[o] = clamp8(r); dst[o + 1] = clamp8(g); dst[o + 2] = clamp8(b)
                    }
                }
            }
        }
        return EG2RGBImage(pixels: out, width: tw, height: th)
    }

    @inline(__always) private static func clamp8(_ v: Float) -> UInt8 {
        let r = (v + 0.5).rounded(.down)
        return r <= 0 ? 0 : (r >= 255 ? 255 : UInt8(r))
    }

    // MARK: Patchify

    /// Soft tokens `img` will occupy, without resizing anything (cheap): lets a
    /// caller size the whole prompt, and reject an over-long one, before doing
    /// any pixel work.
    public static func softTokens(for img: EG2RGBImage,
                                  maxSoftTokens: Int = imageSoftTokens) throws -> Int {
        try softTokens(width: img.width, height: img.height, maxSoftTokens: maxSoftTokens)
    }

    /// Same, from the pixel size alone (a video's frame size is known before any decode).
    public static func softTokens(width: Int, height: Int,
                                  maxSoftTokens: Int = imageSoftTokens) throws -> Int {
        guard supportedSoftTokenBudgets.contains(maxSoftTokens) else {
            throw EG2ImageError.unsupportedSoftTokenBudget(maxSoftTokens)
        }
        let t = try targetSize(height: height, width: width, patchSize: patchSize,
                               maxPatches: maxSoftTokens * poolingKernel * poolingKernel,
                               poolingKernel: poolingKernel)
        return (t.height / patchSize) * (t.width / patchSize) / (poolingKernel * poolingKernel)
    }

    /// Resize to the patch budget and cut into patches.
    public static func prepare(_ img: EG2RGBImage,
                               maxSoftTokens: Int = imageSoftTokens) throws -> EG2PreparedImage {
        guard supportedSoftTokenBudgets.contains(maxSoftTokens) else {
            throw EG2ImageError.unsupportedSoftTokenBudget(maxSoftTokens)
        }
        let t = try targetSize(height: img.height, width: img.width, patchSize: patchSize,
                               maxPatches: maxSoftTokens * poolingKernel * poolingKernel,
                               poolingKernel: poolingKernel)
        let r = resize(img, toWidth: t.width, height: t.height)
        let gw = t.width / patchSize, gh = t.height / patchSize
        let ps = patchSize
        var patches = [Float](repeating: 0, count: gw * gh * ps * ps * 3)
        let inv: Float = 1.0 / 255.0
        r.pixels.withUnsafeBufferPointer { src in
            patches.withUnsafeMutableBufferPointer { dst in
                var o = 0
                for gy in 0 ..< gh {
                    for gx in 0 ..< gw {
                        for py in 0 ..< ps {
                            let rowBase = ((gy * ps + py) * t.width + gx * ps) * 3
                            for k in 0 ..< ps * 3 { dst[o + k] = Float(src[rowBase + k]) * inv }
                            o += ps * 3
                        }
                    }
                }
            }
        }
        return EG2PreparedImage(patches: patches, gridW: gw, gridH: gh,
                                patchSize: ps, poolingKernel: poolingKernel)
    }

    /// Decode + resize + patchify.
    public static func prepare(_ data: Data, maxSoftTokens: Int = imageSoftTokens) throws -> EG2PreparedImage {
        try prepare(decode(data), maxSoftTokens: maxSoftTokens)
    }
}

// MARK: Tower

/// `vision_tower.*` + `embed_vision.*`. Parameter keys equal the checkpoint's
/// keys one for one.
public final class EG2VisionTower: Module {
    @ModuleInfo(key: "vision_tower") var visionTower: VisionEncoder
    @ModuleInfo(key: "embed_vision") var embedVision: MultimodalEmbedder

    let hiddenSize: Int
    public private(set) var computeDtype: DType = .float32

    init(_ cfg: EG2VisionConfig, textHidden: Int) {
        hiddenSize = cfg.hiddenSize
        _visionTower = ModuleInfo(wrappedValue: VisionEncoder(
            hiddenSize: cfg.hiddenSize, intermediateSize: cfg.intermediateSize,
            numLayers: cfg.numLayers, numHeads: cfg.numHeads, numKVHeads: cfg.numKVHeads,
            headDim: cfg.headDim, patchSize: cfg.patchSize, poolingKernelSize: cfg.poolingKernelSize,
            defaultOutputLength: cfg.defaultOutputLength,
            positionEmbeddingSize: cfg.positionEmbeddingSize, ropeTheta: cfg.ropeTheta, eps: cfg.eps),
            key: "vision_tower")
        _embedVision = ModuleInfo(wrappedValue: MultimodalEmbedder(
            embeddingDim: cfg.hiddenSize, textHiddenSize: textHidden, eps: cfg.eps), key: "embed_vision")
    }

    public func setComputeDtype(_ dtype: DType) {
        eg2CastFloatParameters(self, to: dtype)
        computeDtype = dtype
    }

    /// One image -> soft tokens in text space `[softTokens, textHidden]`.
    /// Runs only the real patches (see file header) and pools k x k patches into
    /// one token, scaled by sqrt(hidden) in float32 like the reference pooler.
    public func softTokens(_ image: EG2PreparedImage) -> MLXArray {
        softTokens(patches: image.patches, gridW: image.gridW, gridH: image.gridH)
    }

    func softTokens(patches: [Float], gridW: Int, gridH: Int) -> MLXArray {
        let enc = visionTower
        let P = gridW * gridH
        let k = enc.poolingKernelSize
        let x = MLXArray(patches, [1, P, patches.count / P])
        // Patch embedding: 2 * (x - 0.5) -> Linear.
        // Compute dtype, not `weight.dtype`: a quantized weight is packed uint32.
        var h = enc.patchEmbedder.inputProj((2 * (x - 0.5)).asType(computeDtype))
        // Factored 2-D position table: x-table[x] + y-table[y].
        var xs = [Int32](), ys = [Int32]()
        xs.reserveCapacity(P); ys.reserveCapacity(P)
        for y in 0 ..< gridH { for xx in 0 ..< gridW { xs.append(Int32(xx)); ys.append(Int32(y)) } }
        let table = enc.patchEmbedder.positionEmbeddingTable
        let pos = take(table[0], MLXArray(xs), axis: 0) + take(table[1], MLXArray(ys), axis: 0)
        h = h + pos.reshaped(1, P, -1).asType(h.dtype)
        let positions = MLXArray(zip(xs, ys).flatMap { [$0, $1] }, [1, P, 2])
        h = enc.encoder(h, positions: positions, mask: nil)
        // Pool k x k -> P / k^2 tokens (no padding: every position is real).
        let pooler = VisionPooler(hiddenSize: hiddenSize, defaultOutputLength: P / (k * k))
        let (pooled, _) = pooler(h, patchPositions: positions,
                                 paddingPositions: MLXArray.zeros([1, P]).asType(.bool))
        let soft = embedVision(pooled.asType(computeDtype))  // [1, n, textHidden]
        return soft.reshaped(soft.dim(1), soft.dim(2))
    }
}

// MARK: Loader (strict binding)

public struct EG2VisionLoadReport: Sendable, CustomStringConvertible {
    public let bound: Int
    /// Clip scalars synthesised because `use_clipped_linears` is false and the
    /// checkpoint (correctly) has none; each is a no-op +-1e38 bound.
    public let defaultedClipScalars: Int
    public var description: String {
        "EmbeddingGemma2 vision: bound \(bound) tensors from the checkpoint (strict: every one consumed, "
        + "none missing); \(defaultedClipScalars) disabled clip scalars defaulted (use_clipped_linears=false)"
    }
}

public enum EG2VisionLoadError: Error, CustomStringConvertible {
    case noVisionTower
    case unsupported(String)
    public var description: String {
        switch self {
        case .noVisionTower: return "this checkpoint has no vision tower (no vision_tower.* tensors)"
        case .unsupported(let m): return "unsupported vision tower: \(m)"
        }
    }
}

/// True when the checkpoint holds `vision_tower.*` tensors (cheap header scan
/// via the already-lazy weight dictionary).
public func eg2HasVisionTower(weights: [String: MLXArray]) -> Bool {
    weights.keys.contains { $0.hasPrefix("vision_tower.") }
}

/// Load the vision tower + `embed_vision` with STRICT binding: every
/// `vision_tower.*` / `embed_vision.*` tensor must land on a parameter and every
/// parameter must be covered with the right shape. The only parameters the
/// checkpoint does not carry are the clip scalars, and only when its config says
/// clipping is off; those are defaulted (and counted), never anything else.
public func loadEG2VisionTower(
    directory: URL, weights: [String: MLXArray]? = nil, dtype: DType = .float32
) throws -> (tower: EG2VisionTower, report: EG2VisionLoadReport) {
    let cfgData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
    let vcfg = try JSONDecoder().decode(EG2VisionConfig.self, from: cfgData)
    let tcfg = try JSONDecoder().decode(EmbeddingGemma2Config.self, from: cfgData)
    if vcfg.standardize { throw EG2VisionLoadError.unsupported("standardize=true") }

    let all = try weights ?? loadWeightArrays(from: directory)
    let vision = all.filter { $0.key.hasPrefix("vision_tower.") || $0.key.hasPrefix("embed_vision.") }
    guard vision.keys.contains(where: { $0.hasPrefix("vision_tower.") }) else {
        throw EG2VisionLoadError.noVisionTower
    }
    let tower = EG2VisionTower(vcfg, textHidden: tcfg.hiddenSize)

    var flat: [(String, MLXArray)] = vision.map { ($0.key, $0.value) }
    var defaulted = 0
    if !vcfg.useClippedLinears {
        let have = Set(vision.keys)
        for (k, v) in tower.parameters().flattened() where !have.contains(k) {
            guard k.hasSuffix(".input_min") || k.hasSuffix(".input_max")
                    || k.hasSuffix(".output_min") || k.hasSuffix(".output_max") else { continue }
            flat.append((k, v)); defaulted += 1
        }
    }
    // Quantized checkpoint: swap in the quantized leaves the file ships (and
    // cross-check them against config.json) before the strict bind.
    let quant = try eg2QuantizationConfig(configData: cfgData)
    let qmods = try eg2QuantizedModules(in: all.keys)
    try eg2ValidateQuantization(modules: qmods, config: quant)
    eg2ApplyQuantization(to: tower, prefix: "", modules: qmods, config: quant)
    try tower.update(
        parameters: ModuleParameters.unflattened(flat),
        verify: [.allModelKeysSet, .shapeMismatch, .noUnusedKeys])
    tower.setComputeDtype(dtype)
    eval(tower)
    let report = EG2VisionLoadReport(bound: vision.count, defaultedClipScalars: defaulted)
    FileHandle.standardError.write(Data((report.description + "\n").utf8))
    return (tower, report)
}
