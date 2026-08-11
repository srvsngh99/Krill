import Foundation
import MLX
import MLXNN
import MLXFast

#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

// Native Swift+MLX Muse Glimmer perception encoder — the ~1.8B ViT-G/14 tower
// (`muse_glimmer_vision`), its two-stage projector, and the image
// preprocessor. Ported from
// `transformers/models/muse_glimmer/modeling_muse_glimmer.py` +
// `image_processing_muse_glimmer.py`.
//
// Deltas from the Qwen-VL towers already in Krill (`Qwen35VLVision.swift`):
//
//   * The patch embedder is a plain `Linear(T*C*ph*pw -> hidden)`, NOT a
//     Conv3d, plus a learned 32x32 position table resampled to the image grid
//     at merge size 1 (raster order) with HALF-PIXEL centres and ZERO padding
//     — `F.grid_sample(align_corners=False, padding="zeros")`, not
//     `F.interpolate`. See `axisTaps`; this is where the gate found two bugs.
//   * Attention is WINDOWED (448 px = 32x32 patches) on 3 of every 4 layers,
//     full on the rest — read `layer_types`, never a modulo. Tokens are
//     permuted into window order once, attended per window, and un-permuted
//     after the last layer.
//   * Separate q/k/v/proj WITH bias (Qwen fuses qkv).
//   * 2-axis RoPE interleaved as `[freq_w, freq_h, freq_w, freq_h]` over
//     `head_dim/2` frequencies, with 1-BASED positions (`flip(-1) + 1`).
//   * Spatial merge happens at the END via `pixel_shuffle` (channel-major
//     `out[:, d*4 + j] = in[:, j, d]`), not via a PatchMerger MLP.
//   * The patch payload is laid out `(temporal, channel, ph, pw)` — the
//     image processor comments on this explicitly ("Unlike Glm4v, each
//     flattened patch is laid out (temporal, channel), not (channel,
//     temporal)"). Getting this backwards is silent garbage.
//
// PARITY STATUS: parity green vs the transformers reference on a synthetic
// checkpoint, including the windowed path and a non-square grid. The gate
// caught two real bugs in the position resample on its first run — see
// `axisTaps` below and
// `.claude/skills/native-port/families/muse-glimmer.md`.

// MARK: - Config

public struct MuseGlimmerVisionConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let patchSize: Int
    public let patchTemporal: Int
    public let mergeSize: Int
    public let posEmbHeight: Int
    public let posEmbWidth: Int
    public let layerNormEps: Float
    public let hiddenAct: String
    public let ropeTheta: Float
    /// `"window_attention"` / `"full_attention"` per layer. Authoritative —
    /// the shipped 50-layer tower is NOT a clean 3:1 modulo (the last two
    /// layers are `window, full`).
    public let layerTypes: [String]
    public let inChannels: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case patchSize = "patch_size"
        case patchTemporal = "patch_temporal"
        case mergeSize = "merge_size"
        case posEmbHeight = "pos_emb_height"
        case posEmbWidth = "pos_emb_width"
        case layerNormEps = "layer_norm_eps"
        case hiddenAct = "hidden_act"
        case ropeParameters = "rope_parameters"
        case ropeTheta = "rope_theta"
        case layerTypes = "layer_types"
        case inChannels = "in_channels"
    }

    private struct RopeParameters: Decodable {
        let ropeTheta: Float?
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8960
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 50
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        patchTemporal = try c.decodeIfPresent(Int.self, forKey: .patchTemporal) ?? 2
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        posEmbHeight = try c.decodeIfPresent(Int.self, forKey: .posEmbHeight) ?? 32
        posEmbWidth = try c.decodeIfPresent(Int.self, forKey: .posEmbWidth) ?? 32
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        let rp = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        let flatTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = rp?.ropeTheta ?? flatTheta ?? 10_000.0
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: numHiddenLayers)
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
    }

    /// Convenience initializer for tests and the synthetic parity checkpoint.
    public init(
        hiddenSize: Int, intermediateSize: Int, numHiddenLayers: Int,
        numAttentionHeads: Int, patchSize: Int = 14, patchTemporal: Int = 2,
        mergeSize: Int = 2, posEmbHeight: Int = 32, posEmbWidth: Int = 32,
        layerTypes: [String], layerNormEps: Float = 1e-5, ropeTheta: Float = 10_000.0
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.patchSize = patchSize
        self.patchTemporal = patchTemporal
        self.mergeSize = mergeSize
        self.posEmbHeight = posEmbHeight
        self.posEmbWidth = posEmbWidth
        self.layerNormEps = layerNormEps
        self.hiddenAct = "gelu"
        self.ropeTheta = ropeTheta
        self.layerTypes = layerTypes
        self.inChannels = 3
    }

    public var headDim: Int { hiddenSize / numAttentionHeads }
    /// Flattened patch payload: `patch_temporal * C * ph * pw`.
    public var patchDim: Int { patchTemporal * inChannels * patchSize * patchSize }
    /// Window edge in PATCHES: `pos_emb_height * patch_size / patch_size`.
    /// (The reference computes `window_size = pos_emb_height * patch_size`
    /// pixels, then divides by `patch_size` again.)
    public var windowPatches: Int { posEmbHeight }

    public func isFullAttention(layerIdx: Int) -> Bool {
        guard layerIdx < layerTypes.count else { return true }
        return layerTypes[layerIdx] != "window_attention"
    }
}

// MARK: - Host-side grid geometry

/// The grid bookkeeping the tower needs, all pure integer math on the host.
/// Mirrors `transformers/vision_utils.py`'s `get_vision_window_index`,
/// `get_vision_position_ids`, and `get_vision_cu_seqlens` at
/// `spatial_merge_size = 1`.
enum MuseGlimmerVisionGrid {
    /// Per-frame attention segments (`cu_seqlens`): one segment of `h*w` per
    /// temporal frame of each image.
    static func cuSeqlens(_ grids: [(t: Int, h: Int, w: Int)]) -> [Int] {
        var out = [0]
        for g in grids {
            for _ in 0 ..< g.t { out.append(out.last! + g.h * g.w) }
        }
        return out
    }

    /// Raster `(h, w)` position ids per token, repeated across `t`.
    /// `spatial_merge_size == 1` makes the reference's block reshape a no-op.
    static func positionIds(_ grids: [(t: Int, h: Int, w: Int)]) -> (h: [Float], w: [Float]) {
        var hs: [Float] = [], ws: [Float] = []
        for g in grids {
            hs.reserveCapacity(hs.count + g.t * g.h * g.w)
            ws.reserveCapacity(ws.count + g.t * g.h * g.w)
            for _ in 0 ..< g.t {
                for r in 0 ..< g.h {
                    for c in 0 ..< g.w {
                        hs.append(Float(r))
                        ws.append(Float(c))
                    }
                }
            }
        }
        return (hs, ws)
    }

    /// Window reordering. Returns the permutation into window-major order and
    /// the cumulative window boundaries.
    ///
    /// Faithful to the reference, including its quirk that a grid dimension
    /// already divisible by the window edge is padded by a FULL extra window
    /// (`pad = win - dim % win` is never 0). Those all-padding windows have
    /// length 0 and are removed by the trailing `unique_consecutive`, so the
    /// boundaries still come out right — but the intermediate arithmetic must
    /// match or the window count drifts.
    static func windowIndex(
        _ grids: [(t: Int, h: Int, w: Int)], windowPatches win: Int
    ) -> (index: [Int32], cuWindowSeqlens: [Int]) {
        var index: [Int32] = []
        var cu: [Int] = [0]
        var idOffset = 0

        for g in grids {
            let padH = win - g.h % win
            let padW = win - g.w % win
            let numWinH = (g.h + padH) / win
            let numWinW = (g.w + padW) / win

            for t in 0 ..< g.t {
                for wr in 0 ..< numWinH {
                    for wc in 0 ..< numWinW {
                        var count = 0
                        for ir in 0 ..< win {
                            let r = wr * win + ir
                            if r >= g.h { continue }
                            for ic in 0 ..< win {
                                let c = wc * win + ic
                                if c >= g.w { continue }
                                let flat = (t * g.h + r) * g.w + c
                                index.append(Int32(idOffset + flat))
                                count += 1
                            }
                        }
                        cu.append(cu.last! + count)
                    }
                }
            }
            idOffset += g.t * g.h * g.w
        }

        // `unique_consecutive`: drop the zero-length windows the padding quirk
        // introduces (a repeated cumulative boundary).
        var deduped: [Int] = []
        for v in cu where deduped.last != v { deduped.append(v) }
        return (index, deduped)
    }

    /// Per-axis bilinear taps into the `side`-length position table for `n`
    /// target positions. Mirrors `modeling_muse_glimmer.py`'s OWN
    /// `get_vision_bilinear_indices_and_weights`, which the model file
    /// deliberately overrides from `vision_utils.py`:
    ///
    ///     "The fn is equivalent with F.grid_sample(inputs,
    ///      align_corners=False, padding='zeros')"
    ///
    /// Two consequences, BOTH of which are silent-garbage traps:
    ///
    ///  1. **Half-pixel centres**, `src = (i + 0.5) * side / n - 0.5` — NOT the
    ///     `align_corners=True` (`i * (side-1) / (n-1)`) form that the
    ///     `vision_utils` helper of the same name uses. Using the latter is a
    ///     ~6% error on the patch embeddings.
    ///  2. **Zero padding**: a tap whose UNCLAMPED index falls outside
    ///     `[0, side-1]` contributes NOTHING. The index is still clamped (so the
    ///     gather is in range) but its weight is forced to 0, which is why the
    ///     four corner weights do not generally sum to 1 near the border. Keeping
    ///     the clamped tap's weight instead is a ~2.5% error that only shows up
    ///     when the grid is large relative to the position table.
    ///
    /// Returns the clamped taps plus their weights with the validity mask
    /// already applied.
    private static func axisTaps(
        n: Int, side: Int
    ) -> (lo: [Int], hi: [Int], wLo: [Float], wHi: [Float]) {
        var lo = [Int](), hi = [Int](), wLo = [Float](), wHi = [Float]()
        lo.reserveCapacity(n); hi.reserveCapacity(n)
        wLo.reserveCapacity(n); wHi.reserveCapacity(n)
        for i in 0 ..< n {
            let src = (Float(i) + 0.5) * Float(side) / Float(n) - 0.5
            // `floor`, not truncation: src is negative near the left edge.
            let fl = Int(src.rounded(.down))
            let frac = src - Float(fl)
            let ceil = fl + 1
            let loValid = fl >= 0 && fl <= side - 1
            let hiValid = ceil >= 0 && ceil <= side - 1
            lo.append(min(max(fl, 0), side - 1))
            hi.append(min(max(ceil, 0), side - 1))
            wLo.append(loValid ? 1 - frac : 0)
            wHi.append(hiValid ? frac : 0)
        }
        return (lo, hi, wLo, wHi)
    }

    /// Bilinear resample of the `side x side` learned position table onto each
    /// image's `h x w` raster grid. Returns the four corner index lists and
    /// their weights, ready for a weighted embedding gather — a separable outer
    /// product of the per-axis taps, matching the reference's
    /// `indices = h_tap * side + w_tap`, `weights = h_w * w_w`.
    static func bilinearPosIndices(
        _ grids: [(t: Int, h: Int, w: Int)], side: Int
    ) -> (indices: [[Int32]], weights: [[Float]]) {
        var idx = [[Int32]](repeating: [], count: 4)
        var wgt = [[Float]](repeating: [], count: 4)

        for g in grids {
            let hT = axisTaps(n: g.h, side: side)
            let wT = axisTaps(n: g.w, side: side)

            // Repeated across `t`, matching the reference's per-frame repeat.
            for _ in 0 ..< g.t {
                for r in 0 ..< g.h {
                    let baseLo = hT.lo[r] * side
                    let baseHi = hT.hi[r] * side
                    for c in 0 ..< g.w {
                        idx[0].append(Int32(baseLo + wT.lo[c]))
                        idx[1].append(Int32(baseLo + wT.hi[c]))
                        idx[2].append(Int32(baseHi + wT.lo[c]))
                        idx[3].append(Int32(baseHi + wT.hi[c]))
                        wgt[0].append(hT.wLo[r] * wT.wLo[c])
                        wgt[1].append(hT.wLo[r] * wT.wHi[c])
                        wgt[2].append(hT.wHi[r] * wT.wLo[c])
                        wgt[3].append(hT.wHi[r] * wT.wHi[c])
                    }
                }
            }
        }
        return (idx, wgt)
    }
}

// MARK: - Rotary

@inline(__always)
private func rotateHalfVision(_ x: MLXArray) -> MLXArray {
    let d = x.dim(-1)
    return concatenated([-x[.ellipsis, (d / 2) ..< d], x[.ellipsis, 0 ..< (d / 2)]], axis: -1)
}

/// Apply the vision rotary to `[L, heads, headDim]` given `cos`/`sin` of shape
/// `[L, headDim]` (broadcast over the head axis). Reference computes in fp32.
@inline(__always)
private func applyRotaryVision(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let dtype = x.dtype
    let xf = x.asType(.float32)
    let c = cos.expandedDimensions(axis: 1)   // [L, 1, headDim]
    let s = sin.expandedDimensions(axis: 1)
    return (xf * c + rotateHalfVision(xf) * s).asType(dtype)
}

// MARK: - Patch embedder

final class MuseGlimmerVisionPatchEmbedder: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: Embedding

    let side: Int

    init(_ c: MuseGlimmerVisionConfig) {
        side = c.posEmbHeight
        _patchEmbedding = ModuleInfo(
            wrappedValue: Linear(c.patchDim, c.hiddenSize, bias: false), key: "patch_embedding")
        _positionEmbeddingTable = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: c.posEmbHeight * c.posEmbWidth, dimensions: c.hiddenSize),
            key: "position_embedding_table")
    }

    /// `pixelValues`: `[L, patchDim]`. Returns `[L, hidden]`.
    func callAsFunction(_ pixelValues: MLXArray, grids: [(t: Int, h: Int, w: Int)]) -> MLXArray {
        let embeds = patchEmbedding(pixelValues)
        let (idx, wgt) = MuseGlimmerVisionGrid.bilinearPosIndices(grids, side: side)
        var pos: MLXArray? = nil
        for corner in 0 ..< 4 {
            let e = positionEmbeddingTable(MLXArray(idx[corner]))
            let term = e * MLXArray(wgt[corner]).expandedDimensions(axis: -1)
            pos = pos == nil ? term : pos! + term
        }
        guard let pos else { return embeds }
        return embeds + pos.asType(embeds.dtype)
    }
}

// MARK: - Encoder block

final class MuseGlimmerVisionAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(_ c: MuseGlimmerVisionConfig) {
        numHeads = c.numAttentionHeads
        headDim = c.headDim
        scale = 1.0 / Float(c.headDim).squareRoot()
        let d = c.hiddenSize
        _qProj = ModuleInfo(wrappedValue: Linear(d, d, bias: true), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(d, d, bias: true), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: Linear(d, d, bias: true), key: "v_proj")
        _proj = ModuleInfo(wrappedValue: Linear(d, d, bias: true), key: "proj")
    }

    /// `x`: `[L, hidden]`; `segments`: cumulative boundaries (window or
    /// per-image) so attention never crosses a segment.
    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, segments: [Int]
    ) -> MLXArray {
        let L = x.dim(0)
        var q = qProj(x).reshaped(L, numHeads, headDim)
        var k = kProj(x).reshaped(L, numHeads, headDim)
        let v = vProj(x).reshaped(L, numHeads, headDim)

        q = applyRotaryVision(q, cos: cos, sin: sin)
        k = applyRotaryVision(k, cos: cos, sin: sin)

        let qh = q.transposed(1, 0, 2)   // [heads, L, headDim]
        let kh = k.transposed(1, 0, 2)
        let vh = v.transposed(1, 0, 2)

        var outs: [MLXArray] = []
        outs.reserveCapacity(max(1, segments.count - 1))
        for i in 0 ..< (segments.count - 1) {
            let s = segments[i], e = segments[i + 1]
            guard e > s else { continue }
            let att = MLXFast.scaledDotProductAttention(
                queries: qh[0..., s ..< e].expandedDimensions(axis: 0),
                keys: kh[0..., s ..< e].expandedDimensions(axis: 0),
                values: vh[0..., s ..< e].expandedDimensions(axis: 0),
                scale: scale, mask: nil)
            outs.append(att[0])
        }
        let merged = outs.count == 1 ? outs[0] : concatenated(outs, axis: 1)
        return proj(merged.transposed(1, 0, 2).reshaped(L, numHeads * headDim))
    }
}

final class MuseGlimmerVisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    let act = GELU()

    init(dim: Int, hidden: Int) {
        _fc1 = ModuleInfo(wrappedValue: Linear(dim, hidden, bias: true), key: "fc1")
        _fc2 = ModuleInfo(wrappedValue: Linear(hidden, dim, bias: true), key: "fc2")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(act(fc1(x))) }
}

final class MuseGlimmerVisionEncoderLayer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: MuseGlimmerVisionAttention
    @ModuleInfo(key: "mlp") var mlp: MuseGlimmerVisionMLP

    init(_ c: MuseGlimmerVisionConfig) {
        // The reference hardcodes eps 1e-5 on the block norms (only ln_pre /
        // ln_post read `layer_norm_eps`); they are equal in the shipped config.
        _norm1 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: 1e-5), key: "norm1")
        _norm2 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: 1e-5), key: "norm2")
        _attn = ModuleInfo(wrappedValue: MuseGlimmerVisionAttention(c), key: "attn")
        _mlp = ModuleInfo(
            wrappedValue: MuseGlimmerVisionMLP(dim: c.hiddenSize, hidden: c.intermediateSize),
            key: "mlp")
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, segments: [Int]
    ) -> MLXArray {
        let h = x + attn(norm1(x), cos: cos, sin: sin, segments: segments)
        return h + mlp(norm2(h))
    }
}

// MARK: - Vision tower

public final class MuseGlimmerVisionModel: Module {
    @ModuleInfo(key: "patch_embedder") var patchEmbedder: MuseGlimmerVisionPatchEmbedder
    @ModuleInfo(key: "ln_pre") var lnPre: LayerNorm
    @ModuleInfo(key: "layers") var layers: [MuseGlimmerVisionEncoderLayer]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm

    let config: MuseGlimmerVisionConfig

    public init(_ c: MuseGlimmerVisionConfig) {
        config = c
        _patchEmbedder = ModuleInfo(
            wrappedValue: MuseGlimmerVisionPatchEmbedder(c), key: "patch_embedder")
        _lnPre = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps), key: "ln_pre")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< c.numHiddenLayers).map { _ in MuseGlimmerVisionEncoderLayer(c) },
            key: "layers")
        _lnPost = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps), key: "ln_post")
    }

    /// Build the `[L, headDim]` cos/sin tables. Frequencies are computed over
    /// `spatial_dim = head_dim / 2` and interleaved `[w, h, w, h]`; positions
    /// are 1-based (`flip(-1) + 1` in the reference).
    func rotaryTables(hPos: [Float], wPos: [Float]) -> (cos: MLXArray, sin: MLXArray) {
        let spatialDim = config.headDim / 2
        let half = spatialDim / 2
        let invFreq = (0 ..< half).map { i -> Float in
            1.0 / powf(config.ropeTheta, Float(2 * i) / Float(spatialDim))
        }
        let inv = MLXArray(invFreq)                                   // [half]
        // `+ 1`: the reference's 1-based position offset.
        let h = (MLXArray(hPos) + MLXArray(Float(1.0))).expandedDimensions(axis: -1)
        let w = (MLXArray(wPos) + MLXArray(Float(1.0))).expandedDimensions(axis: -1)
        let freqH = h * inv                                           // [L, half]
        let freqW = w * inv
        let freq = concatenated([freqW, freqH, freqW, freqH], axis: -1)  // [L, headDim]
        return (MLX.cos(freq), MLX.sin(freq))
    }

    /// `pixel_shuffle`: merge each `merge x merge` block of patches into one
    /// token of `hidden * merge²`, channel-major.
    func pixelShuffle(_ x: MLXArray, grids: [(t: Int, h: Int, w: Int)]) -> MLXArray {
        let f = config.mergeSize
        let dim = x.dim(-1)
        var offset = 0
        var pieces: [MLXArray] = []
        for g in grids {
            let n = g.t * g.h * g.w
            let chunk = x[offset ..< (offset + n)]
            offset += n
            // [t, h/f, f, w/f, f, dim] -> [t, h/f, w/f, f, f, dim]
            let blocks = chunk
                .reshaped(g.t, g.h / f, f, g.w / f, f, dim)
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped(g.t * (g.h / f) * (g.w / f), f * f, dim)
            // -> [n, dim, f*f] -> [n, dim*f*f] (channel-major)
            pieces.append(blocks.transposed(0, 2, 1).reshaped(-1, dim * f * f))
        }
        return pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0)
    }

    /// `pixelValues`: `[L, patchDim]`. Returns `[L / merge², hidden * merge²]`.
    public func callAsFunction(
        _ pixelValues: MLXArray, grids: [(t: Int, h: Int, w: Int)]
    ) -> MLXArray {
        let (windowIdx, cuWindow) = MuseGlimmerVisionGrid.windowIndex(
            grids, windowPatches: config.windowPatches)
        let cuFull = MuseGlimmerVisionGrid.cuSeqlens(grids)
        let (hPos, wPos) = MuseGlimmerVisionGrid.positionIds(grids)

        var h = lnPre(patchEmbedder(pixelValues, grids: grids))

        // Permute into window-major order once; positions travel with it.
        let permutation = MLXArray(windowIdx)
        h = take(h, permutation, axis: 0)
        let permutedH = windowIdx.map { hPos[Int($0)] }
        let permutedW = windowIdx.map { wPos[Int($0)] }
        let (cos, sin) = rotaryTables(hPos: permutedH, wPos: permutedW)

        for (i, layer) in layers.enumerated() {
            let segments = config.isFullAttention(layerIdx: i) ? cuFull : cuWindow
            h = layer(h, cos: cos, sin: sin, segments: segments)
            if (i + 1) % 10 == 0 { MLX.eval(h) }
        }

        // Un-permute, then norm and merge.
        var inverse = [Int32](repeating: 0, count: windowIdx.count)
        for (dst, src) in windowIdx.enumerated() { inverse[Int(src)] = Int32(dst) }
        h = take(h, MLXArray(inverse), axis: 0)
        h = lnPost(h)
        return pixelShuffle(h, grids: grids)
    }
}

// MARK: - Projector

/// `act(fc2(act(fc1(x))))` — note the activation is applied to the OUTPUT too,
/// which is not the usual projector shape.
public final class MuseGlimmerVisionAdapter: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    let act = GELU()

    public init(inDim: Int, hidden: Int, act _: String) {
        _fc1 = ModuleInfo(wrappedValue: Linear(inDim, hidden, bias: false), key: "fc1")
        _fc2 = ModuleInfo(wrappedValue: Linear(hidden, hidden, bias: false), key: "fc2")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        act(fc2(act(fc1(x))))
    }
}

// MARK: - Image preprocessing

/// Muse Glimmer image preprocessing (`MuseGlimmerImageProcessor`).
///
/// Two things differ from every Qwen-VL preprocessor in this repo:
///   * `smart_resize` picks the integer patch grid closest to the input aspect
///     ratio under a TOKEN CAP (`max_image_tokens`, 4096), working in units of
///     `patch_size * merge_size` (28 px) — not Qwen's min/max-pixels rounding.
///   * The flattened patch payload is `(temporal, channel, ph, pw)`, i.e. the
///     still image is duplicated across the temporal axis FIRST, then channel.
public enum MuseGlimmerImagePreprocessor {
    /// `[0.5, 0.5, 0.5]` mean/std -> `[-1, 1]`.
    public static func normalize(_ pixels: MLXArray) -> MLXArray {
        (pixels - MLXArray(Float(0.5))) / MLXArray(Float(0.5))
    }

    /// Returns the resize target in pixels for a `factor`-px grid unit.
    static func smartResize(
        height: Int, width: Int, factor: Int, maxTokens: Int
    ) -> (h: Int, w: Int) {
        var idealH = Double(height) / Double(factor)
        var idealW = Double(width) / Double(factor)
        let ratio = idealH > 0 ? idealW / idealH : 1.0
        if idealH * idealW > Double(maxTokens) {
            idealH = (Double(maxTokens) / ratio).squareRoot()
            idealW = idealH * ratio
        }
        var candidates: [(Int, Int)] = []
        for ph in Set([Int(idealH.rounded(.down)), Int(idealH.rounded(.up))]) {
            for pw in Set([Int(idealW.rounded(.down)), Int(idealW.rounded(.up))]) {
                if ph >= 1 && pw >= 1 && ph * pw <= maxTokens { candidates.append((ph, pw)) }
            }
        }
        if candidates.isEmpty {
            candidates = [(max(1, Int(idealH.rounded())), max(1, Int(idealW.rounded())))]
        }
        let target = Double(height) / Double(width)
        let best = candidates.min {
            abs(Double($0.0) / Double($0.1) - target) < abs(Double($1.0) / Double($1.1) - target)
        }!
        return (best.0 * factor, best.1 * factor)
    }

    /// Decode + resize to the smart-resize target, returning `[H, W, 3]` in
    /// `[0, 1]`.
    ///
    /// NOTE: the reference resamples with LANCZOS + antialias; CoreGraphics
    /// interpolates with its own high-quality filter. Sub-pixel differences in
    /// the resampled image are expected and are NOT a parity failure of the
    /// tower itself — gate the tower on preprocessed tensors, not raw JPEGs.
    public static func decode(_ imageData: Data, vision: MuseGlimmerVisionConfig) throws -> MLXArray {
        guard !imageData.isEmpty else { throw MultimodalPreprocessingError.emptyImageData }
        #if canImport(CoreGraphics) && canImport(ImageIO)
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw MultimodalPreprocessingError.emptyImageData
        }
        let (newH, newW) = smartResize(
            height: cgImage.height, width: cgImage.width,
            factor: vision.patchSize * vision.mergeSize, maxTokens: 4096)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: newW, height: newH, bitsPerComponent: 8,
            bytesPerRow: newW * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw MultimodalPreprocessingError.emptyImageData }
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: newW, height: newH))
        guard let data = context.data else {
            throw MultimodalPreprocessingError.emptyImageData
        }

        let pixelCount = newH * newW
        var floats = [Float](repeating: 0, count: pixelCount * 3)
        let ptr = data.bindMemory(to: UInt8.self, capacity: pixelCount * 4)
        // NO row flip. A CGBitmapContext's backing buffer is stored TOP row
        // first — the bottom-left origin applies to the drawing coordinate
        // system, not to memory layout — so `context.data` row 0 is already the
        // top of the drawn image. An earlier revision flipped here "because
        // CGContext is bottom-up", which inverted every image: colours and
        // shapes survived a flip untouched, so it showed up only as the model
        // confidently reversing every above/below relation.
        // Pinned by `testDecodeKeepsImageTopAtRowZero`.
        for row in 0 ..< newH {
            for col in 0 ..< newW {
                let src = (row * newW + col) * 4
                let dst = (row * newW + col) * 3
                floats[dst] = Float(ptr[src]) / 255.0
                floats[dst + 1] = Float(ptr[src + 1]) / 255.0
                floats[dst + 2] = Float(ptr[src + 2]) / 255.0
            }
        }
        return MLXArray(floats, [newH, newW, 3])
        #else
        throw MultimodalPreprocessingError.imagePreprocessingUnavailable
        #endif
    }

    /// `[H, W, 3]` -> `[gridH*gridW, patchTemporal*C*ph*pw]` in `(t, c, ph, pw)`
    /// order, matching `MuseGlimmerImageProcessor.patchify`.
    public static func patchify(
        _ pixels: MLXArray, vision: MuseGlimmerVisionConfig
    ) -> (patches: MLXArray, gridH: Int, gridW: Int) {
        let H = pixels.dim(0), W = pixels.dim(1), C = pixels.dim(2)
        let ps = vision.patchSize
        precondition(H % ps == 0 && W % ps == 0,
            "image dims must be multiples of patch_size; got \(H)x\(W), patch \(ps)")
        let gridH = H / ps, gridW = W / ps

        // [H, W, C] -> [gridH, ph, gridW, pw, C] -> [gridH, gridW, C, ph, pw]
        let blocks = pixels
            .reshaped(gridH, ps, gridW, ps, C)
            .transposed(0, 2, 4, 1, 3)
            .reshaped(gridH * gridW, 1, C * ps * ps)
        // Duplicate over the temporal axis FIRST -> (t, c, ph, pw).
        let tiledPatches = tiled(blocks, repetitions: [1, vision.patchTemporal, 1])
        return (
            patches: tiledPatches.reshaped(gridH * gridW, vision.patchDim),
            gridH: gridH, gridW: gridW
        )
    }

    /// Full pipeline: decode -> normalize -> patchify. The returned grid is the
    /// FULL patch grid; the LM placeholder run is
    /// `(gridH / merge) * (gridW / merge)` `<|patch|>` tokens.
    public static func preprocess(
        _ imageData: Data, vision: MuseGlimmerVisionConfig
    ) throws -> (patches: MLXArray, grid: (t: Int, h: Int, w: Int)) {
        let pixels = try decode(imageData, vision: vision)
        let (patches, gridH, gridW) = patchify(normalize(pixels), vision: vision)
        return (patches, (t: 1, h: gridH, w: gridW))
    }
}
