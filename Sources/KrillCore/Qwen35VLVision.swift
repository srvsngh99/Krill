import Foundation
import MLX
import MLXNN
import MLXFast

// Native Swift+MLX Qwen3.5-VL vision tower — the image/video encoder for
// Ornith-1.0-9B (`qwen3_5_vision`). Ported from mlx_vlm's shared Qwen3-VL
// tower (`mlx_vlm/models/qwen3_vl/vision.py`, which qwen3_5/vision.py
// subclasses unchanged). Two things make this simpler than the general
// Qwen3-VL tower AND different from Krill's Qwen 2.5-VL tower:
//
//   * DeepStack is DISABLED for Ornith (`vision_config.deepstack_visual_indexes`
//     is forced to `[]` in the reference config), so there are no per-layer
//     deepstack mergers — just the single final PatchMerger.
//   * Attention is FULL over each image (per-image `cu_seqlens` segments); there
//     is no windowed attention (unlike Qwen 2.5-VL). Blocks use `LayerNorm`
//     (not RMSNorm) and a plain `linear_fc1`/`linear_fc2` GELU-tanh MLP
//     (not SwiGLU), and there IS a learnable, bilinearly-interpolated position
//     embedding on top of the rotary — which Qwen 2.5-VL lacks.
//
// Weight subtree: `vision_tower.*` (the loader rewrites `model.visual.*` /
// `model.language_model.visual.*` to this prefix, mirroring mlx_vlm sanitize).
// Parity oracle: `mlx_vlm.models.qwen3_5.vision.VisionModel`.

// MARK: - Config

public struct Qwen35VLVisionConfig: Codable, Sendable {
    public let depth: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHeads: Int
    public let inChannels: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let spatialMergeSize: Int
    public let numPositionEmbeddings: Int
    public let outHiddenSize: Int
    /// Empty for Ornith (DeepStack disabled). Kept for forward-compat so a
    /// future qwen3_vl checkpoint with deepstack can reuse the config decode.
    public let deepstackVisualIndexes: [Int]

    enum CodingKeys: String, CodingKey {
        case depth
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHeads = "num_heads"
        case inChannels = "in_channels"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case numPositionEmbeddings = "num_position_embeddings"
        case outHiddenSize = "out_hidden_size"
        case deepstackVisualIndexes = "deepstack_visual_indexes"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        depth = try c.decodeIfPresent(Int.self, forKey: .depth) ?? 27
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 16
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
        spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
        numPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .numPositionEmbeddings) ?? 2304
        outHiddenSize = try c.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? 4096
        // Ornith forces this empty; tolerate a checkpoint that sets it and
        // just ignore the extra mergers (they are not built here).
        deepstackVisualIndexes = try c.decodeIfPresent([Int].self, forKey: .deepstackVisualIndexes) ?? []
    }

    /// Per-head rotary dimension (`head_dim`).
    public var headDim: Int { hiddenSize / numHeads }
    /// Edge length of the learned position-embedding grid (`sqrt(num_pos_emb)`).
    public var numGridPerSide: Int { Int(Double(numPositionEmbeddings).squareRoot().rounded()) }
}

// MARK: - Rotary (2-axis vision RoPE)

/// `rotate_half`: `[-x2, x1]` over the last axis (two contiguous halves).
@inline(__always)
private func rotateHalfVision(_ x: MLXArray) -> MLXArray {
    let d = x.dim(-1)
    let x1 = x[.ellipsis, 0 ..< (d / 2)]
    let x2 = x[.ellipsis, (d / 2) ..< d]
    return concatenated([-x2, x1], axis: -1)
}

/// Apply vision rotary to `q`/`k` of shape `[L, heads, headDim]`.
/// `freqs` is `[L, headDim/2]`; cos/sin are tiled twice to `[L, headDim]`
/// (matching `apply_rotary_pos_emb_vision`'s `mx.tile(..., (1,1,2))`) and
/// broadcast over the head axis.
@inline(__always)
private func applyRotaryPosEmbVision(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
    let dtype = x.dtype
    let xf = x.asType(.float32)
    let f = tiled(freqs, repetitions: [1, 2]).asType(.float32)     // [L, headDim]
    let cos = MLX.cos(f).expandedDimensions(axis: 1)               // [L, 1, headDim]
    let sin = MLX.sin(f).expandedDimensions(axis: 1)               // [L, 1, headDim]
    return (xf * cos + rotateHalfVision(xf) * sin).asType(dtype)
}

// MARK: - Patch embedding (Conv3d)

/// `patch_embed.proj` is `nn.Conv3d(in, hidden, kernel=[T,ph,pw],
/// stride=same, bias=True)`. MLX weight layout `[O, kT, kH, kW, I]`; the
/// checkpoint ships that shape post-sanitize so no load-time reshape is
/// needed. Each patch is one Conv3d output position, so the whole batch of
/// flattened patches is a single 1×1×1 cross-correlation.
final class Qwen35VLPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv3d
    let patchSize: Int
    let temporalPatchSize: Int
    let inChannels: Int
    let hiddenSize: Int

    init(_ c: Qwen35VLVisionConfig) {
        patchSize = c.patchSize
        temporalPatchSize = c.temporalPatchSize
        inChannels = c.inChannels
        hiddenSize = c.hiddenSize
        _proj = ModuleInfo(
            wrappedValue: Conv3d(
                inputChannels: c.inChannels,
                outputChannels: c.hiddenSize,
                kernelSize: .init((c.temporalPatchSize, c.patchSize, c.patchSize)),
                stride: .init((c.temporalPatchSize, c.patchSize, c.patchSize)),
                bias: true),
            key: "proj")
    }

    /// `hidden_states`: `[numPatches, C*T*ph*pw]` (the flattened patch batch
    /// the preprocessor emits). Reshape to `[numPatches, C, T, ph, pw]`, move
    /// channels last for MLX `NDHWC`, run the single-position Conv3d, and
    /// flatten back to `[numPatches, hidden]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let n = x.dim(0)
        let r = x.reshaped(n, inChannels, temporalPatchSize, patchSize, patchSize)
            .movedAxis(source: 1, destination: 4)                 // [n, T, ph, pw, C]
        let out = proj(r)                                          // [n, 1, 1, 1, hidden]
        return out.reshaped(n, hiddenSize)
    }
}

// MARK: - Patch merger (LayerNorm + fc1 + GELU + fc2)

/// Merges a `spatial_merge_size²` block of vision tokens into one LM token.
/// Ornith uses `use_postshuffle_norm = False`: the `LayerNorm` is over the
/// per-patch `hidden_size` and is applied BEFORE the spatial reshape.
final class Qwen35VLPatchMerger: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_fc1") var linearFc1: Linear
    @ModuleInfo(key: "linear_fc2") var linearFc2: Linear
    let act = GELU()
    let mergedDim: Int

    init(hidden: Int, outHidden: Int, spatialMergeSize: Int, eps: Float = 1e-6) {
        mergedDim = hidden * spatialMergeSize * spatialMergeSize
        _norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hidden, eps: eps), key: "norm")
        _linearFc1 = ModuleInfo(wrappedValue: Linear(mergedDim, mergedDim, bias: true), key: "linear_fc1")
        _linearFc2 = ModuleInfo(wrappedValue: Linear(mergedDim, outHidden, bias: true), key: "linear_fc2")
    }

    /// `x`: `[numPatches, hidden]` in merge-grouped order (the preprocessor
    /// lays 4 consecutive patches out as one merge block). Returns
    /// `[numPatches/merge², outHidden]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = norm(x).reshaped(-1, mergedDim)
        return linearFc2(act(linearFc1(normed)))
    }
}

// MARK: - Vision block

final class Qwen35VLVisionAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(hidden: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hidden / numHeads
        self.scale = 1.0 / Float(headDim).squareRoot()
        _qkv = ModuleInfo(wrappedValue: Linear(hidden, hidden * 3, bias: true), key: "qkv")
        _proj = ModuleInfo(wrappedValue: Linear(hidden, hidden, bias: true), key: "proj")
    }

    /// `x`: `[L, hidden]`. `freqs`: `[L, headDim/2]` vision-rotary table.
    /// `segments`: cumulative segment boundaries (`cu_seqlens`) so attention
    /// stays within one image/frame; SDPA runs once per segment.
    func callAsFunction(_ x: MLXArray, freqs: MLXArray, segments: [Int]) -> MLXArray {
        let L = x.dim(0)
        let qkvOut = qkv(x).reshaped(L, 3, numHeads, headDim)
        var q = qkvOut[0..., 0]                                    // [L, heads, headDim]
        var k = qkvOut[0..., 1]
        let v = qkvOut[0..., 2]
        q = applyRotaryPosEmbVision(q, freqs: freqs)
        k = applyRotaryPosEmbVision(k, freqs: freqs)

        // Per-segment SDPA: transpose to [heads, seg, headDim], attend, and
        // stitch the outputs back in patch order.
        let qh = q.transposed(1, 0, 2)                            // [heads, L, headDim]
        let kh = k.transposed(1, 0, 2)
        let vh = v.transposed(1, 0, 2)
        var outs: [MLXArray] = []
        outs.reserveCapacity(max(1, segments.count - 1))
        for i in 0 ..< (segments.count - 1) {
            let s = segments[i], e = segments[i + 1]
            guard e > s else { continue }
            let qs = qh[0..., s ..< e].expandedDimensions(axis: 0)   // [1, heads, seg, headDim]
            let ks = kh[0..., s ..< e].expandedDimensions(axis: 0)
            let vs = vh[0..., s ..< e].expandedDimensions(axis: 0)
            let att = MLXFast.scaledDotProductAttention(
                queries: qs, keys: ks, values: vs, scale: scale, mask: nil)
            outs.append(att[0])                                      // [heads, seg, headDim]
        }
        let merged = outs.count == 1 ? outs[0] : concatenated(outs, axis: 1)
        let o = merged.transposed(1, 0, 2).reshaped(L, numHeads * headDim)
        return proj(o)
    }
}

final class Qwen35VLVisionMLP: Module {
    @ModuleInfo(key: "linear_fc1") var linearFc1: Linear
    @ModuleInfo(key: "linear_fc2") var linearFc2: Linear
    let act = GELU(approximation: .tanh)

    init(hidden: Int, intermediate: Int) {
        _linearFc1 = ModuleInfo(wrappedValue: Linear(hidden, intermediate, bias: true), key: "linear_fc1")
        _linearFc2 = ModuleInfo(wrappedValue: Linear(intermediate, hidden, bias: true), key: "linear_fc2")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linearFc2(act(linearFc1(x)))
    }
}

final class Qwen35VLVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: Qwen35VLVisionAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen35VLVisionMLP

    init(_ c: Qwen35VLVisionConfig, eps: Float = 1e-6) {
        _norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: eps), key: "norm1")
        _norm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: eps), key: "norm2")
        _attn = ModuleInfo(wrappedValue: Qwen35VLVisionAttention(hidden: c.hiddenSize, numHeads: c.numHeads), key: "attn")
        _mlp = ModuleInfo(wrappedValue: Qwen35VLVisionMLP(hidden: c.hiddenSize, intermediate: c.intermediateSize), key: "mlp")
    }

    func callAsFunction(_ x: MLXArray, freqs: MLXArray, segments: [Int]) -> MLXArray {
        let h = x + attn(norm1(x), freqs: freqs, segments: segments)
        return h + mlp(norm2(h))
    }
}

// MARK: - Vision tower

/// Full Qwen3.5-VL vision tower: Conv3d patch embed + bilinearly-interpolated
/// learnable position embedding + `depth` full-attention blocks (per-image
/// segments) + final PatchMerger to the LM hidden size.
public final class Qwen35VLVisionModel: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: Qwen35VLPatchEmbed
    @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
    @ModuleInfo(key: "blocks") var blocks: [Qwen35VLVisionBlock]
    @ModuleInfo(key: "merger") var merger: Qwen35VLPatchMerger

    let config: Qwen35VLVisionConfig
    let mergeSize: Int
    let numGridPerSide: Int
    let rotaryDim: Int         // headDim / 2

    public init(_ c: Qwen35VLVisionConfig) {
        config = c
        mergeSize = c.spatialMergeSize
        numGridPerSide = c.numGridPerSide
        rotaryDim = c.headDim / 2
        _patchEmbed = ModuleInfo(wrappedValue: Qwen35VLPatchEmbed(c), key: "patch_embed")
        _posEmbed = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: c.numPositionEmbeddings, dimensions: c.hiddenSize),
            key: "pos_embed")
        _blocks = ModuleInfo(wrappedValue: (0 ..< c.depth).map { _ in Qwen35VLVisionBlock(c) }, key: "blocks")
        _merger = ModuleInfo(
            wrappedValue: Qwen35VLPatchMerger(
                hidden: c.hiddenSize, outHidden: c.outHiddenSize, spatialMergeSize: c.spatialMergeSize),
            key: "merger")
    }

    // MARK: Rotary position table (`rot_pos_emb`)

    /// Build the `[L, rotaryDim]` vision-rotary frequency table for the given
    /// grids. Positions are (row, col) within each image at full patch
    /// resolution, laid out in merge-block order (matching the preprocessor's
    /// patch ordering). `h`/`w` halves each get `rotaryDim/2` frequencies.
    func rotPosEmb(_ grids: [(t: Int, h: Int, w: Int)]) -> MLXArray {
        let ms = mergeSize
        let maxHW = grids.map { max($0.h, $0.w) }.max() ?? 1
        // inv_freq[i] = 1 / theta^(2i/dim), theta = 10000, dim = rotaryDim.
        let half = rotaryDim / 2
        let invFreq = (0 ..< half).map { i -> Float in
            1.0 / powf(10_000.0, Float(2 * i) / Float(rotaryDim))
        }
        // freq_table[pos] = pos * inv_freq  -> [maxHW, half]
        var freqTable = [Float](repeating: 0, count: maxHW * half)
        for p in 0 ..< maxHW {
            for i in 0 ..< half { freqTable[p * half + i] = Float(p) * invFreq[i] }
        }

        var hRows: [Float] = []
        var wRows: [Float] = []
        for g in grids {
            let mh = g.h / ms, mw = g.w / ms
            // (row, col) full-res coords in merge-block order, tiled over t.
            var coords: [(Int, Int)] = []
            coords.reserveCapacity(g.h * g.w)
            for br in 0 ..< mh {
                for bc in 0 ..< mw {
                    for ir in 0 ..< ms {
                        for ic in 0 ..< ms {
                            coords.append((br * ms + ir, bc * ms + ic))
                        }
                    }
                }
            }
            for _ in 0 ..< g.t {
                for (r, col) in coords {
                    for i in 0 ..< half { hRows.append(freqTable[r * half + i]) }
                    for i in 0 ..< half { wRows.append(freqTable[col * half + i]) }
                }
            }
        }
        let total = hRows.count / half
        let hArr = MLXArray(hRows).reshaped(total, half)
        let wArr = MLXArray(wRows).reshaped(total, half)
        return concatenated([hArr, wArr], axis: -1)               // [L, rotaryDim]
    }

    // MARK: Interpolated learnable position embedding (`fast_pos_embed_interpolate`)

    /// Bilinearly interpolate the `numGridPerSide²` learned position grid to
    /// each image's `h×w` patch grid, then permute into merge-block order and
    /// concatenate. Returns `[L, hidden]`.
    func fastPosEmbedInterpolate(_ grids: [(t: Int, h: Int, w: Int)]) -> MLXArray {
        let side = numGridPerSide
        // Four corner index lists + bilinear weights, gathered from pos_embed.
        var idx = [[Int32]](repeating: [], count: 4)
        var wgt = [[Float]](repeating: [], count: 4)

        for g in grids {
            let h = g.h, w = g.w
            let hIdx = linspaceHost(0, Float(side - 1), h)
            let wIdx = linspaceHost(0, Float(side - 1), w)
            let hFloor = hIdx.map { Int($0) }
            let wFloor = wIdx.map { Int($0) }
            let hCeil = hFloor.map { min($0 + 1, side - 1) }
            let wCeil = wFloor.map { min($0 + 1, side - 1) }
            let dh = zip(hIdx, hFloor).map { $0 - Float($1) }
            let dw = zip(wIdx, wFloor).map { $0 - Float($1) }

            for r in 0 ..< h {
                let baseH = hFloor[r] * side
                let baseHC = hCeil[r] * side
                for col in 0 ..< w {
                    idx[0].append(Int32(baseH + wFloor[col]))
                    idx[1].append(Int32(baseH + wCeil[col]))
                    idx[2].append(Int32(baseHC + wFloor[col]))
                    idx[3].append(Int32(baseHC + wCeil[col]))
                    wgt[0].append((1 - dh[r]) * (1 - dw[col]))
                    wgt[1].append((1 - dh[r]) * dw[col])
                    wgt[2].append(dh[r] * (1 - dw[col]))
                    wgt[3].append(dh[r] * dw[col])
                }
            }
        }

        // pos_embeds[c] = pos_embed(idx[c]) * weight[c]; sum the 4 corners.
        var summed: MLXArray? = nil
        for c in 0 ..< 4 {
            let e = posEmbed(MLXArray(idx[c]))                    // [totalPatches, hidden]
            let wv = MLXArray(wgt[c]).expandedDimensions(axis: -1)
            let term = e * wv
            summed = summed == nil ? term : summed! + term
        }
        var patchPos = summed!                                    // [totalPatches, hidden]

        // Per-image merge-block permute (t tiling + spatial regroup), matching
        // the patch order the merger and rotary expect.
        let featureDim = patchPos.dim(-1)
        var offset = 0
        var pieces: [MLXArray] = []
        for g in grids {
            let hw = g.h * g.w
            var pe = patchPos[offset ..< (offset + hw)]           // [h*w, hidden]
            offset += hw
            if g.t > 1 { pe = tiled(pe, repetitions: [g.t, 1]) }
            let ms = mergeSize
            pe = pe.reshaped(g.t, g.h / ms, ms, g.w / ms, ms, featureDim)
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped(-1, featureDim)
            pieces.append(pe)
        }
        patchPos = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0)
        return patchPos
    }

    // MARK: Forward

    /// `pixelValues`: `[numPatches, C*T*ph*pw]`. `grids`: one `(t,h,w)` per
    /// image/video (host-known from the preprocessor). Returns the merged
    /// LM-ready image features `[numPatches/merge², outHidden]`.
    public func callAsFunction(_ pixelValues: MLXArray, grids: [(t: Int, h: Int, w: Int)]) -> MLXArray {
        var h = patchEmbed(pixelValues)                           // [L, hidden]
        h = h + fastPosEmbedInterpolate(grids)
        let freqs = rotPosEmb(grids)                              // [L, rotaryDim]

        // cu_seqlens: each temporal frame of each image is one attention
        // segment of length h*w.
        var segments: [Int] = [0]
        for g in grids {
            for _ in 0 ..< g.t { segments.append(segments.last! + g.h * g.w) }
        }

        for block in blocks {
            h = block(h, freqs: freqs, segments: segments)
        }
        return merger(h)
    }
}

/// Host-side `linspace(start, stop, num)` (inclusive), matching `mx.linspace`.
@inline(__always)
private func linspaceHost(_ start: Float, _ stop: Float, _ num: Int) -> [Float] {
    guard num > 1 else { return [start] }
    let step = (stop - start) / Float(num - 1)
    return (0 ..< num).map { start + Float($0) * step }
}
