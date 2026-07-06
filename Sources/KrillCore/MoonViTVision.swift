import Foundation
import MLX
import MLXNN
import MLXFast

// Native Swift+MLX MoonViT vision tower — the image encoder for
// NVIDIA LocateAnything-3B (`locateanything`). Ported from the model's custom
// PyTorch reference (`modeling_vit.py`, `MoonVitPretrainedModel` — the Kimi-VL
// MoonViT-SO-400M tower). It is a native-resolution SigLIP-SO400M variant, so
// it shares dims with the Qwen3.5-VL tower (hidden 1152, depth 27, 16 heads,
// intermediate 4304) but differs in four load-bearing ways vs Krill's existing
// Qwen-VL towers:
//
//   * Patch embed is a 2D `Conv2d(3, hidden, k=patch, stride=patch)` applied to
//     already-patchified `[N, C, ph, pw]` input — i.e. a per-patch linear map.
//     We run it as a matmul over the flattened patch (`sanitize` reshapes the
//     checkpoint's `[O,C,ph,pw]` conv weight to `[O, C*ph*pw]`).
//   * A learnable `[initH, initW, hidden]` position grid is **bicubically**
//     interpolated (PyTorch `F.interpolate(mode="bicubic", align_corners=False)`)
//     to each image's `h×w` patch grid and added — not the bilinear scheme the
//     Qwen3.5 tower uses.
//   * Rotary is 2D **complex** RoPE (`Rope2DPosEmb`): adjacent element pairs are
//     rotated, and the per-pair frequency alternates x-axis / y-axis. This is
//     NOT the split-half real RoPE of the Qwen towers.
//   * Attention is fully bidirectional over each image; blocks use fused `wqkv`
//     + `wo`, `LayerNorm` `norm0`/`norm1`, and a GELU-**tanh** `fc0`/`fc1` MLP.
//     The final 2×2 patch merge concatenates the block in (kh,kw) raster order
//     into `hidden*4`, which the top-level `mlp1` connector then projects.
//
// Weight subtree: `vision_model.*`. Parity oracle:
// `tools/verify_locateanything_parity.py` (runs the NVIDIA `modeling_vit.py`).

// MARK: - Config

public struct MoonViTVisionConfig: Codable, Sendable {
    public let patchSize: Int
    public let initPosEmbHeight: Int
    public let initPosEmbWidth: Int
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let mergeKernelSize: [Int]

    enum CodingKeys: String, CodingKey {
        case patchSize = "patch_size"
        case initPosEmbHeight = "init_pos_emb_height"
        case initPosEmbWidth = "init_pos_emb_width"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case mergeKernelSize = "merge_kernel_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        initPosEmbHeight = try c.decodeIfPresent(Int.self, forKey: .initPosEmbHeight) ?? 64
        initPosEmbWidth = try c.decodeIfPresent(Int.self, forKey: .initPosEmbWidth) ?? 64
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 27
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
        mergeKernelSize = try c.decodeIfPresent([Int].self, forKey: .mergeKernelSize) ?? [2, 2]
    }

    public var headDim: Int { hiddenSize / numAttentionHeads }
    public var mergeH: Int { mergeKernelSize[0] }
    public var mergeW: Int { mergeKernelSize[1] }
}

// MARK: - Bicubic resample (host, matches PyTorch align_corners=False, A=-0.75)

private enum Bicubic {
    static func cubic1(_ x: Double, _ a: Double) -> Double {
        ((a + 2) * x - (a + 3)) * x * x + 1
    }
    static func cubic2(_ x: Double, _ a: Double) -> Double {
        (((a * x) - 5 * a) * x + 8 * a) * x - 4 * a
    }
    /// 1D bicubic resample matrix `[dst, src]` (align_corners=False).
    static func matrix(src: Int, dst: Int) -> [Float] {
        let a = -0.75
        var m = [Float](repeating: 0, count: dst * src)
        if src == dst {
            for i in 0 ..< dst { m[i * src + i] = 1 }
            return m
        }
        let scale = Double(src) / Double(dst)
        for d in 0 ..< dst {
            let real = scale * (Double(d) + 0.5) - 0.5
            let inX = Int(floor(real))
            let t = real - Double(inX)
            let coeffs = [
                cubic2(t + 1, a),
                cubic1(t, a),
                cubic1(1 - t, a),
                cubic2(2 - t, a),
            ]
            for k in 0 ..< 4 {
                var s = inX - 1 + k
                s = min(max(s, 0), src - 1)          // replicate-clamp edges
                m[d * src + s] += Float(coeffs[k])
            }
        }
        return m
    }
}

// MARK: - Learnable 2D interpolated position embedding

/// Holds the `[initH, initW, hidden]` learned grid; bicubically resamples it to
/// each image's `[h, w]` patch grid at forward time and returns `[h*w, hidden]`
/// in raster (row-major) order.
final class MoonViTLearnable2DPosEmb: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray   // [initH, initW, hidden]
    let initH: Int
    let initW: Int
    let hidden: Int

    init(initH: Int, initW: Int, hidden: Int) {
        self.initH = initH
        self.initW = initW
        self.hidden = hidden
        _weight = ParameterInfo(wrappedValue: MLXArray.zeros([initH, initW, hidden]), key: "weight")
    }

    /// `grids`: one `(h, w)` per image. Returns `[sum(h*w), hidden]`.
    func callAsFunction(_ grids: [(h: Int, w: Int)]) -> MLXArray {
        var pieces: [MLXArray] = []
        for g in grids {
            if g.h == initH && g.w == initW {
                pieces.append(weight.reshaped(initH * initW, hidden))
                continue
            }
            // Separable bicubic: rows then cols. pos [initH, initW, hidden].
            let rh = MLXArray(Bicubic.matrix(src: initH, dst: g.h)).reshaped(g.h, initH)
            let rw = MLXArray(Bicubic.matrix(src: initW, dst: g.w)).reshaped(g.w, initW)
            // rows: [g.h, initW*hidden]
            var tmp = matmul(rh.asType(weight.dtype), weight.reshaped(initH, initW * hidden))
            tmp = tmp.reshaped(g.h, initW, hidden)
            // cols: out[h,w,:] = sum_sw rw[w,sw] * tmp[h,sw,:]
            let tmpT = tmp.movedAxis(source: 1, destination: 0)       // [initW, g.h, hidden]
            var out = matmul(rw.asType(weight.dtype), tmpT.reshaped(initW, g.h * hidden))
            out = out.reshaped(g.w, g.h, hidden).movedAxis(source: 0, destination: 1)  // [g.h, g.w, hidden]
            pieces.append(out.reshaped(g.h * g.w, hidden))
        }
        return pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0)
    }
}

// MARK: - Patch embed (Conv2d run as a per-patch matmul)

/// `proj` is `Conv2d(C, hidden, k=patch, stride=patch)` on `[N, C, ph, pw]`
/// input — equivalent to a linear map over the flattened patch. `sanitize`
/// reshapes the checkpoint's `[O, C, ph, pw]` weight to `[O, C*ph*pw]`, so this
/// is a plain `Linear`. Input arrives flattened `[N, C*ph*pw]` (C-major).
final class MoonViTPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "pos_emb") var posEmb: MoonViTLearnable2DPosEmb

    init(_ c: MoonViTVisionConfig) {
        let patchDim = 3 * c.patchSize * c.patchSize
        _proj = ModuleInfo(wrappedValue: Linear(patchDim, c.hiddenSize, bias: true), key: "proj")
        _posEmb = ModuleInfo(
            wrappedValue: MoonViTLearnable2DPosEmb(
                initH: c.initPosEmbHeight, initW: c.initPosEmbWidth, hidden: c.hiddenSize),
            key: "pos_emb")
    }

    /// `x`: `[N, C*ph*pw]`. `grids`: `(h,w)` per image. Returns `[N, hidden]`.
    func callAsFunction(_ x: MLXArray, grids: [(h: Int, w: Int)]) -> MLXArray {
        proj(x) + posEmb(grids)
    }
}

// MARK: - 2D complex rotary

/// Apply MoonViT 2D complex RoPE to `x` of shape `[L, heads, headDim]`.
/// `cos`/`sin` are `[L, headDim/2]` host-built tables (see `MoonViTVisionModel
/// .ropeTables`). Adjacent element pairs `(x[2j], x[2j+1])` are rotated by the
/// per-pair angle; pair `j`'s frequency alternates x-axis (`j` even) / y-axis
/// (`j` odd), matching `Rope2DPosEmb` + `apply_rope` (`view_as_complex`).
@inline(__always)
private func applyRope2D(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let dtype = x.dtype
    let L = x.dim(0), heads = x.dim(1), dim = x.dim(2)
    let xf = x.asType(.float32).reshaped(L, heads, dim / 2, 2)
    let even = xf[.ellipsis, 0]                                   // [L, heads, dim/2]
    let odd = xf[.ellipsis, 1]
    let c = cos.expandedDimensions(axis: 1)                       // [L, 1, dim/2]
    let s = sin.expandedDimensions(axis: 1)
    let outEven = even * c - odd * s
    let outOdd = even * s + odd * c
    let stacked = stacked([outEven, outOdd], axis: -1)           // [L, heads, dim/2, 2]
    return stacked.reshaped(L, heads, dim).asType(dtype)
}

// MARK: - Encoder layer

final class MoonViTEncoderLayer: Module {
    @ModuleInfo(key: "norm0") var norm0: LayerNorm
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "mlp") var mlp: MoonViTMLP

    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(_ c: MoonViTVisionConfig, eps: Float = 1e-5) {
        numHeads = c.numAttentionHeads
        headDim = c.headDim
        scale = 1.0 / Float(headDim).squareRoot()
        _norm0 = ModuleInfo(wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: eps), key: "norm0")
        _norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: eps), key: "norm1")
        _wqkv = ModuleInfo(wrappedValue: Linear(c.hiddenSize, c.hiddenSize * 3, bias: true), key: "wqkv")
        _wo = ModuleInfo(wrappedValue: Linear(c.hiddenSize, c.hiddenSize, bias: true), key: "wo")
        _mlp = ModuleInfo(wrappedValue: MoonViTMLP(hidden: c.hiddenSize, intermediate: c.intermediateSize), key: "mlp")
    }

    /// `x`: `[L, hidden]`. `segments`: cumulative per-image boundaries so
    /// attention stays within one image. `cos`/`sin`: `[L, headDim/2]`.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, segments: [Int]) -> MLXArray {
        let residual = x
        let normed = norm0(x)
        let L = normed.dim(0)
        let qkvR = wqkv(normed).reshaped(L, 3, numHeads, headDim)
        var q = qkvR[0..., 0]                                     // [L, heads, headDim]
        var k = qkvR[0..., 1]
        let v = qkvR[0..., 2]
        q = applyRope2D(q, cos: cos, sin: sin)
        k = applyRope2D(k, cos: cos, sin: sin)

        let qh = q.transposed(1, 0, 2)                           // [heads, L, headDim]
        let kh = k.transposed(1, 0, 2)
        let vh = v.transposed(1, 0, 2)
        var outs: [MLXArray] = []
        outs.reserveCapacity(max(1, segments.count - 1))
        for i in 0 ..< (segments.count - 1) {
            let s = segments[i], e = segments[i + 1]
            guard e > s else { continue }
            let qs = qh[0..., s ..< e].expandedDimensions(axis: 0)
            let ks = kh[0..., s ..< e].expandedDimensions(axis: 0)
            let vs = vh[0..., s ..< e].expandedDimensions(axis: 0)
            let att = MLXFast.scaledDotProductAttention(
                queries: qs, keys: ks, values: vs, scale: scale, mask: nil)
            outs.append(att[0])                                  // [heads, seg, headDim]
        }
        let merged = outs.count == 1 ? outs[0] : concatenated(outs, axis: 1)
        let attnOut = wo(merged.transposed(1, 0, 2).reshaped(L, numHeads * headDim))
        let h = residual + attnOut
        return h + mlp(norm1(h))
    }
}

final class MoonViTMLP: Module {
    @ModuleInfo(key: "fc0") var fc0: Linear
    @ModuleInfo(key: "fc1") var fc1: Linear
    let act = GELU(approximation: .tanh)

    init(hidden: Int, intermediate: Int) {
        _fc0 = ModuleInfo(wrappedValue: Linear(hidden, intermediate, bias: true), key: "fc0")
        _fc1 = ModuleInfo(wrappedValue: Linear(intermediate, hidden, bias: true), key: "fc1")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc1(act(fc0(x))) }
}

// MARK: - Vision tower

/// Full MoonViT tower: patch embed (+ interpolated pos-emb) → `depth` complex-
/// rotary full-attention blocks → final LayerNorm → 2×2 patch merge. Output is
/// `[sum(h*w)/merge², hidden*mergeH*mergeW]` — the connector (`mlp1`) input.
public final class MoonViTVisionModel: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: MoonViTPatchEmbed
    @ModuleInfo(key: "encoder") var encoder: MoonViTEncoder

    let config: MoonViTVisionConfig

    public init(_ c: MoonViTVisionConfig) {
        config = c
        _patchEmbed = ModuleInfo(wrappedValue: MoonViTPatchEmbed(c), key: "patch_embed")
        _encoder = ModuleInfo(wrappedValue: MoonViTEncoder(c), key: "encoder")
    }

    /// Reshape the checkpoint's `patch_embed.proj.weight` `[O,C,ph,pw]` conv
    /// kernel to the `[O, C*ph*pw]` matrix this tower runs as a Linear. Keyed on
    /// the (possibly prefixed) suffix so it works for the fixture
    /// (`vision_model.patch_embed.proj.weight`) and the real loader alike.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = weights
        for (k, v) in weights where k.hasSuffix("patch_embed.proj.weight") && v.ndim == 4 {
            out[k] = v.reshaped(v.dim(0), v.dim(1) * v.dim(2) * v.dim(3))
        }
        return out
    }

    /// Host-build the `[L, headDim/2]` cos/sin rotary tables for `grids` in
    /// raster order. Pair `j` uses freq `1/theta^(4*(j/2)/headDim)` on the
    /// x-axis (col) if `j` even, y-axis (row) if `j` odd.
    func ropeTables(_ grids: [(h: Int, w: Int)]) -> (MLXArray, MLXArray) {
        let dim = config.headDim
        let half = dim / 2
        let theta = 10_000.0
        var freq = [Double](repeating: 0, count: half)
        for j in 0 ..< half {
            let freqIdx = j / 2
            freq[j] = 1.0 / pow(theta, Double(4 * freqIdx) / Double(dim))
        }
        var cosv: [Float] = []
        var sinv: [Float] = []
        for g in grids {
            for row in 0 ..< g.h {
                for col in 0 ..< g.w {
                    for j in 0 ..< half {
                        let pos = (j % 2 == 0) ? Double(col) : Double(row)   // even=x(col), odd=y(row)
                        let angle = pos * freq[j]
                        cosv.append(Float(Foundation.cos(angle)))
                        sinv.append(Float(Foundation.sin(angle)))
                    }
                }
            }
        }
        let L = cosv.count / half
        return (MLXArray(cosv).reshaped(L, half), MLXArray(sinv).reshaped(L, half))
    }

    /// `pixelValues`: `[N, C*ph*pw]` flattened patches (raster order per image).
    /// `grids`: `(h,w)` per image. Returns the pre-connector merged features.
    public func callAsFunction(_ pixelValues: MLXArray, grids: [(h: Int, w: Int)]) -> MLXArray {
        var h = patchEmbed(pixelValues, grids: grids)            // [L, hidden]
        let (cos, sin) = ropeTables(grids)

        var segments: [Int] = [0]
        for g in grids { segments.append(segments.last! + g.h * g.w) }

        h = encoder(h, cos: cos, sin: sin, segments: segments)  // [L, hidden]

        // 2×2 patch merge per image: [h,w,d] -> [nh,kh,nw,kw,d] -> [nh,nw,kh,kw,d]
        // -> [nh*nw, kh*kw*d], concatenated over images.
        let kh = config.mergeH, kw = config.mergeW
        let d = config.hiddenSize
        var merged: [MLXArray] = []
        var off = 0
        for g in grids {
            let hw = g.h * g.w
            let seq = h[off ..< (off + hw)]
            off += hw
            let nh = g.h / kh, nw = g.w / kw
            let block = seq.reshaped(nh, kh, nw, kw, d)
                .transposed(0, 2, 1, 3, 4)
                .reshaped(nh * nw, kh * kw * d)
            merged.append(block)
        }
        return merged.count == 1 ? merged[0] : concatenated(merged, axis: 0)
    }
}

/// Encoder = `depth` blocks + final LayerNorm (`encoder.final_layernorm`).
public final class MoonViTEncoder: Module {
    @ModuleInfo(key: "blocks") var blocks: [MoonViTEncoderLayer]
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: LayerNorm

    init(_ c: MoonViTVisionConfig, eps: Float = 1e-5) {
        _blocks = ModuleInfo(wrappedValue: (0 ..< c.numHiddenLayers).map { _ in MoonViTEncoderLayer(c) }, key: "blocks")
        _finalLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: eps), key: "final_layernorm")
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, segments: [Int]) -> MLXArray {
        var h = x
        for block in blocks { h = block(h, cos: cos, sin: sin, segments: segments) }
        return finalLayerNorm(h)
    }
}

// MARK: - Connector (mlp1)

/// The top-level `mlp1` connector: `LayerNorm(vit*4) → Linear(vit*4, llm) →
/// GELU(exact) → Linear(llm, llm)`. Maps merged MoonViT tokens into the Qwen2.5
/// text hidden space. Keys `mlp1.{0,1,3}` (the `nn.Sequential` indices; index 2
/// is the parameterless GELU).
public final class LocateAnythingConnector: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    let act = GELU()      // exact erf GELU (nn.GELU() default)

    public init(vitHidden: Int, llmHidden: Int, eps: Float = 1e-5) {
        let inDim = vitHidden * 4
        _norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: inDim, eps: eps), key: "norm")
        _fc1 = ModuleInfo(wrappedValue: Linear(inDim, llmHidden, bias: true), key: "fc1")
        _fc2 = ModuleInfo(wrappedValue: Linear(llmHidden, llmHidden, bias: true), key: "fc2")
    }

    /// Map the checkpoint's `nn.Sequential` numeric indices (relative to the
    /// `mlp1.` prefix, which the caller strips) to named children: `0`→`norm`,
    /// `1`→`fc1`, `3`→`fc2` (index 2 is the parameterless GELU). MLX would
    /// otherwise read `"0"/"1"/"3"` as array slots (with a gap at 2) and fault.
    public static func remapKeys(_ weights: [(String, MLXArray)]) -> [(String, MLXArray)] {
        weights.map { key, v in
            let head = key.split(separator: ".", maxSplits: 1).first.map(String.init) ?? key
            let rest = key.dropFirst(head.count)   // includes leading "." if any
            let name: String
            switch head {
            case "0": name = "norm"
            case "1": name = "fc1"
            case "3": name = "fc2"
            default: name = head
            }
            return (name + rest, v)
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(act(fc1(norm(x))))
    }
}
