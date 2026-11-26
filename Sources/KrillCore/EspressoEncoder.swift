import MLX
import MLXNN

struct EspressoAttentionWindow {
    private let length: Int
    private let queries: MLXArray
    private let keys: MLXArray
    private let mask: MLXArray

    init(length: Int, width: Int) {
        self.length = length
        let blocks = (length + width - 1) / width
        let start = MLXArray(Int32(0) ..< Int32(blocks)).reshaped(blocks, 1) * Int32(width)
        let q = start + MLXArray(Int32(0) ..< Int32(width))
        let k = start + MLXArray(Int32(-width) ..< Int32(2 * width))
        queries = MLX.minimum(q, Int32(length - 1))
        keys = MLX.minimum(MLX.maximum(k, 0), Int32(length - 1))
        let near = MLX.abs(q.expandedDimensions(axis: -1) - k.expandedDimensions(axis: 1)) .<= Int32(width)
        let valid = MLX.logicalAnd(k .>= 0, k .< Int32(length)).expandedDimensions(axis: 1)
        mask = MLX.logicalAnd(near, valid).expandedDimensions(axis: 1)
    }

    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float) -> MLXArray {
        func blocks(_ x: MLXArray, at positions: MLXArray) -> MLXArray {
            x[0].take(positions, axis: 1).transposed(1, 0, 2, 3)
        }
        let heads = MLXFast.scaledDotProductAttention(
            queries: blocks(q, at: queries), keys: blocks(k, at: keys), values: blocks(v, at: keys),
            scale: scale, mask: mask)
        return heads.transposed(1, 0, 2, 3).reshaped(1, q.dim(1), -1, q.dim(3))[0..., 0..., 0 ..< length, 0...]
    }
}

final class EspressoEncoder: Module {
    // The bundle config describes rendering. These dimensions and layer settings come from Vela3 Fast.
    static let hiddenSize = 1024
    static let vocabSize = 49160
    private static let window = 128

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [TransformerBlock]
    @ModuleInfo var norm: RMSNorm

    private static func isGlobal(_ layer: Int) -> Bool { layer % 3 == 2 }

    override init() {
        let hidden = Self.hiddenSize
        let count = 24
        _embedTokens = ModuleInfo(wrappedValue: Embedding(embeddingCount: Self.vocabSize, dimensions: hidden),
                                  key: "embed_tokens")
        _layers = ModuleInfo(wrappedValue: (0 ..< count).map { layer in
            TransformerBlock(LlamaConfig(
                hiddenSize: hidden, intermediateSize: 3072, numAttentionHeads: 8, numKeyValueHeads: 8,
                numHiddenLayers: count, vocabSize: Self.vocabSize, rmsNormEps: 1e-6,
                ropeTheta: Self.isGlobal(layer) ? 2_560_000 : 10_000))
        })
        _norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: hidden, eps: 1e-6))
        super.init()
    }

    func callAsFunction(_ ids: [Int32]) -> MLXArray {
        let length = ids.count
        let window = length > Self.window + 1 ? EspressoAttentionWindow(length: length, width: Self.window) : nil
        var hidden = embedTokens(MLXArray(ids, [1, length]))
        for (i, layer) in layers.enumerated() {
            if let window, !Self.isGlobal(i) {
                let attention = layer.selfAttn
                let x = layer.inputLayernorm(hidden)
                func heads(_ y: MLXArray) -> MLXArray {
                    y.reshaped(1, length, attention.numHeads, attention.headDim).transposed(0, 2, 1, 3)
                }
                let q = attention.rope(heads(attention.qProj(x)), offset: 0)
                let k = attention.rope(heads(attention.kProj(x)), offset: 0)
                let out = window.attend(q, k, heads(attention.vProj(x)), scale: attention.scale)
                hidden = hidden + attention.oProj(out.transposed(0, 2, 1, 3).reshaped(1, length, Self.hiddenSize))
                hidden = hidden + layer.mlp(layer.postAttentionLayernorm(hidden))
            } else {
                hidden = layer(hidden)
            }
        }
        return norm(hidden)[0]
    }
}
