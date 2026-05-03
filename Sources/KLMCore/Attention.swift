import MLX
import MLXNN
import MLXFast
import KLMCache

/// Grouped-Query Attention with Rotary Position Embeddings.
///
/// Supports GQA where num_kv_heads < num_attention_heads. Key and value heads
/// are repeated to match query heads via expand + reshape (no copy).
public class Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    public init(_ config: LlamaConfig) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Float(config.headDim).squareRoot()

        _qProj = ModuleInfo(
            wrappedValue: Linear(dim, numHeads * headDim, bias: false), key: "q_proj")
        _kProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: false), key: "k_proj")
        _vProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: false), key: "v_proj")
        _oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * headDim, dim, bias: false), key: "o_proj")

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // [B, L, numHeads*headDim] -> [B, numHeads, L, headDim]
        queries = queries.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        // Rotary position embeddings
        let offset = cache?.sequenceLength ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // Accumulate into KV cache
        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // Scaled dot-product attention (handles GQA repeat internally)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )

        // [B, numHeads, L, headDim] -> [B, L, numHeads*headDim]
        let reshaped = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(reshaped)
    }
}
