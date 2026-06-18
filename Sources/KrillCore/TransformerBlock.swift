import MLX
import MLXNN
import KrillCache

/// Single transformer decoder block with pre-norm residual connections.
///
/// Architecture: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
public class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Attention
    @ModuleInfo(key: "mlp") var mlp: FeedForward
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    public init(_ config: LlamaConfig) {
        _selfAttn = ModuleInfo(
            wrappedValue: Attention(config), key: "self_attn")
        _mlp = ModuleInfo(
            wrappedValue: FeedForward(config), key: "mlp")
        _inputLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "input_layernorm")
        _postAttentionLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "post_attention_layernorm")
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil,
        rowOffsets: [Int]? = nil
    ) -> MLXArray {
        // Pre-norm attention with residual
        let attnOut = selfAttn(inputLayernorm(x), mask: mask, cache: cache, rowOffsets: rowOffsets)
        let h = x + attnOut

        // Pre-norm FFN with residual
        let ffnOut = mlp(postAttentionLayernorm(h))
        return h + ffnOut
    }
}
