import MLX
import MLXNN

/// SwiGLU feed-forward network.
///
/// Computes: down_proj(silu(gate_proj(x)) * up_proj(x))
/// Standard in Llama, Qwen, and Mistral families.
public class FeedForward: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(_ config: LlamaConfig) {
        let dim = config.hiddenSize
        let hidden = config.intermediateSize

        _gateProj = ModuleInfo(
            wrappedValue: Linear(dim, hidden, bias: false), key: "gate_proj")
        _upProj = ModuleInfo(
            wrappedValue: Linear(dim, hidden, bias: false), key: "up_proj")
        _downProj = ModuleInfo(
            wrappedValue: Linear(hidden, dim, bias: false), key: "down_proj")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}
