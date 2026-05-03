import MLX
import MLXNN
import KLMKernels

/// SwiGLU feed-forward network with fused Metal kernel.
///
/// Computes: down_proj(silu(gate_proj(x)) * up_proj(x))
/// Uses fused Metal kernel to avoid materializing the intermediate silu result.
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
        let gate = gateProj(x)
        let up = upProj(x)
        // Fused SwiGLU: single kernel pass instead of silu + multiply
        let activated = KLMKernels.fusedSwiGLU(gate: gate, up: up)
        return downProj(activated)
    }
}
