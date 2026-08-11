import Foundation
import MLX

/// The metric definitions behind `krill perplexity`, kept out of the CLI so
/// they can be unit-tested — the executable target cannot be linked into a test
/// bundle (see `KrillLaunch`), and an unverifiable metric is worse than none:
/// it produces confident numbers that quietly mean something else.
public enum PerplexityMath {

    /// Total negative log-likelihood, in NATS, of `targets` under `logits`.
    ///
    /// - Parameters:
    ///   - logits: `[T, V]` — one distribution per predicted position.
    ///   - targets: `T` gold token ids.
    ///
    /// Computed as `logsumexp(logits) - logits[target]`, which is the numerically
    /// stable form and avoids materializing a full log-softmax tensor beside the
    /// logits (at a 202k vocab that second tensor is the difference between
    /// fitting and paging).
    public static func totalNLL(logits: MLXArray, targets: [Int32]) -> Float {
        precondition(logits.ndim == 2, "logits must be [T, V]")
        precondition(logits.dim(0) == targets.count,
                     "got \(targets.count) targets for \(logits.dim(0)) positions")
        guard !targets.isEmpty else { return 0 }
        let f32 = logits.asType(.float32)
        let idx = MLXArray(targets).reshaped(targets.count, 1)
        let lse = MLX.logSumExp(f32, axis: -1)
        let picked = MLX.takeAlong(f32, idx, axis: -1).squeezed(axis: -1)
        let nll = (lse - picked).sum()
        MLX.eval(nll)
        return nll.item(Float.self)
    }

    /// `exp(mean NLL per token)`. Comparable ONLY across builds that share a
    /// tokenizer: a tokenizer that packs more text into each token earns a lower
    /// perplexity for free, so this says nothing across model families.
    public static func perplexity(totalNLL: Double, tokens: Int) -> Double {
        guard tokens > 0 else { return .nan }
        return exp(totalNLL / Double(tokens))
    }

    /// Total information over the ORIGINAL UTF-8 byte count. Tokenizer
    /// -independent, so this is the honest cross-family comparison.
    public static func bitsPerByte(totalNLL: Double, bytes: Int) -> Double {
        guard bytes > 0 else { return .nan }
        return totalNLL / Double(bytes) / log(2.0)
    }
}
