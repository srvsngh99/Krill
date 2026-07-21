import Foundation
import MLX
import KrillCache
import KrillSampler

/// One-shot, thread-safe cancellation flag for a single generation stream.
/// Set from an `AsyncStream` termination callback and polled by the decode
/// loop so abandoned replies stop consuming compute.
final class GenerationCancelToken: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }
}

/// Lock-backed holder because the generation task writes stats while callers
/// may poll the accessor from another executor.
final class StatsHolder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: GenerationStats?

    var stats: GenerationStats? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return stored
        }
        set {
            lock.lock()
            stored = newValue
            lock.unlock()
        }
    }
}

/// Mutable flag box for escaping token callbacks. Each instance is confined to
/// a single generation loop.
final class FlagBox: @unchecked Sendable {
    var value: Bool

    init(_ value: Bool) {
        self.value = value
    }
}

/// Forward a text prompt through the model in query chunks so the attention
/// score matrix stays `[heads, chunk, context]` instead of `[heads, L, L]`.
/// The shared cache accumulates across chunks exactly as in a single prefill;
/// only the final chunk's logits are returned.
func chunkedTextPrefill(
    input: MLXArray,
    caches: [KVCacheProtocol],
    chunkSize: Int,
    prefillForward: ((MLXArray, [KVCacheProtocol]) -> MLXArray)?,
    forward: (MLXArray, [KVCacheProtocol]) -> MLXArray
) -> MLXArray {
    // Non-escaping closures require an explicit branch; using `??` would force
    // the selected closure to escape.
    func runChunk(_ tokens: MLXArray) -> MLXArray {
        if let prefillForward {
            return prefillForward(tokens, caches)
        }
        return forward(tokens, caches)
    }

    let total = input.dim(1)
    if chunkSize <= 0 || total <= chunkSize {
        return runChunk(input)
    }

    var start = 0
    var last: MLXArray?
    while start < total {
        let end = Swift.min(start + chunkSize, total)
        let logits = runChunk(input[0..., start ..< end])
        MLX.eval(logits)
        last = logits
        start = end
    }
    return last!
}
