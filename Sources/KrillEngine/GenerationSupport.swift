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

/// Forward a text prompt through the model in query-chunks so the attention
/// score matrix stays `[heads, chunk, ctx]` instead of `[heads, L, L]`.
///
/// MLX's SDPA has no flash prefill kernel: it materializes the full per-head
/// `L x L` bf16 score matrix for any query length > 1 (verified - peak grows
/// quadratically and a single >14.3GB Metal buffer hard-OOMs around L ~ 21k
/// tokens on a 24GB box, and spikes peak well before that). Chunking bounds the
/// query dimension to `chunk`; the shared KV cache accumulates across chunks
/// exactly as a single pass would (the same cached-suffix forward the
/// partial-prefix-reuse path already relies on), so the result is numerically
/// the single-pass prefill. Each chunk uses the `lastTokenOnly` prefill closure
/// (full KV update, cheap single-token LM head) and is `eval`'d before the next
/// so its scores free first; only the final chunk's logits are returned.
///
/// `chunkSize <= 0` or a prompt that already fits in one chunk forwards in a
/// single call (zero behavior change for short prompts and prefix-cache hits).
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

/// Raw per-position logprobs for EVERY token in `tokenIds` (except the very
/// last, which has no "next" token within `tokenIds` to score), chunk by
/// chunk through `forward`, reusing the SAME `caches` array across every
/// chunk so a real attention-based model's causal state (and RoPE position)
/// accumulates exactly as one un-chunked forward would — mirroring
/// `chunkedTextPrefill`'s "same shared cache, sequential chunk forward"
/// pattern, but keeping every position's logits (not just the last) since
/// `echo` needs a logprob for every prompt token, not just the next one.
///
/// Factored out of `InferenceEngine.echoPromptLogprobs` (docs/LOGPROBS_PLAN.md
/// Phase 3, §5.4) so the chunk-boundary arithmetic — the running KV offset
/// across chunks, a chunk's LAST position scoring the FIRST token of the
/// NEXT chunk, and the final chunk's one-fewer `scoreCount` — can be
/// unit-tested against a synthetic context-DEPENDENT model with no
/// tokenizer/`InferenceEngine` involved (`EchoChunkingTests`), independent
/// of BOS-stripping or any other echo-specific text handling (that lives in
/// `echoPromptLogprobs` itself, which calls this).
///
/// - Parameters:
///   - tokenIds: the full token sequence to score (position i predicts
///     `tokenIds[i+1]`).
///   - caches: freshly-allocated caches (one per layer), never touched
///     before this call — reused across every chunk in this one invocation.
///   - chunkSize: `<= 0` forwards the whole sequence in one call.
///   - forward: the model's plain (NOT last-token-only) forward closure —
///     must return logits for every position in its input, `[1, len, vocab]`.
/// - Returns: `[TokenLogprobInfo?]`, length `tokenIds.count`. Index 0 is
///   always `nil` (no preceding context at all within `tokenIds`).
func chunkedPromptLogprobs(
    tokenIds: [Int],
    caches: [KVCacheProtocol],
    chunkSize: Int,
    topLogprobs: Int,
    forward: (MLXArray, [KVCacheProtocol]?) -> MLXArray
) -> [TokenLogprobInfo?] {
    guard tokenIds.count > 1 else {
        return tokenIds.isEmpty ? [] : [nil]
    }
    let total = tokenIds.count
    let effectiveChunk = chunkSize > 0 ? chunkSize : total
    var infos: [TokenLogprobInfo?] = [nil]
    var start = 0
    while start < total - 1 {
        let end = Swift.min(start + effectiveChunk, total)
        let chunkIds = Array(tokenIds[start ..< end])
        let inputArray = MLXArray(chunkIds.map { Int32($0) }).reshaped(1, chunkIds.count)
        let logits = forward(inputArray, caches)
        MLX.eval(logits)
        let chunkLen = chunkIds.count
        let logits2D = logits.reshaped(chunkLen, -1)
        let (logSoftmax, topIdx, topVals, n) = Sampler.rawLogSoftmaxAndTopN(
            logits2D, topLogprobs: topLogprobs)
        // Position p (local to this chunk) predicts the NEXT token, globally
        // at `start + p + 1`. The very last token overall has no "next"
        // token within `tokenIds` to score - callers that need one (e.g.
        // Server.swift stitching the first GENERATED token's own logprob
        // onto the end) get it from elsewhere, so the final chunk scores one
        // fewer position than it has tokens. A chunk that is NOT the final
        // one scores EVERY position, including its own last - that position's
        // "next" token is `tokenIds[end]`, the FIRST token of the NEXT chunk,
        // deliberately reached via a plain array index across the chunk
        // boundary (not anything the chunk loop needs to special-case).
        let scoreCount = (end < total) ? chunkLen : chunkLen - 1
        if scoreCount > 0 {
            let nextIds = Array(tokenIds[(start + 1) ... (start + scoreCount)])
            let chosenIdsArr = MLXArray(nextIds.map { Int32($0) })
            let scoredLogSoftmax = logSoftmax[0 ..< scoreCount, 0...]
            let chosenLogprobArr = Sampler.gatherChosenLogprob(
                scoredLogSoftmax, chosenIds: chosenIdsArr)
            let scoredTopIdx = topIdx?[0 ..< scoreCount, 0...]
            let scoredTopVals = topVals?[0 ..< scoreCount, 0...]
            if let scoredTopIdx, let scoredTopVals {
                eval(chosenLogprobArr, scoredTopIdx, scoredTopVals)
            } else {
                eval(chosenLogprobArr)
            }
            let chunkInfos = Sampler.logprobInfos(
                chosenLogprobs: chosenLogprobArr, topIdx: scoredTopIdx,
                topVals: scoredTopVals, n: n)
            infos.append(contentsOf: chunkInfos)
        }
        start = end
    }
    return infos
}
