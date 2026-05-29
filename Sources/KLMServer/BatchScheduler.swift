import Foundation
import KLMEngine
import KLMSampler

/// Routes concurrent same-model generation requests into the engine's
/// continuous batcher (follow-up #8, Stage C1). Sits between the server
/// handlers and a single resident ``InferenceEngine``: eligible requests are
/// submitted to ``InferenceEngine/submitBatched(_:maxRows:windowMs:)``, which
/// admits each into one persistent running batch and drops finished rows
/// between steps. Every other request — `numParallel < 2`, an ineligible
/// family, multimodal, per-row seeded sampling, or an explicit speculative
/// opt-in — falls straight through to serial
/// ``InferenceEngine/generate(messages:...)`` and is byte-identical to before.
///
/// One scheduler exists per resident model (owned by ``EngineRegistry``), so a
/// batch only ever contains rows that share an engine + KV layout.
///
/// Stage B was static (a fixed cohort decoded to completion); Stage C1 makes
/// admission continuous — a newcomer joins the running batch at the next epoch
/// boundary instead of waiting for the current cohort to finish. The coalescing
/// behaviour now lives inside the batcher (cold-start gather window + rolling
/// admission), so this type is a thin eligibility gate + pass-through.
actor BatchScheduler {
    typealias GenResult = (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?)

    private let engine: InferenceEngine
    private let numParallel: Int
    private let windowMs: Int

    /// - Parameters:
    ///   - engine: the resident engine this scheduler batches for.
    ///   - numParallel: `KRILL_NUM_PARALLEL` — max simultaneously-decoding rows
    ///     in the running batch (and the in-flight cap the `GenerationQueue`
    ///     already enforces). `< 2` disables batching entirely.
    ///   - windowMs: cold-start gather window passed to the batcher; on an
    ///     idle->busy transition it waits this long once so genuinely-concurrent
    ///     arrivals start in one batch instead of the first decoding solo.
    init(engine: InferenceEngine, numParallel: Int, windowMs: Int) {
        self.engine = engine
        self.numParallel = max(1, numParallel)
        self.windowMs = max(0, windowMs)
    }

    /// Default coalescing window when `KRILL_BATCH_WINDOW_MS` is unset. Small
    /// enough that a lone request under `numParallel >= 2` pays only a few ms
    /// of extra latency, large enough to gather genuinely-concurrent arrivals.
    static let defaultWindowMs = 8

    static func windowMsFromEnvironment() -> Int {
        if let v = ProcessInfo.processInfo.environment["KRILL_BATCH_WINDOW_MS"],
           let i = Int(v), i >= 0 {
            return i
        }
        return defaultWindowMs
    }

    /// Whether a request can join a batch. Batching helps only at
    /// `numParallel >= 2` on a batch-capable engine; multimodal rows and
    /// per-row seeded non-greedy sampling (whose RNG can't be isolated under a
    /// shared step) take the serial path.
    private func isEligible(params: SamplingParams, imageData: Data?, audioData: Data?,
                            useSpeculative: Bool?) -> Bool {
        guard numParallel >= 2, engine.supportsBatchedDecode else { return false }
        guard imageData == nil, audioData == nil else { return false }
        if useSpeculative == true { return false }   // honor explicit spec opt-in serially
        let greedy = params.temperature <= 0 && params.mirostat == 0
        if params.seed != nil && !greedy { return false }
        return true
    }

    /// Submit a generation request. Returns the same `(stream, stats)` shape as
    /// ``InferenceEngine/generate(messages:...)`` so handlers are unchanged
    /// beyond the call site. Eligible requests join the running batch via the
    /// continuous batcher; everyone else takes the serial path.
    func submit(messages: [[String: String]], params: SamplingParams, maxTokens: Int,
                useSpeculative: Bool?, usePrefixCache: Bool,
                imageData: Data?, audioData: Data?,
                contextLimit: Int?, promptTemplateOverride: String?) async -> GenResult {
        func serial() -> GenResult {
            engine.generate(
                messages: messages, params: params, maxTokens: maxTokens,
                useSpeculative: useSpeculative, usePrefixCache: usePrefixCache,
                imageData: imageData, audioData: audioData,
                contextLimit: contextLimit, promptTemplateOverride: promptTemplateOverride)
        }
        guard isEligible(params: params, imageData: imageData, audioData: audioData,
                         useSpeculative: useSpeculative) else {
            return serial()
        }

        let req = BatchGenRequest(
            messages: messages, params: params, maxTokens: maxTokens,
            contextLimit: contextLimit, promptTemplateOverride: promptTemplateOverride,
            useSpeculative: useSpeculative, usePrefixCache: usePrefixCache)
        // submitBatched returns nil only when the model turned out not to be
        // batch-eligible after prompt construction (unknown family / empty
        // prompt) — fall back to serial so the request is always served.
        return engine.submitBatched(req, maxRows: numParallel, windowMs: windowMs)
            ?? serial()
    }
}
