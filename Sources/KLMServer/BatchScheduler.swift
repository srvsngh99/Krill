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

    /// One-shot guard so the speculative+batch exclusion notice is logged at
    /// most once per resident model, not on every request.
    private var didNoteSpecBatchExclusion = false

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

    /// Concurrency at/below which a fully-greedy request prefers the serial
    /// n-gram (prompt-lookup) speculative path over batching. Default 1: n-gram
    /// only when the request is solo (its bandwidth win is largest at batch 1;
    /// above this, batching amortizes weights across rows and wins instead).
    /// `KRILL_SPEC_CONCURRENCY_MAX`.
    static let defaultSpecConcurrencyMax = 1

    static func specConcurrencyMaxFromEnvironment() -> Int {
        if let v = ProcessInfo.processInfo.environment["KRILL_SPEC_CONCURRENCY_MAX"],
           let i = Int(v), i >= 0 {
            return i
        }
        return defaultSpecConcurrencyMax
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
                contextLimit: Int?, promptTemplateOverride: String?,
                format: OutputFormat? = nil, currentConcurrency: Int = 1,
                imagesData: [Data] = []) async -> GenResult {
        func serial() -> GenResult {
            engine.generate(
                messages: messages, params: params, maxTokens: maxTokens,
                useSpeculative: useSpeculative, usePrefixCache: usePrefixCache,
                imageData: imageData, audioData: audioData,
                contextLimit: contextLimit, promptTemplateOverride: promptTemplateOverride,
                format: format, imagesData: imagesData)
        }
        // A multi-image request carries its images in `imagesData` (not the
        // single `imageData`); treat either as "has image" so the batched/spec
        // gates below route it to the serial VLM driver.
        let anyImage: Data? = imageData ?? imagesData.first
        // Grammar-constrained decoding advances a per-sequence JSON automaton
        // each step; the shared-step batched loop cannot isolate per-row
        // masks, so a format request always takes the serial path.
        if format != nil { return serial() }

        // Load-adaptive spec/batch decision. When n-gram (prompt-lookup)
        // speculative decode is enabled on the engine and this request is solo
        // (or below the configured concurrency), take the SERIAL path so the
        // engine's n-gram branch engages: its bandwidth win is largest at batch
        // 1. Above the threshold, fall through to batching — one weight read then
        // serves many rows, and speculation's marginal win shrinks. Only fully
        // greedy, text-only requests use n-gram (the engine gate is stricter than
        // `isEligible`'s greedy), so a non-greedy request still batches.
        let fullyGreedy = params.temperature <= 0 && params.topP >= 1.0
            && params.topK <= 0 && params.minP <= 0 && params.mirostat == 0
        if engine.willUseNgramByDefault, useSpeculative != true, fullyGreedy,
           anyImage == nil, audioData == nil,
           currentConcurrency <= Self.specConcurrencyMaxFromEnvironment() {
            return serial()
        }
        guard isEligible(params: params, imageData: anyImage, audioData: audioData,
                         useSpeculative: useSpeculative) else {
            // Surface the one case where the user asked for two features that
            // do not compose: an explicit speculative opt-in on a request that
            // would batch if ONLY the speculative flag were dropped. Speculative
            // decode verifies a draft run against the target's own sequential
            // greedy sampler, which the shared-step batched loop cannot
            // interleave, so the request runs serially (no batching). WS2 found
            // the speculative speedup gate structurally unreachable on M-series,
            // so the actionable advice is to drop the speculative opt-in to
            // regain batching.
            //
            // The condition mirrors every OTHER `isEligible` gate (so spec is
            // the sole remaining blocker): a seeded non-greedy request is also
            // excluded, and for it "unset useSpeculative to batch" would be
            // wrong advice - it stays serial regardless. Only notice when
            // dropping the speculative flag genuinely restores batching.
            let greedy = params.temperature <= 0 && params.mirostat == 0
            let seedBlocks = params.seed != nil && !greedy
            if useSpeculative == true, numParallel >= 2, engine.supportsBatchedDecode,
               anyImage == nil, audioData == nil, !seedBlocks {
                noteSpecBatchExclusionOnce()
            }
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

    /// Log the speculative+batch incompatibility once per resident model. Actor
    /// isolation makes the `didNote...` check-and-set race-free without a lock.
    private func noteSpecBatchExclusionOnce() {
        guard !didNoteSpecBatchExclusion else { return }
        didNoteSpecBatchExclusion = true
        FileHandle.standardError.write(Data(
            ("[KrillLM] speculative decode requested with KRILL_NUM_PARALLEL=\(numParallel): "
             + "speculative and batched decode are not composable, so this request runs "
             + "serially (no batching). Unset useSpeculative to batch concurrent requests.\n").utf8))
    }
}
