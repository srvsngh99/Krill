import Foundation
import KLMEngine
import KLMSampler

/// Coalesces concurrent same-model generation requests into one batched
/// forward (follow-up #8, Stage B — wiring). Sits between the server handlers
/// and a single resident ``InferenceEngine``: eligible requests are gathered
/// into a static cohort (up to `numParallel`, within a small coalescing
/// window) and driven through ``InferenceEngine/generateBatched(_:)``; every
/// other request — `numParallel < 2`, an ineligible family, multimodal,
/// per-row seeded sampling, or an explicit speculative opt-in — falls straight
/// through to today's serial ``InferenceEngine/generate(messages:...)`` and is
/// byte-identical to before.
///
/// One scheduler exists per resident model (owned by ``EngineRegistry``), so
/// the cohort only ever batches rows that share an engine + KV layout.
///
/// v1 (Stage B) is **static** batching: a cohort runs to completion before the
/// next one starts. Continuous admission (splicing a newcomer into a running
/// batch) is Stage C.
actor BatchScheduler {
    typealias GenResult = (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?)

    private let engine: InferenceEngine
    private let numParallel: Int
    private let windowNanos: UInt64

    private struct Waiter {
        let req: BatchGenRequest
        let cont: CheckedContinuation<GenResult, Never>
    }
    /// Requests waiting to be batched, in arrival order.
    private var pending: [Waiter] = []
    /// True while a coalescing-window timer is in flight, so we arm at most one.
    private var timerArmed = false

    /// - Parameters:
    ///   - engine: the resident engine this scheduler batches for.
    ///   - numParallel: `KRILL_NUM_PARALLEL` — max cohort size (and the
    ///     in-flight cap the `GenerationQueue` already enforces). `< 2`
    ///     disables batching entirely.
    ///   - windowMs: coalescing window; the cohort fires when it reaches
    ///     `numParallel` rows OR this many ms after the first row arrives.
    init(engine: InferenceEngine, numParallel: Int, windowMs: Int) {
        self.engine = engine
        self.numParallel = max(1, numParallel)
        self.windowNanos = UInt64(max(0, windowMs)) * 1_000_000
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
    /// beyond the call site. Eligible requests await their cohort; everyone
    /// else returns immediately on the serial path.
    func submit(messages: [[String: String]], params: SamplingParams, maxTokens: Int,
                useSpeculative: Bool?, usePrefixCache: Bool,
                imageData: Data?, audioData: Data?,
                contextLimit: Int?, promptTemplateOverride: String?) async -> GenResult {
        guard isEligible(params: params, imageData: imageData, audioData: audioData,
                         useSpeculative: useSpeculative) else {
            return engine.generate(
                messages: messages, params: params, maxTokens: maxTokens,
                useSpeculative: useSpeculative, usePrefixCache: usePrefixCache,
                imageData: imageData, audioData: audioData,
                contextLimit: contextLimit, promptTemplateOverride: promptTemplateOverride)
        }

        let req = BatchGenRequest(
            messages: messages, params: params, maxTokens: maxTokens,
            contextLimit: contextLimit, promptTemplateOverride: promptTemplateOverride,
            useSpeculative: useSpeculative, usePrefixCache: usePrefixCache)
        return await withCheckedContinuation { (cont: CheckedContinuation<GenResult, Never>) in
            pending.append(Waiter(req: req, cont: cont))
            if pending.count >= numParallel {
                fireCohort()
            } else {
                armTimer()
            }
        }
    }

    private func armTimer() {
        guard !timerArmed else { return }
        timerArmed = true
        let ns = windowNanos
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: ns)
            await self?.timerFired()
        }
    }

    private func timerFired() {
        timerArmed = false
        if !pending.isEmpty { fireCohort() }
    }

    /// Take up to `numParallel` waiters and drive them through one batched
    /// generate, handing each its row's stream. `generateBatched` returns one
    /// result per request in order (and itself falls back to serial for a
    /// single-row cohort), so the mapping is positional.
    private func fireCohort() {
        let take = min(numParallel, pending.count)
        guard take > 0 else { return }
        let cohort = Array(pending.prefix(take))
        pending.removeFirst(take)

        let results = engine.generateBatched(cohort.map { $0.req })
        // generateBatched returns exactly one result per request on every path,
        // so this is positional. Guard defensively anyway: a count mismatch
        // must never leave a waiter's continuation un-resumed (the HTTP request
        // would hang with no timeout) — fall each unmatched waiter back to a
        // direct serial generate so every continuation is resumed exactly once.
        if results.count == cohort.count {
            for (i, w) in cohort.enumerated() {
                w.cont.resume(returning: results[i])
            }
        } else {
            for (i, w) in cohort.enumerated() {
                if i < results.count {
                    w.cont.resume(returning: results[i])
                } else {
                    w.cont.resume(returning: engine.generate(
                        messages: w.req.messages, params: w.req.params,
                        maxTokens: w.req.maxTokens, useSpeculative: w.req.useSpeculative,
                        usePrefixCache: w.req.usePrefixCache, contextLimit: w.req.contextLimit,
                        promptTemplateOverride: w.req.promptTemplateOverride))
                }
            }
        }

        // Anything that arrived during assembly: fire again if a full cohort
        // is ready, else re-arm the window for the stragglers.
        if pending.count >= numParallel {
            fireCohort()
        } else if !pending.isEmpty {
            armTimer()
        }
    }
}
