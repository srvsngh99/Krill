import XCTest
@testable import KLMServer
import KLMEngine
import KLMSampler

/// Unit tests for ``BatchScheduler``'s routing/eligibility and no-deadlock
/// guarantees. True batched-decode correctness needs a real checkpoint and is
/// covered by `BatchedDecodeLiveTests` (gated on `KLM_BATCH_MODEL_PATH`); here
/// we use an unloaded engine, which is never batch-eligible, to prove every
/// path still terminates and falls through to the serial contract.
final class BatchSchedulerTests: XCTestCase {
    private func unloadedEngine(_ name: String = "x") -> InferenceEngine {
        InferenceEngine(modelDirectory: URL(fileURLWithPath: "/tmp/krill-batch-\(name)"))
    }

    /// `numParallel == 1` disables batching: submit passes straight to the
    /// serial path and returns a finishing stream — never coalesces, never
    /// waits on a window, never deadlocks.
    func testNumParallelOneIsPassthrough() async {
        let sched = BatchScheduler(engine: unloadedEngine(), numParallel: 1, windowMs: 0)
        let (stream, _) = await batchSubmitOnce(sched)
        var count = 0
        for await _ in stream { count += 1; if count > 4 { break } }
        XCTAssertLessThanOrEqual(count, 4)   // unloaded engine → empty, finishing stream
    }

    /// An engine that is not batch-eligible (here: unloaded) must fall through
    /// to the serial path even at `numParallel >= 2`, rather than parking the
    /// request waiting for a cohort that can never be batched.
    func testIneligibleEngineFallsThroughAtHighParallelism() async {
        let sched = BatchScheduler(engine: unloadedEngine(), numParallel: 4, windowMs: 50)
        let (stream, _) = await batchSubmitOnce(sched)
        for await _ in stream { /* drain */ }
        XCTAssertTrue(true)   // reaching here means the submit returned + finished
    }

    /// Several concurrent submits to an ineligible engine all return promptly
    /// (each on the serial path) — exercises the actor under concurrency with
    /// no batching, ensuring no submit is lost or stuck.
    func testConcurrentSubmitsAllReturn() async {
        let sched = BatchScheduler(engine: unloadedEngine(), numParallel: 4, windowMs: 10)
        await withTaskGroup(of: Bool.self) { group in
            for _ in 0 ..< 6 {
                group.addTask {
                    // Capture only `sched` (an actor, Sendable); calling a
                    // file-scope helper avoids capturing `self` (the
                    // non-Sendable XCTestCase) into this @Sendable task.
                    let (stream, _) = await batchSubmitOnce(sched)
                    for await _ in stream { /* drain */ }
                    return true
                }
            }
            var completed = 0
            for await ok in group where ok { completed += 1 }
            XCTAssertEqual(completed, 6)
        }
    }

    /// An explicit speculative opt-in is never batched (the two do not
    /// compose); it falls through to the serial path and still returns a
    /// finishing stream. The one-time exclusion notice only logs when the
    /// engine is genuinely batch-capable (a loaded checkpoint), so here - on an
    /// unloaded engine - we just assert the spec request still terminates and
    /// the new exclusion branch never hangs or traps.
    func testSpeculativeOptInFallsThroughSerially() async {
        let sched = BatchScheduler(engine: unloadedEngine(), numParallel: 4, windowMs: 10)
        let (stream, _) = await sched.submit(
            messages: [["role": "user", "content": "hi"]],
            params: .greedy, maxTokens: 8,
            useSpeculative: true, usePrefixCache: true,
            imageData: nil, audioData: nil,
            contextLimit: nil, promptTemplateOverride: nil)
        for await _ in stream { /* drain */ }
        XCTAssertTrue(true)   // reaching here means the spec request returned + finished
    }

    /// The coalescing window default is used when the env var is unset.
    func testWindowDefaultFromEnvironment() {
        if ProcessInfo.processInfo.environment["KRILL_BATCH_WINDOW_MS"] == nil {
            XCTAssertEqual(BatchScheduler.windowMsFromEnvironment(),
                           BatchScheduler.defaultWindowMs)
        }
    }

    /// The spec-vs-batch concurrency threshold default is used when unset.
    func testSpecConcurrencyMaxDefaultFromEnvironment() {
        if ProcessInfo.processInfo.environment["KRILL_SPEC_CONCURRENCY_MAX"] == nil {
            XCTAssertEqual(BatchScheduler.specConcurrencyMaxFromEnvironment(),
                           BatchScheduler.defaultSpecConcurrencyMax)
        }
    }

    /// With n-gram enabled on the engine, a solo (low-concurrency) greedy request
    /// takes the serial path so the n-gram branch can engage; a high-concurrency
    /// one falls through to batching. On an unloaded engine both paths end in the
    /// same finishing serial stream — this asserts the new `currentConcurrency`
    /// argument is threaded and neither routing decision hangs or traps.
    func testNgramAdaptiveRoutingTerminatesAtBothConcurrencies() async {
        let engine = unloadedEngine("ngram")
        engine.setNgramSpec(true)
        XCTAssertTrue(engine.willUseNgramByDefault)
        let sched = BatchScheduler(engine: engine, numParallel: 4, windowMs: 10)
        for concurrency in [1, 8] {
            let (stream, _) = await sched.submit(
                messages: [["role": "user", "content": "hi"]],
                params: .greedy, maxTokens: 8,
                useSpeculative: nil, usePrefixCache: true,
                imageData: nil, audioData: nil,
                contextLimit: nil, promptTemplateOverride: nil,
                format: nil, currentConcurrency: concurrency)
            for await _ in stream { /* drain */ }
        }
        XCTAssertTrue(true)
    }
}

/// Submit one canonical request to `sched`. File-scope (not a method) so the
/// concurrent-submit test can call it from a `@Sendable` task without
/// capturing the non-Sendable `XCTestCase`.
private func batchSubmitOnce(_ sched: BatchScheduler) async
    -> (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?)
{
    await sched.submit(
        messages: [["role": "user", "content": "hi"]],
        params: .greedy, maxTokens: 8,
        useSpeculative: nil, usePrefixCache: true,
        imageData: nil, audioData: nil,
        contextLimit: nil, promptTemplateOverride: nil)
}
