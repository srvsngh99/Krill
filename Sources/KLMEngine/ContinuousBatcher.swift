import Foundation
import MLX
import KLMCore
import KLMCache
import KLMSampler

/// Continuous (in-flight) batching for plain-causal families — follow-up #8,
/// Stage C1. One persistent decode loop per resident model admits new requests
/// into the *running* batch between steps and drops finished/cancelled rows,
/// rather than running a fixed cohort to completion (Stage B). Each row keeps
/// its own authoritative `[KVCache]`; the loop re-stacks the active set
/// left-padded only when the set changes (an "epoch"), so within an epoch the
/// per-step cost is identical to the verified Stage B core.
///
/// Why this is correct: the verified core proved a batched row reproduces its
/// solo decode given per-row RoPE offsets + a left-pad mask. Admission is just
/// "append a row to the set" and shrink is "remove a row"; the set is
/// re-stacked from per-row caches whenever it changes, so every active row is
/// always decoded exactly as if it ran alone (no cross-row bleed).
///
/// Scope: fp16 KV, no speculative decode. Batched families now include Gemma 4
/// (dense + MoE) and Qwen3 MoE alongside Llama 3.x / Qwen 2.5-3 dense (Stage
/// C2/C3). The shared prefix cache IS consulted per row at prefill (Stage C4,
/// via `Deps.prefillRow`); int8-on-batched and speculative remain later
/// sub-stages.
final class ContinuousBatcher: @unchecked Sendable {
    /// Engine-supplied closures + constants the decode loop needs. Bundled into
    /// one `@unchecked Sendable` value (the loop runs off the caller's context).
    struct Deps: @unchecked Sendable {
        /// Prefill one row's prompt into its own fp16 caches and return the
        /// prefill (last-token) logits. The engine bakes the shared prefix-cache
        /// lookup/restore/trim/store into this closure (honoring the row's
        /// `usePrefixCache`), so a full prefix hit restores the prompt's KV and
        /// forwards just the last token - identical KV to a cold prefill, so the
        /// stacked decode is byte-for-byte the same. Caches are
        /// `[KVCacheProtocol]` so the same loop drives the fp16 (`KVCache`) and
        /// the int8 (`QuantizedKVCache`) paths.
        let prefillRow: ([Int], [KVCacheProtocol], Bool) -> MLXArray
        let batchedForward: (MLXArray, [KVCacheProtocol], MLXArray, [Int]) -> MLXArray
        let numLayers: Int
        /// When true, rows allocate `QuantizedKVCache` (int8) instead of fp16
        /// `KVCache` (Stage C4, Gemma 4 only - the engine sets this from
        /// `usesQuantizedKVCache` AND a quantized batched forward being wired).
        let useQuantizedKV: Bool
        let decode: (Int) -> String
        let stopIds: Set<Int>
    }

    /// Thread-safe cancellation flag for one row (set from the stream's
    /// `onTermination`, read by the decode loop).
    final class CancelFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return value }
        func cancel() { lock.lock(); value = true; lock.unlock() }
    }

    /// Thread-safe holder for one row's final stats.
    final class StatsBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value: GenerationStats?
        var stats: GenerationStats? {
            get { lock.lock(); defer { lock.unlock() }; return value }
            set { lock.lock(); value = newValue; lock.unlock() }
        }
    }

    /// A request that has been submitted but not yet prefilled/admitted.
    private struct Pending {
        let promptTokens: [Int]
        let req: BatchGenRequest
        let sampler: Sampler
        let cont: AsyncStream<TokenEvent>.Continuation
        let stats: StatsBox
        let cancelled: CancelFlag
    }

    /// An active row in the running batch.
    ///
    /// THREADING INVARIANT: a `Row`'s mutable state (`caches`, `current`,
    /// `generated`, `recent`, `epochBaseLen`, `isFinished`, `decodeStartedAt`)
    /// is owned exclusively by the single `runLoop` Task and is only ever read
    /// or written there — never from `submit`/`stop`/`onTermination`. Those
    /// other entry points touch only the lock-guarded `pending`/`running`/
    /// `stopped`/`loopTask` and the row's own thread-safe `cancelled`
    /// (`CancelFlag`) / `stats` (`StatsBox`) boxes. So the bare `var`s here
    /// need no lock, and `finalize`'s `isFinished` guard runs single-threaded.
    /// Preserve this: any new cross-thread access to a Row field must go through
    /// a thread-safe box like `cancelled`/`stats`, not a bare `var`.
    private final class Row {
        let promptLen: Int
        let sampler: Sampler
        let needsHistory: Bool
        let cont: AsyncStream<TokenEvent>.Continuation
        let stats: StatsBox
        let cancelled: CancelFlag
        let maxTokens: Int
        var caches: [KVCacheProtocol]
        var current: Int        // last-emitted token; the next forward's input
        var generated = 0
        var recent: [Int]
        var prefillTime: Double
        var admittedAt: Double
        var epochBaseLen = 0    // cache length at the current epoch's start
        var isFinished = false  // set once by finalize(); guards double-finish
        var decodeStartedAt: Double?   // wall time of this row's first batched step
        init(promptLen: Int, sampler: Sampler, cont: AsyncStream<TokenEvent>.Continuation,
             stats: StatsBox, cancelled: CancelFlag, maxTokens: Int, caches: [KVCacheProtocol],
             current: Int, recent: [Int], prefillTime: Double, admittedAt: Double) {
            self.promptLen = promptLen
            self.sampler = sampler
            self.needsHistory = sampler.needsHistory
            self.cont = cont
            self.stats = stats
            self.cancelled = cancelled
            self.maxTokens = maxTokens
            self.caches = caches
            self.current = current
            self.recent = recent
            self.prefillTime = prefillTime
            self.admittedAt = admittedAt
        }
    }

    private let deps: Deps
    /// Max simultaneously-decoding rows (`KRILL_NUM_PARALLEL`).
    private let maxRows: Int
    /// Cold-start gather window (`KRILL_BATCH_WINDOW_MS`): on an idle->busy
    /// transition the loop waits this long once so genuinely-concurrent
    /// arrivals start in one batch instead of the first decoding solo.
    private let windowNanos: UInt64

    private let lock = NSLock()
    private var pending: [Pending] = []
    private var running = false
    private var stopped = false
    /// Handle to the single decode loop Task, so `stop()` can cancel it and
    /// halt active decode promptly (rather than waiting for the next epoch
    /// boundary to observe `stopped`). Guarded by `lock`.
    private var loopTask: Task<Void, Never>?

    init(deps: Deps, maxRows: Int, windowMs: Int) {
        self.deps = deps
        self.maxRows = max(1, maxRows)
        self.windowNanos = UInt64(max(0, windowMs)) * 1_000_000
    }

    /// Submit one already-tokenized request. Returns its own token stream and a
    /// stats accessor, matching `InferenceEngine.generate`'s contract. The
    /// decode happens on the shared background loop (spawned on demand).
    func submit(promptTokens: [Int], req: BatchGenRequest)
        -> (stream: AsyncStream<TokenEvent>, stats: @Sendable () -> GenerationStats?)
    {
        let statsBox = StatsBox()
        let cancelled = CancelFlag()
        var captured: AsyncStream<TokenEvent>.Continuation!
        let stream = AsyncStream<TokenEvent> { captured = $0 }
        let cont = captured!
        cont.onTermination = { _ in cancelled.cancel() }

        let pendingItem = Pending(
            promptTokens: promptTokens, req: req,
            sampler: Sampler(params: req.params), cont: cont,
            stats: statsBox, cancelled: cancelled)

        lock.lock()
        if stopped {
            // Engine torn down: finish immediately rather than stranding the row.
            lock.unlock()
            cont.finish()
            return (stream, { statsBox.stats })
        }
        pending.append(pendingItem)
        // Spawn a loop only on the idle->busy edge. The `running` flip and the
        // spawn happen together under `lock`, and the loop only ever clears
        // `running` inside `takePending` (also under `lock`) at the same instant
        // it decides to idle-exit with zero active rows. So at most one loop is
        // ever live: a submit that races a just-idle-exiting loop sees
        // `running == false` and spawns the sole successor.
        let wasRunning = running
        running = true
        if !wasRunning {
            // Capture self STRONGLY: the running loop must keep the batcher
            // alive until it drains, even after `unload()` clears the engine's
            // reference (otherwise in-flight rows' streams would be stranded).
            // The Task releases the reference when `runLoop()` returns.
            loopTask = Task(priority: .userInitiated) { await self.runLoop() }
        }
        lock.unlock()
        return (stream, { statsBox.stats })
    }

    /// Stop the loop and finish any waiting (not-yet-admitted) requests. Called
    /// from `InferenceEngine.unload()` so a swap/unload never strands a stream.
    /// Cancels the decode loop so it halts active decode at the next step check
    /// (rather than draining the whole batch first) and finalizes its in-flight
    /// rows itself; here we only finish the not-yet-admitted `pending` ones.
    func stop() {
        lock.lock()
        stopped = true
        let waiting = pending
        pending.removeAll()
        let task = loopTask
        lock.unlock()
        task?.cancel()
        for p in waiting { p.cont.finish() }
    }

    // MARK: - Loop

    /// Take up to `maxCount` pending requests, or signal idle-exit when there is
    /// nothing pending and no active rows. Atomic with the `running` flag so a
    /// concurrent `submit` either is seen here or spawns a fresh loop.
    private func takePending(max maxCount: Int, activeEmpty: Bool)
        -> (taken: [Pending], idleExit: Bool, stop: Bool)
    {
        lock.lock(); defer { lock.unlock() }
        if stopped { running = false; return ([], false, true) }
        if pending.isEmpty {
            if activeEmpty { running = false; return ([], true, false) }
            return ([], false, false)
        }
        guard maxCount > 0 else { return ([], false, false) }
        let n = min(maxCount, pending.count)
        let taken = Array(pending.prefix(n))
        pending.removeFirst(n)
        return (taken, false, false)
    }

    private func pendingIsEmpty() -> Bool {
        lock.lock(); defer { lock.unlock() }
        return pending.isEmpty
    }

    private func runLoop() async {
        // Cold-start gather window: let concurrent arrivals collect so they
        // start batched rather than the first request decoding alone.
        if windowNanos > 0 { try? await Task.sleep(nanoseconds: windowNanos) }

        var rows: [Row] = []
        while true {
            // Cancelled by stop() (unload/swap): halt before touching the model
            // again and finish every in-flight row's stream.
            if Task.isCancelled {
                for row in rows {
                    finalize(row, decodeEnd: CFAbsoluteTimeGetCurrent(), emitTerminal: true)
                }
                return
            }

            // --- Admit / shrink (epoch boundary): per-row caches are
            //     authoritative here; drop finished/cancelled, pull newcomers. ---
            rows = rows.filter { row in
                if row.cancelled.isCancelled {
                    finalize(row, decodeEnd: CFAbsoluteTimeGetCurrent(), emitTerminal: false)
                    return false
                }
                return true
            }

            let (newcomers, idleExit, didStop) = takePending(
                max: maxRows - rows.count, activeEmpty: rows.isEmpty)
            if didStop {
                for row in rows {
                    finalize(row, decodeEnd: CFAbsoluteTimeGetCurrent(), emitTerminal: true)
                }
                return
            }
            if idleExit { return }   // nothing active, nothing pending: loop stops

            for p in newcomers {
                if p.cancelled.isCancelled { p.cont.finish(); continue }
                if let row = prefillAndAdmit(p) { rows.append(row) }
            }
            if rows.isEmpty { continue }   // all newcomers terminated on first token

            // --- Build the epoch: stack the active set left-padded once. ---
            for row in rows { row.epochBaseLen = row.caches.first?.sequenceLength ?? row.promptLen }
            let sMax = rows.map { $0.epochBaseLen }.max() ?? 0
            let stacked = stackCachesLeftPaddedAny(
                perRow: rows.map { $0.caches }, lengths: rows.map { $0.epochBaseLen })
            let kDtype = stacked.first?.snapshot()?.keys.dtype ?? .float16

            // --- Decode within the epoch until the set changes. ---
            var step = 0
            var cancelledMidEpoch = false
            while true {
                if Task.isCancelled { cancelledMidEpoch = true; break }
                let R = rows.count
                let offsets = rows.map { $0.epochBaseLen + step }
                let totalLen = sMax + step + 1
                let mask = buildBatchedDecodeMask(
                    lengths: rows.map { $0.epochBaseLen }, sMax: sMax,
                    totalLen: totalLen, dtype: kDtype)
                let input = MLXArray(rows.map { Int32($0.current) }).reshaped(R, 1)
                let logits = deps.batchedForward(input, stacked, mask, offsets)
                MLX.eval(logits)
                step += 1

                let now = CFAbsoluteTimeGetCurrent()
                var setChanged = false
                for i in 0 ..< R {
                    let row = rows[i]
                    // Per-row decode clock starts at the row's first batched step,
                    // so decodeTime excludes prefill + admission/epoch wait.
                    if row.decodeStartedAt == nil { row.decodeStartedAt = now }
                    if row.needsHistory { row.recent.append(row.current) }
                    let next = row.sampler.sample(logits[i ..< (i + 1)], recent: row.recent)
                    row.current = next
                    if emit(row, token: next, now: now) { setChanged = true }
                    if row.cancelled.isCancelled { setChanged = true }
                }

                // Break the epoch (scatter + re-stack) when the active set must
                // change: a row finished/cancelled, or a newcomer is waiting.
                if setChanged || !pendingIsEmpty() { break }
            }

            if cancelledMidEpoch {
                // stop() cancelled us: do NOT scatter back (the model is being
                // torn down); just finish every in-flight row's stream.
                for row in rows {
                    finalize(row, decodeEnd: CFAbsoluteTimeGetCurrent(), emitTerminal: true)
                }
                return
            }

            // Scatter the grown stacked cache back into each row's own cache so
            // per-row caches stay authoritative across the next epoch rebuild.
            // scatterBack forces evaluation of the sliced K/V so each row owns
            // realized, independent storage (not a lazy view aliasing `stacked`,
            // which is discarded when the next epoch re-stacks).
            if step > 0 {
                scatterBack(stacked: stacked, rows: rows, sMax: sMax, steps: step)
            }
            // Finished/cancelled rows are dropped at the top of the next iteration.
            rows = rows.filter { !$0.isFinished }
        }
    }

    // MARK: - Helpers

    /// Prefill one pending request (B=1, the unchanged path), emit its first
    /// token, and return an active Row — or nil if the first token already
    /// terminates the row (EOS / stop / maxTokens==0) or the prompt is empty.
    private func prefillAndAdmit(_ p: Pending) -> Row? {
        let start = CFAbsoluteTimeGetCurrent()
        let caches: [KVCacheProtocol] = deps.useQuantizedKV
            ? makeQuantizedKVCaches(numLayers: deps.numLayers)
            : makeKVCaches(numLayers: deps.numLayers)
        // Prefix-cache aware: a full hit restores this prompt's KV and forwards
        // only the last token, yielding KV identical to a cold prefill (so the
        // stacked decode is unchanged); a miss prefills cold and stores
        // write-behind. Honors the row's own `usePrefixCache`.
        let logits = deps.prefillRow(p.promptTokens, caches, p.req.usePrefixCache)
        MLX.eval(logits)
        let firstTok = p.sampler.sample(logits)
        let prefillTime = CFAbsoluteTimeGetCurrent() - start

        let row = Row(
            promptLen: p.promptTokens.count, sampler: p.sampler, cont: p.cont,
            stats: p.stats, cancelled: p.cancelled, maxTokens: p.req.maxTokens,
            caches: caches, current: firstTok,
            recent: p.sampler.needsHistory ? Array(p.promptTokens.suffix(512)) : [],
            prefillTime: prefillTime, admittedAt: CFAbsoluteTimeGetCurrent())

        // Emit the first (prefill-sampled) token, matching the serial path which
        // yields the prefill token before the decode loop.
        if emit(row, token: firstTok, now: CFAbsoluteTimeGetCurrent()) {
            return nil   // first token already ended the row
        }
        return row
    }

    /// Emit one token to a row's stream, applying the serial loop's terminal
    /// semantics (stop-token / maxTokens). Returns true if the row finished.
    private func emit(_ row: Row, token: Int, now: Double) -> Bool {
        if deps.stopIds.contains(token) {
            row.cont.yield(TokenEvent(tokenId: token, text: "",
                                      elapsed: now - row.admittedAt, isEnd: true))
            finalize(row, decodeEnd: now, emitTerminal: false)
            return true
        }
        row.cont.yield(TokenEvent(tokenId: token, text: deps.decode(token),
                                  elapsed: now - row.admittedAt))
        row.generated += 1
        if row.generated >= row.maxTokens {
            row.cont.yield(TokenEvent(tokenId: -1, text: "",
                                      elapsed: now - row.admittedAt, isEnd: true))
            finalize(row, decodeEnd: now, emitTerminal: false)
            return true
        }
        return false
    }

    /// Record stats and finish a row's stream exactly once.
    private func finalize(_ row: Row, decodeEnd: Double, emitTerminal: Bool) {
        guard !row.isFinished else { return }
        if emitTerminal {
            row.cont.yield(TokenEvent(tokenId: -1, text: "",
                                      elapsed: decodeEnd - row.admittedAt, isEnd: true))
        }
        // Decode time runs from the row's first batched step (set in the loop),
        // so it excludes prefill and any admission/epoch wait. A row that
        // finished on its prefill token never decoded, so its decode time is 0.
        let decodeStart = row.decodeStartedAt ?? decodeEnd
        row.stats.stats = GenerationStats(
            promptTokens: row.promptLen, generatedTokens: row.generated,
            prefillTime: row.prefillTime,
            decodeTime: max(0, decodeEnd - decodeStart))
        row.cont.finish()
        row.isFinished = true
    }

    /// Slice each row's own (non-pad) suffix out of the grown stacked cache and
    /// restore it into that row's authoritative per-row cache. After an epoch of
    /// `steps` forwards, row i holds `epochBaseLen[i] + steps` real tokens at the
    /// last that-many columns of the stacked `[R, h, sMax+steps, hd]` tensor.
    private func scatterBack(stacked: [KVCacheProtocol], rows: [Row], sMax: Int, steps: Int) {
        let validEnd = sMax + steps
        // Collect every per-row, per-layer slice, force them all realized in one
        // eval, THEN restore. Without the eval each slice is a lazy view that
        // aliases `stacked`'s storage; the next epoch discards `stacked` and
        // re-stacks from these caches, so unevaluated views would both pin the
        // old stacked tensor (unbounded memory + lazy-graph growth across
        // epochs) and re-derive every row from a shared buffer. Evaluated, each
        // row owns independent, realized K/V.
        //
        // fp16 and int8 are scattered the same way - slice row i's own non-pad
        // suffix out of the grown stacked cache - but the int8 path slices the
        // QUANTIZED storage (uint8 keys/values + fp16 scales/zeros) directly
        // rather than via the dequantizing `snapshot()`, so re-stacking never
        // dequant->requant round-trips (which would compound int8 error each
        // epoch). A KV-shared / empty layer snapshots nil and is skipped, so the
        // row's empty cache stays empty (matching the fp16 contract).
        var fp16Realized: [(dst: KVCache, k: MLXArray, v: MLXArray)] = []
        var quantRealized: [(dst: QuantizedKVCache, snap: QuantizedKVSnapshot)] = []
        for (i, row) in rows.enumerated() {
            let start = sMax - row.epochBaseLen   // left-pad width for row i
            for l in 0 ..< row.caches.count {
                if let q = stacked[l] as? QuantizedKVCache,
                   let dst = row.caches[l] as? QuantizedKVCache {
                    guard let s = q.quantizedSnapshot() else { continue }
                    let sliced = QuantizedKVSnapshot(
                        keys: s.keys[i ..< (i + 1), 0..., start ..< validEnd, 0...],
                        values: s.values[i ..< (i + 1), 0..., start ..< validEnd, 0...],
                        keyScales: s.keyScales[i ..< (i + 1), 0..., start ..< validEnd, 0...],
                        keyZeros: s.keyZeros[i ..< (i + 1), 0..., start ..< validEnd, 0...],
                        valueScales: s.valueScales[i ..< (i + 1), 0..., start ..< validEnd, 0...],
                        valueZeros: s.valueZeros[i ..< (i + 1), 0..., start ..< validEnd, 0...])
                    quantRealized.append((dst, sliced))
                } else if let f = stacked[l] as? KVCache,
                          let dst = row.caches[l] as? KVCache {
                    guard let snap = f.snapshot() else { continue }
                    let k = snap.keys[i ..< (i + 1), 0..., start ..< validEnd, 0...]
                    let v = snap.values[i ..< (i + 1), 0..., start ..< validEnd, 0...]
                    fp16Realized.append((dst, k, v))
                }
            }
        }
        MLX.eval(fp16Realized.flatMap { [$0.k, $0.v] }
            + quantRealized.flatMap {
                [$0.snap.keys, $0.snap.values, $0.snap.keyScales,
                 $0.snap.keyZeros, $0.snap.valueScales, $0.snap.valueZeros]
            })
        for e in fp16Realized { e.dst.restore(keys: e.k, values: e.v) }
        for e in quantRealized { e.dst.restoreQuantized(e.snap) }
    }
}
