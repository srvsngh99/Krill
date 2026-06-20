import Foundation

/// Process-wide serializer for model generation. The in-process path has no
/// continuous batcher (that lives in the HTTP server), so two concurrent
/// `engine.generate` calls on one model would race on the GPU. This gate makes
/// every generation - the foreground chat turn and every background agent turn -
/// acquire a single ticket, so decodes run one at a time, FIFO.
///
/// Background agents therefore still exist, progress turn-by-turn, and are
/// switchable; their decodes simply do not overlap (which on a single-GPU box
/// would not help throughput anyway). The fair FIFO queue keeps an interactive
/// chat turn from being starved by a busy agent loop.
final class GenerationGate: @unchecked Sendable {
    static let shared = GenerationGate()

    private let lock = NSLock()
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    /// Suspend until the gate is free, then hold it. Pair with exactly one
    /// `release()`. Resolves immediately when uncontended.
    func acquire() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            enqueue(c)
        }
    }

    private func enqueue(_ c: CheckedContinuation<Void, Never>) {
        lock.lock(); defer { lock.unlock() }
        if !busy {
            busy = true
            c.resume()          // uncontended: no real suspension
        } else {
            waiters.append(c)   // queued; resumed by a later release()
        }
    }

    /// Release the gate, handing it to the next waiter (FIFO) or marking it free.
    /// Synchronous so it is safe in a `defer`.
    func release() {
        lock.lock()
        var next: CheckedContinuation<Void, Never>?
        if waiters.isEmpty { busy = false } else { next = waiters.removeFirst() }
        lock.unlock()
        next?.resume()
    }
}
