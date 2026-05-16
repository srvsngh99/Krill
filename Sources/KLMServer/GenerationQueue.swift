import Foundation

/// Serializes inference so concurrent clients are *queued, not dropped*
/// (WS-E / T1-5), and the single-flight engine + persistent prefix/int8-KV
/// caches are never entered concurrently (a real correctness guard — the
/// caches assume single-flight today).
///
/// `numParallel` slots run at once (default 1 = fully serialized, matching
/// Ollama's per-model default). Up to `maxQueue` requests may wait; beyond
/// that `enter()` throws ``QueueFull`` and the handler returns HTTP 503.
/// True batched execution across slots is a tracked follow-up — the parity
/// plan's WS-E acceptance explicitly permits serialized-first.
public actor GenerationQueue {
    public struct QueueFull: Error {}

    private let numParallel: Int
    private let maxQueue: Int
    private var active = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init(numParallel: Int = 1, maxQueue: Int = 512) {
        self.numParallel = max(1, numParallel)
        self.maxQueue = max(0, maxQueue)
    }

    /// Current queue depth (waiting requests), for diagnostics/tests.
    public var depth: Int { waiters.count }

    public func enter() async throws {
        if active < numParallel {
            active += 1
            return
        }
        if waiters.count >= maxQueue {
            throw QueueFull()
        }
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            waiters.append(c)
        }
        // Resumed by a leave(); the slot was handed directly to us.
    }

    public func leave() {
        if let next = waiters.first {
            waiters.removeFirst()
            next.resume()          // hand the slot straight to the next waiter
        } else {
            active = max(0, active - 1)
        }
    }

    /// Run `body` holding a slot; the slot is always released, even on
    /// throw/cancel, so the queue can never deadlock.
    public func withSlot<T>(_ body: () async throws -> T) async throws -> T {
        try await enter()
        defer { leave() }
        return try await body()
    }
}
