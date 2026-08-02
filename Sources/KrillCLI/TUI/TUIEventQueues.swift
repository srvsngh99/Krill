import Foundation
import KrillHarness

/// Thread-safe hand-off of agent events from a background run to the main TUI
/// render loop. Keeping this synchronization primitive separate prevents the
/// already-large view controller from owning low-level queue mechanics.
final class EventQueue: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [AgentEvent] = []
    private var done = false
    private var result: AgentTranscript?

    func push(_ event: AgentEvent) {
        lock.lock()
        events.append(event)
        lock.unlock()
    }

    func markDone() {
        lock.lock()
        done = true
        lock.unlock()
    }

    func finish(_ transcript: AgentTranscript) {
        lock.lock()
        result = transcript
        done = true
        lock.unlock()
    }

    func drain() -> [AgentEvent] {
        lock.lock()
        defer { lock.unlock() }
        let drained = events
        events.removeAll()
        return drained
    }

    var isFinished: Bool {
        lock.lock()
        defer { lock.unlock() }
        return done && events.isEmpty
    }

    var finishedResult: AgentTranscript? {
        lock.lock()
        defer { lock.unlock() }
        return result
    }
}

/// Thread-safe hand-off of deep-research progress and its final report to the
/// render loop.
final class ResearchProgressQueue: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [DeepResearch.Progress] = []
    private var done = false
    private var report: DeepResearch.Report?

    func push(_ progress: DeepResearch.Progress) {
        lock.lock()
        items.append(progress)
        lock.unlock()
    }

    func finish(_ finalReport: DeepResearch.Report) {
        lock.lock()
        report = finalReport
        done = true
        lock.unlock()
    }

    func drain() -> [DeepResearch.Progress] {
        lock.lock()
        defer { lock.unlock() }
        let drained = items
        items.removeAll()
        return drained
    }

    var isFinished: Bool {
        lock.lock()
        defer { lock.unlock() }
        return done && items.isEmpty
    }
}
