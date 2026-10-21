import Foundation

/// Append-only JSONL writer that can never block or fail the caller.
///
/// `append` takes a lock, appends a value-type event to an in-memory queue and (at most) schedules
/// one drain on a utility queue; encoding, file I/O, rotation and directory creation all happen
/// there. The queue is bounded: past `ObsLimits.maxPending` new events are dropped and counted
/// (a `log_dropped` line records the loss) so a stuck disk can never grow memory without bound.
/// `flush()` is the synchronous barrier used at shutdown.
public final class LineWriter: @unchecked Sendable {
    public typealias PathFor = @Sendable (Date) -> URL

    private let pathFor: PathFor
    private let maxFileBytes: Int
    private let queue: DispatchQueue
    private let lock = NSLock()
    private var pending: [LogEvent] = []
    private var scheduled = false
    private var dropped = 0

    // Confined to `queue`.
    private var fd: Int32 = -1
    private var currentPath: String?
    private var currentSize = 0
    private var warned = false

    public init(label: String = "krill.obs.writer", maxFileBytes: Int = ObsLimits.maxFileBytes,
                pathFor: @escaping PathFor) {
        self.pathFor = pathFor
        self.maxFileBytes = maxFileBytes
        self.queue = DispatchQueue(label: label, qos: .utility)
    }

    deinit { if fd >= 0 { close(fd) } }

    /// Queue one event. O(1), allocation-light, never blocks on I/O.
    public func append(_ event: LogEvent) {
        lock.lock()
        if pending.count >= ObsLimits.maxPending {
            dropped += 1
            lock.unlock()
            return
        }
        pending.append(event)
        let needsSchedule = !scheduled
        scheduled = true
        lock.unlock()
        if needsSchedule { queue.async { [self] in drain() } }
    }

    /// Block until everything queued so far is on disk.
    public func flush() {
        queue.sync { drain() }
    }

    /// Events dropped because the queue was full (diagnostics/tests).
    public var droppedCount: Int { lock.lock(); defer { lock.unlock() }; return dropped }

    private func drain() {
        lock.lock()
        let batch = pending
        pending = []
        scheduled = false
        let lost = dropped
        dropped = 0
        lock.unlock()
        var buffer = ""
        var bufferPath: URL?
        func commit() {
            guard let p = bufferPath, !buffer.isEmpty else { return }
            write(buffer, to: p)
            buffer = ""
        }
        for ev in batch {
            let path = pathFor(ev.ts)
            if bufferPath?.path != path.path { commit(); bufferPath = path }
            buffer += JSONLine.encode(ev)
            buffer += "\n"
        }
        if lost > 0 {
            let now = Date()
            let p = pathFor(now)
            if bufferPath?.path != p.path { commit(); bufferPath = p }
            buffer += JSONLine.encode(LogEvent("log_dropped", ts: now, [("count", .int(lost))])) + "\n"
        }
        commit()
    }

    private func write(_ text: String, to url: URL) {
        if currentPath != url.path { reopen(url) }
        // JSON lines never contain a raw newline (it is escaped), so splitting on it is exact.
        // Rotate between lines so no file passes the cap by more than one line.
        var group: [UInt8] = []
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            var bytes = Array(line.utf8)
            bytes.append(0x0A)
            if currentSize + group.count > 0, currentSize + group.count + bytes.count > maxFileBytes {
                emit(group); group.removeAll(keepingCapacity: true)
                rotate(url)
            }
            group.append(contentsOf: bytes)
        }
        emit(group)
    }

    private func emit(_ bytes: [UInt8]) {
        guard fd >= 0, !bytes.isEmpty else { return }
        var off = 0
        while off < bytes.count {
            let n = bytes[off...].withUnsafeBufferPointer { Foundation.write(fd, $0.baseAddress, $0.count) }
            if n <= 0 { warnOnce("write failed (errno \(errno))"); break }
            off += n
        }
        currentSize += off
    }

    private func reopen(_ url: URL) {
        if fd >= 0 { close(fd); fd = -1 }
        currentPath = url.path
        ObsPaths.ensureDir(url.deletingLastPathComponent())
        let f = open(url.path, O_WRONLY | O_APPEND | O_CREAT, 0o600)
        if f < 0 { warnOnce("cannot open \(url.path) (errno \(errno))"); currentSize = 0; return }
        fd = f
        var st = stat()
        currentSize = fstat(f, &st) == 0 ? Int(st.st_size) : 0
    }

    /// `serve-2026-10-22.jsonl` -> `serve-2026-10-22.1.jsonl` (first free N), then start a new file.
    private func rotate(_ url: URL) {
        if fd >= 0 { close(fd); fd = -1 }
        let dir = url.deletingLastPathComponent()
        let stem = url.deletingPathExtension().lastPathComponent
        let ext = url.pathExtension
        var n = 1
        while FileManager.default.fileExists(atPath: dir.appendingPathComponent("\(stem).\(n).\(ext)").path) { n += 1 }
        try? FileManager.default.moveItem(at: url, to: dir.appendingPathComponent("\(stem).\(n).\(ext)"))
        currentPath = nil
        reopen(url)
    }

    private func warnOnce(_ msg: String) {
        guard !warned else { return }
        warned = true
        FileHandle.standardError.write(Data("krill: log writer disabled for this file: \(msg)\n".utf8))
    }
}
