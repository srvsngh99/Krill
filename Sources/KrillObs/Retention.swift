import Foundation

public struct PruneResult: Sendable, Equatable {
    public var deletedFiles = 0
    public var deletedBytes = 0
    public var remainingBytes = 0
}

/// Age and size retention for `logs/` and `sessions/`. One directory walk, a handful of stat
/// calls per file; the work is skipped entirely if it ran within the last hour.
public enum Retention {
    /// Do not touch files written to in the last 10 minutes (they may be open).
    static let activeWindow: TimeInterval = 600

    public static func pruneIfDue(settings: ObsSettings, now: Date = Date(),
                                  minInterval: TimeInterval = 3600) {
        let stamp = settings.logsDir.appendingPathComponent(".pruned")
        if let m = (try? stamp.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate,
           now.timeIntervalSince(m) < minInterval { return }
        _ = prune(settings: settings, now: now)
        ObsPaths.ensureDir(settings.logsDir)
        try? Data().write(to: stamp)
    }

    @discardableResult
    public static func prune(settings: ObsSettings, now: Date = Date()) -> PruneResult {
        struct Entry { var url: URL; var mtime: Date; var size: Int; var isSession: Bool }
        let fm = FileManager.default
        var entries: [Entry] = []
        func scan(_ dir: URL, isSession: Bool) {
            guard let en = fm.enumerator(at: dir, includingPropertiesForKeys:
                [.contentModificationDateKey, .fileSizeKey, .isRegularFileKey]) else { return }
            for case let u as URL in en {
                guard u.pathExtension == "jsonl", u.lastPathComponent != "index.jsonl",
                      let v = try? u.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey, .isRegularFileKey]),
                      v.isRegularFile == true else { continue }
                entries.append(Entry(url: u, mtime: v.contentModificationDate ?? now, size: v.fileSize ?? 0, isSession: isSession))
            }
        }
        scan(settings.logsDir, isSession: false)
        scan(settings.sessionsDir, isSession: true)

        var result = PruneResult()
        var deletedSession = false
        func delete(_ e: Entry) {
            if (try? fm.removeItem(at: e.url)) != nil {
                result.deletedFiles += 1
                result.deletedBytes += e.size
                if e.isSession { deletedSession = true }
            }
        }
        let cutoff = now.addingTimeInterval(-Double(max(1, settings.retentionDays)) * 86_400)
        var kept: [Entry] = []
        for e in entries {
            if e.mtime < cutoff, now.timeIntervalSince(e.mtime) > activeWindow { delete(e) } else { kept.append(e) }
        }
        let indexSize = (try? settings.sessionsDir.appendingPathComponent("index.jsonl")
            .resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
        var total = kept.reduce(indexSize) { $0 + $1.size }
        let cap = max(1, settings.maxMB) * 1_048_576
        if total > cap {
            for e in kept.sorted(by: { $0.mtime < $1.mtime }) {
                if total <= cap { break }
                if now.timeIntervalSince(e.mtime) <= activeWindow { continue }
                delete(e)
                total -= e.size
            }
        }
        result.remainingBytes = max(0, total)
        if deletedSession {
            SessionIndex.compact(settings: settings)
            removeEmptyDirs(settings.sessionsDir)
        }
        return result
    }

    static func removeEmptyDirs(_ root: URL) {
        let fm = FileManager.default
        guard let years = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey]) else { return }
        for y in years where (try? y.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
            for m in (try? fm.contentsOfDirectory(at: y, includingPropertiesForKeys: nil)) ?? [] {
                for d in (try? fm.contentsOfDirectory(at: m, includingPropertiesForKeys: nil)) ?? [] {
                    if ((try? fm.contentsOfDirectory(atPath: d.path))?.isEmpty ?? false) { try? fm.removeItem(at: d) }
                }
                if ((try? fm.contentsOfDirectory(atPath: m.path))?.isEmpty ?? false) { try? fm.removeItem(at: m) }
            }
            if ((try? fm.contentsOfDirectory(atPath: y.path))?.isEmpty ?? false) { try? fm.removeItem(at: y) }
        }
    }
}
