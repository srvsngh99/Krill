import XCTest
@testable import KrillObs

class ObsTestCase: XCTestCase {
    var tmp: URL!
    var settings: ObsSettings!

    override func setUp() {
        super.setUp()
        tmp = FileManager.default.temporaryDirectory.appendingPathComponent("krill-obs-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        settings = ObsSettings(eventLog: true, sessionLog: .full, retentionDays: 30, maxMB: 1024, root: tmp)
        Obs.resetForTesting()
    }

    override func tearDown() {
        Obs.resetForTesting()
        try? FileManager.default.removeItem(at: tmp)
        super.tearDown()
    }

    func lines(_ url: URL) -> [String] {
        ((try? String(contentsOf: url, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init)
    }

    func json(_ line: String) -> [String: Any] {
        ((try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any]) ?? [:]
    }

    func files(_ dir: URL, ext: String = "jsonl") -> [URL] {
        let en = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil)
        var out: [URL] = []
        while let u = en?.nextObject() as? URL { if u.pathExtension == ext { out.append(u) } }
        return out.sorted { $0.path < $1.path }
    }

    func touch(_ url: URL, daysAgo: Double) {
        let d = Date().addingTimeInterval(-daysAgo * 86_400)
        try? FileManager.default.setAttributes([.modificationDate: d], ofItemAtPath: url.path)
    }

    func write(_ url: URL, _ text: String) {
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? text.write(to: url, atomically: true, encoding: .utf8)
    }
}
