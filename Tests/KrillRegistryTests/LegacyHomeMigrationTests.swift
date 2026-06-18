import XCTest
@testable import KrillRegistry

/// Covers the one-time KrillLM → Krill home migration: pre-rename installs kept
/// everything under `~/.krillm`; the registry must move that tree to `~/.krill`
/// on first use so models/cache/config survive the rebrand without a re-download.
final class LegacyHomeMigrationTests: XCTestCase {
    private func makeTempHome() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-migrate-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func testMigratesLegacyHomeWhenNewAbsent() throws {
        let home = try makeTempHome()
        defer { try? FileManager.default.removeItem(at: home) }
        let fm = FileManager.default
        let legacy = home.appendingPathComponent(".krillm")
        let blob = legacy.appendingPathComponent("models/blobs")
        try fm.createDirectory(at: blob, withIntermediateDirectories: true)
        let weight = blob.appendingPathComponent("weight.safetensors")
        try Data([0xCA, 0xFE]).write(to: weight)

        let moved = Registry.migrateLegacyHomeIfNeeded(home: home)

        XCTAssertTrue(moved)
        let new = home.appendingPathComponent(".krill")
        XCTAssertTrue(fm.fileExists(atPath: new.path))
        XCTAssertFalse(fm.fileExists(atPath: legacy.path), "legacy dir should be gone after move")
        XCTAssertEqual(
            try Data(contentsOf: new.appendingPathComponent("models/blobs/weight.safetensors")),
            Data([0xCA, 0xFE]),
            "migrated weights must be byte-identical")
    }

    func testNoOpWhenNewHomeAlreadyExists() throws {
        let home = try makeTempHome()
        defer { try? FileManager.default.removeItem(at: home) }
        let fm = FileManager.default
        // Both present: never clobber an existing ~/.krill.
        try fm.createDirectory(at: home.appendingPathComponent(".krillm"), withIntermediateDirectories: true)
        let new = home.appendingPathComponent(".krill")
        try fm.createDirectory(at: new, withIntermediateDirectories: true)
        try Data([0x01]).write(to: new.appendingPathComponent("keep"))

        let moved = Registry.migrateLegacyHomeIfNeeded(home: home)

        XCTAssertFalse(moved)
        XCTAssertTrue(fm.fileExists(atPath: new.appendingPathComponent("keep").path))
    }

    func testNoOpWhenNothingToMigrate() throws {
        let home = try makeTempHome()
        defer { try? FileManager.default.removeItem(at: home) }
        XCTAssertFalse(Registry.migrateLegacyHomeIfNeeded(home: home))
    }

    func testIdempotentSecondCall() throws {
        let home = try makeTempHome()
        defer { try? FileManager.default.removeItem(at: home) }
        try FileManager.default.createDirectory(
            at: home.appendingPathComponent(".krillm"), withIntermediateDirectories: true)
        XCTAssertTrue(Registry.migrateLegacyHomeIfNeeded(home: home))
        XCTAssertFalse(Registry.migrateLegacyHomeIfNeeded(home: home), "second call is a no-op")
    }
}
