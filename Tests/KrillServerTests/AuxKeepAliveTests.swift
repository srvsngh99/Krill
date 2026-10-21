import XCTest
import Foundation
@testable import KrillServer
@testable import KrillEngine

/// Idle-unload policy for the embedding / rerank slots: same keep-alive
/// semantics as chat models (positive TTL, 0 = unload after the request,
/// negative = pinned), never while a request is in flight.
final class AuxKeepAliveTests: XCTestCase {

    func testExpiredIdleSlotIsEvictedAndRearmed() async {
        let aux = AuxKeepAlive(defaultSeconds: 20)
        let now = Date()
        var ev = await aux.embed.shouldEvict(now: now)
        XCTAssertFalse(ev)
        ev = await aux.embed.shouldEvict(now: now.addingTimeInterval(21))
        XCTAssertTrue(ev)
        // nothing resident: nothing reported, state reset
        let names = await aux.evictExpired(embedEngine: EmbeddingEngine(), rerankEngine: RerankEngine(),
                                           now: now.addingTimeInterval(30))
        XCTAssertTrue(names.isEmpty)
        let after = await aux.embed.shouldEvict(now: now.addingTimeInterval(1000))
        XCTAssertFalse(after, "markEvicted clears the stale deadline")
    }

    func testNeverEvictsWhileInFlight() async {
        let aux = AuxKeepAlive(defaultSeconds: 1)
        await aux.rerank.beginRequest()
        let ev = await aux.rerank.shouldEvict(now: Date().addingTimeInterval(100))
        XCTAssertFalse(ev)
        await aux.rerank.endRequest()
        let ev2 = await aux.rerank.shouldEvict(now: Date().addingTimeInterval(100))
        XCTAssertTrue(ev2)
    }

    func testKeepAliveZeroEvictsAfterDrainAndNegativePins() async {
        let aux = AuxKeepAlive(defaultSeconds: 300)
        await aux.embed.beginRequest()
        await aux.embed.endRequest()
        await aux.embed.touch(override: 0)
        let zero = await aux.embed.shouldEvict()
        XCTAssertTrue(zero)
        await aux.embed.touch(override: -1)
        let pinned = await aux.embed.shouldEvict(now: Date().addingTimeInterval(1_000_000))
        XCTAssertFalse(pinned)
        await aux.embed.touch(override: 5)
        let five = await aux.embed.shouldEvict(now: Date().addingTimeInterval(6))
        XCTAssertTrue(five)
    }
}

/// Check-and-unload atomicity and independent tower clocks.
final class AuxKeepAliveRaceTests: XCTestCase {
    private final class Slot: @unchecked Sendable {
        private let lock = NSLock()
        private var resident = true
        func unload() { lock.lock(); resident = false; lock.unlock() }
        /// What `load()` does: no-op when resident, else reload.
        func ensureLoaded() { lock.lock(); resident = true; lock.unlock() }
        var isResident: Bool { lock.lock(); defer { lock.unlock() }; return resident }
    }

    /// A request that has begun must never see its model unloaded under it, however the
    /// evictor interleaves. (With a separate shouldEvict() then unload() this fails.)
    func testEvictIfDueNeverUnloadsUnderABegunRequest() async {
        for _ in 0 ..< 300 {
            let c = KeepAliveController(defaultSeconds: 0)
            await c.touch(override: 0)          // due: nothing in flight
            let slot = Slot()
            async let evict: Bool = c.evictIfDue { slot.unload() }
            async let request: Bool = {
                await c.beginRequest()
                slot.ensureLoaded()              // load(): reloads only if the evictor won
                await Task.yield()
                let held = slot.isResident       // embed() reads the model here
                await c.endRequest()
                return held
            }()
            let (_, held) = await (evict, request)
            XCTAssertTrue(held, "request lost its model between load() and embed()")
        }
    }

    func testEvictIfDueResetsAndRespectsInFlight() async {
        let c = KeepAliveController(defaultSeconds: 1)
        await c.beginRequest()
        var ran = false
        var due = await c.evictIfDue(now: Date().addingTimeInterval(100)) { ran = true }
        XCTAssertFalse(due); XCTAssertFalse(ran)
        await c.endRequest()
        due = await c.evictIfDue(now: Date().addingTimeInterval(100)) { ran = true }
        XCTAssertTrue(due); XCTAssertTrue(ran)
        due = await c.evictIfDue(now: Date().addingTimeInterval(200)) { XCTFail("deadline was cleared") }
        XCTAssertFalse(due)
    }

    func testTowerClocksAreIndependentOfTextAndEachOther() async {
        let aux = AuxKeepAlive(defaultSeconds: 300)
        let later = Date().addingTimeInterval(400)
        await aux.embed.touch(override: 10_000)      // text tower stays warm
        await aux.vision.touch(override: 1)          // an image was served once
        await aux.audio.touch(override: 10_000)
        let text = await aux.embed.shouldEvict(now: later)
        let vis = await aux.vision.shouldEvict(now: later)
        let aud = await aux.audio.shouldEvict(now: later)
        XCTAssertFalse(text); XCTAssertTrue(vis); XCTAssertFalse(aud)
        let engine = EmbeddingEngine()
        XCTAssertFalse(engine.unloadVisionTower())   // nothing resident: no-op
        XCTAssertFalse(engine.unloadAudioTower())
        let names = await aux.evictExpired(embedEngine: engine, rerankEngine: RerankEngine(), now: later)
        XCTAssertTrue(names.isEmpty)
    }

    func testMediaKindsSelectTowers() {
        let img = EmbeddingInput(parts: [.text("a"), .image(Data([1]))])
        let aud = EmbeddingInput(parts: [.audio(Data([1]), format: "")])
        XCTAssertTrue(img.usesVision); XCTAssertFalse(img.usesAudio)
        XCTAssertTrue(aud.usesAudio); XCTAssertFalse(aud.usesVision)
        XCTAssertFalse(EmbeddingInput(text: "x").usesVision)
    }
}
