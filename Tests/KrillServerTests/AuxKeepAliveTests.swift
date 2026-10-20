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
