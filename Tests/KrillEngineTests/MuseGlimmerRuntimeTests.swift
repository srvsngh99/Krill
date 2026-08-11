import XCTest
@testable import KrillEngine

/// Checkpoint-free tests for the Muse Glimmer image driver's geometry contract.
final class MuseGlimmerRuntimeTests: XCTestCase {

    /// The placeholder run must be exactly the MERGED token count: the projector
    /// pixel-shuffles `merge x merge` patches into one LM token, so a 4x6 patch
    /// grid at merge 2 is 6 placeholders, not 24. Getting this wrong is not a
    /// shape error at the tower — it surfaces as a span mismatch during the
    /// embed splice, one layer removed from its cause.
    func testPlaceholderCountIsTheMergedGrid() {
        XCTAssertEqual(
            MuseGlimmerRuntime.placeholderCount(grid: (t: 1, h: 4, w: 6), mergeSize: 2), 6)
        XCTAssertEqual(
            MuseGlimmerRuntime.placeholderCount(grid: (t: 1, h: 32, w: 32), mergeSize: 2), 256)
    }

    /// Temporal frames multiply the count; merge 1 is the identity.
    func testPlaceholderCountHandlesFramesAndIdentityMerge() {
        XCTAssertEqual(
            MuseGlimmerRuntime.placeholderCount(grid: (t: 2, h: 4, w: 4), mergeSize: 2), 8)
        XCTAssertEqual(
            MuseGlimmerRuntime.placeholderCount(grid: (t: 1, h: 3, w: 3), mergeSize: 1), 9)
    }

    /// A zero/negative merge would divide by zero. Clamp rather than trap: a
    /// malformed config should not take the process down.
    func testPlaceholderCountClampsDegenerateMerge() {
        XCTAssertEqual(
            MuseGlimmerRuntime.placeholderCount(grid: (t: 1, h: 4, w: 4), mergeSize: 0), 16)
    }
}
