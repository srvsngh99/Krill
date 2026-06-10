import XCTest
import MLX
@testable import KLMEngine
@testable import KLMCache

/// Regression gate for the cross-path prefix-cache corruption found in PR #194
/// review: the serial rotating path stores sliding-layer snapshots as the
/// WINDOW TAIL only (span < prompt length), but consumers using full-history
/// `KVCache`s (the batched/concurrent per-row prefill, or a serial run with
/// `KRILL_ROTATING_KV=0`) would adopt that span at the wrong length - the
/// follow-up `truncate(to: count-1)` no-ops and the last token forwards at the
/// wrong RoPE position. `spanCanSeed` is the guard: a trimmed span may seed a
/// `RotatingKVCache` (which places it at its absolute position) but NEVER a
/// full-history cache, which must decline the hit and prefill cold.
final class PrefixSpanGuardTests: XCTestCase {

    private func hit(layers: Int, spanLen: Int, prefixLength: Int,
                     storedLength: Int) -> PrefixCacheHit {
        let span = MLXArray.zeros([1, 2, spanLen, 4], dtype: .float16)
        return PrefixCacheHit(
            keys: Array(repeating: [span], count: layers),
            values: Array(repeating: [span], count: layers),
            prefixLength: prefixLength,
            storedLength: storedLength)
    }

    func testFullWidthSpanSeedsAnyCache() {
        // A full-width span (old-format entry, or any standard-layer span)
        // seeds both cache kinds.
        XCTAssertTrue(InferenceEngine.spanCanSeed(
            cache: KVCache(), spanLength: 2000, storedLength: 2000))
        XCTAssertTrue(InferenceEngine.spanCanSeed(
            cache: RotatingKVCache(window: 1024), spanLength: 2000, storedLength: 2000))
    }

    func testTrimmedSpanSeedsOnlyRotatingCache() {
        // A window-trimmed span (serial rotating store, prompt > window) must
        // seed ONLY a rotating cache. A full-history cache adopting it would
        // sit at length `window` instead of `storedLength` - the corruption
        // this guard exists to prevent.
        XCTAssertTrue(InferenceEngine.spanCanSeed(
            cache: RotatingKVCache(window: 1024), spanLength: 1023, storedLength: 2000))
        XCTAssertFalse(InferenceEngine.spanCanSeed(
            cache: KVCache(), spanLength: 1023, storedLength: 2000))
    }

    func testSpansCanSeedDeclinesMixedHitOnStandardCaches() {
        // Batched-path shape: ALL standard caches, hit carries a trimmed span
        // on (sliding) layer 1 -> the whole hit must be declined.
        let caches: [KVCache] = [KVCache(), KVCache()]
        let trimmed = PrefixCacheHit(
            keys: [[MLXArray.zeros([1, 2, 2000, 4], dtype: .float16)],
                   [MLXArray.zeros([1, 2, 1023, 4], dtype: .float16)]],
            values: [[MLXArray.zeros([1, 2, 2000, 4], dtype: .float16)],
                     [MLXArray.zeros([1, 2, 1023, 4], dtype: .float16)]],
            prefixLength: 2000, storedLength: 2000)
        XCTAssertFalse(InferenceEngine.spansCanSeed(hit: trimmed, caches: caches))
    }

    func testSpansCanSeedAcceptsTrimmedHitOnMatchingSpec() {
        // Serial rotating shape: layer 0 standard (full attention, full span),
        // layer 1 rotating (sliding, trimmed span) -> accepted.
        let caches: [RestorableKVCache] = [KVCache(), RotatingKVCache(window: 1024)]
        let trimmed = PrefixCacheHit(
            keys: [[MLXArray.zeros([1, 2, 2000, 4], dtype: .float16)],
                   [MLXArray.zeros([1, 2, 1023, 4], dtype: .float16)]],
            values: [[MLXArray.zeros([1, 2, 2000, 4], dtype: .float16)],
                     [MLXArray.zeros([1, 2, 1023, 4], dtype: .float16)]],
            prefixLength: 2000, storedLength: 2000)
        XCTAssertTrue(InferenceEngine.spansCanSeed(hit: trimmed, caches: caches))
    }

    func testSpansCanSeedSkipsLayersTheHitDoesNotCover() {
        // Gemma KV-shared shape: the hit covers fewer layers than the cache
        // array (shared suffix layers store nothing) - uncovered layers pass.
        let caches: [KVCache] = [KVCache(), KVCache(), KVCache()]
        let h = hit(layers: 2, spanLen: 500, prefixLength: 500, storedLength: 500)
        XCTAssertTrue(InferenceEngine.spansCanSeed(hit: h, caches: caches))
    }
}
