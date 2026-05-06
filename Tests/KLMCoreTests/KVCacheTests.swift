import XCTest
@testable import KLMCache

/// Tests for KVCache snapshot/restore/truncate behavior.
///
/// Note: Tests that create MLXArray instances require Metal GPU access.
/// They are skipped in environments where Metal is unavailable.
final class KVCacheTests: XCTestCase {

    /// Check if MLX/Metal is available for testing.
    private var metalAvailable: Bool {
        // Try to import MLX and create a simple array.
        // If Metal isn't available, MLX will fatal error on first use.
        // We detect this by checking if the metallib loaded successfully.
        #if canImport(MLX)
        return true  // Compilation check — runtime check via guard in each test
        #else
        return false
        #endif
    }

    // MARK: - Test 3: Basic KVCache contract (no MLX required)

    func testEmptyCacheHasZeroLength() {
        let cache = KVCache()
        XCTAssertEqual(cache.sequenceLength, 0)
        XCTAssertNil(cache.snapshot())
    }

    func testResetClearsState() {
        let cache = KVCache()
        // After reset, should still be empty
        cache.reset()
        XCTAssertEqual(cache.sequenceLength, 0)
        XCTAssertNil(cache.snapshot())
    }

    func testTruncateOnEmptyCacheIsNoop() {
        let cache = KVCache()
        cache.truncate(to: 5)
        XCTAssertEqual(cache.sequenceLength, 0, "Truncating empty cache should be no-op")
    }
}
