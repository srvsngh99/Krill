import XCTest
import MLX
@testable import KrillCore

/// Unit tests for `createSlidingWindowCausalMask` (Gemma sliding-window layers).
/// A query at absolute position `q` may attend key `k` iff causal (`k <= q`) AND
/// inside the window (`q - k < window`).
final class SlidingWindowMaskTests: XCTestCase {

    private func grid(_ m: MLXArray) -> [[Float]] {
        // [1,1,R,C] additive mask -> [R][C] of 0 (allowed) / negative (masked).
        let r = m.dim(2), c = m.dim(3)
        let flat = m.reshaped(r * c).asType(.float32).asArray(Float.self)
        return (0..<r).map { i in Array(flat[(i * c)..<((i + 1) * c)]) }
    }

    func testWindowedPatternIsCausalAndBounded() {
        // newLen=4, cacheLen=0, window=2 -> query i sees keys [i-1, i].
        let m = try! XCTUnwrap(createSlidingWindowCausalMask(
            newLen: 4, cacheLen: 0, window: 2, dtype: .float32))
        XCTAssertEqual(m.shape, [1, 1, 4, 4])
        let g = grid(m)
        func allowed(_ i: Int, _ j: Int) -> Bool { g[i][j] == 0 }
        // Causal: no query attends a future key.
        for i in 0..<4 { for j in (i + 1)..<4 { XCTAssertFalse(allowed(i, j), "future \(i),\(j)") } }
        // Window=2: query i attends exactly j in {i-1, i} (clamped at 0).
        XCTAssertEqual(g.map { row in (0..<4).filter { row[$0] == 0 } },
                       [[0], [0, 1], [1, 2], [2, 3]])
    }

    func testDelegatesToPlainCausalWhenWindowDoesNotBite() {
        // total (3) <= window (10): identical to the plain causal mask.
        let sliding = try! XCTUnwrap(createSlidingWindowCausalMask(
            newLen: 3, cacheLen: 0, window: 10, dtype: .float32))
        let plain = try! XCTUnwrap(createCachedCausalMask(
            newLen: 3, cacheLen: 0, dtype: .float32))
        XCTAssertEqual(grid(sliding), grid(plain), "short prompt must equal plain causal")
        // A single token within the window needs no mask at all.
        XCTAssertNil(createSlidingWindowCausalMask(newLen: 1, cacheLen: 5, window: 512))
    }

    func testDecodeStepWindowsOldKeys() {
        // newLen=1, cacheLen=600, window=512: query abs 600 attends keys (88, 600].
        let m = try! XCTUnwrap(createSlidingWindowCausalMask(
            newLen: 1, cacheLen: 600, window: 512, dtype: .float32))
        XCTAssertEqual(m.shape, [1, 1, 1, 601])
        let row = grid(m)[0]
        XCTAssertLessThan(row[0], 0, "key 0 is too old, must be masked")
        XCTAssertLessThan(row[88], 0, "key 88 (600-88=512, not < window) must be masked")
        XCTAssertEqual(row[89], 0, "key 89 (600-89=511 < 512) must be allowed")
        XCTAssertEqual(row[600], 0, "key 600 (self) must be allowed")
        XCTAssertEqual(row.filter { $0 == 0 }.count, 512, "exactly window keys visible")
    }
}
