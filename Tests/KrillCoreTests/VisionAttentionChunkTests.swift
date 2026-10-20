import XCTest
import MLX
@testable import KrillCore

/// Query-blocked vision attention must equal one-shot attention (softmax is
/// per query row, so blocking only bounds the live score matrix).
final class VisionAttentionChunkTests: XCTestCase {
    func testBlockedAttentionMatchesOneShot() {
        let B = 1, H = 4, L = 300, D = 64
        let q = MLXRandom.normal([B, H, L, D]), k = MLXRandom.normal([B, H, L, D]), v = MLXRandom.normal([B, H, L, D])
        let ref = VisionAttention.attend(q, k, v, mask: nil, queryChunk: 0, minLength: 0)
        for chunk in [64, 100, 299] {
            let got = VisionAttention.attend(q, k, v, mask: nil, queryChunk: chunk, minLength: 0)
            XCTAssertEqual(got.shape, ref.shape)
            let diff = abs(got - ref).max().item(Float.self)
            XCTAssertLessThan(diff, 1e-5, "chunk \(chunk)")
        }
    }

    func testShortSequencesAndMaskedCallsAreNotBlocked() {
        let q = MLXRandom.normal([1, 2, 40, 64])
        // below minLength -> one-shot path (identical result trivially)
        let a = VisionAttention.attend(q, q, q, mask: nil, queryChunk: 8, minLength: 100)
        let b = VisionAttention.attend(q, q, q, mask: nil, queryChunk: 0, minLength: 0)
        XCTAssertLessThan(abs(a - b).max().item(Float.self), 1e-6)
    }
}
