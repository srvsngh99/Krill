import XCTest
import MLX
import MLXFast
@testable import KrillCore

/// The long-input attention path (query blocks; sliding layers read only their
/// +-window keys; no dense L x L mask) must equal the single-call dense path,
/// and the per-layer PLE slice must equal the full projection.
final class EmbeddingGemma2BlockedAttentionTests: XCTestCase {
    private func dense(q: MLXArray, k: MLXArray, v: MLXArray, keyMask: MLXArray?, window: Int?) -> MLXArray {
        let T = q.dim(2)
        var mask = keyMask
        if let w = window {
            let i = MLXArray(0 ..< Int32(T)).reshaped(T, 1), j = MLXArray(0 ..< Int32(T)).reshaped(1, T)
            let band = MLX.where(MLX.abs(i - j) .<= MLXArray(Int32(w)), MLXArray(Float(0)), MLXArray(Float(-1e9)))
                .reshaped(1, 1, T, T)
            mask = mask.map { $0 + band } ?? band
        }
        return MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: mask.map { .array($0) } ?? .none)
    }

    func testBlockedMatchesDenseForSlidingFullAndPadding() {
        let B = 2, H = 4, T = 83, D = 32
        let q = MLXRandom.normal([B, H, T, D]), k = MLXRandom.normal([B, 2, T, D]), v = MLXRandom.normal([B, 2, T, D])
        // GQA is handled inside the kernel; give the dense reference the same shapes.
        let lengths: [Int32] = [83, 51]
        let pos = MLXArray(0 ..< Int32(T)).reshaped(1, T)
        let keyMask = MLX.where(pos .< MLXArray(lengths).reshaped(B, 1), MLXArray(Float(0)), MLXArray(Float(-1e9)))
            .reshaped(B, 1, 1, T)
        for window in [nil, 9, 200] as [Int?] {
            for km in [nil, keyMask] as [MLXArray?] {
                let ref = dense(q: q, k: k, v: v, keyMask: km, window: window)
                for block in [16, 40, 82] {
                    let got = EG2Attention.blockedAttention(q: q, k: k, v: v, keyMask: km, window: window, block: block)
                    XCTAssertEqual(got.shape, ref.shape)
                    // padded query rows are discarded by the pooling, so compare the real ones
                    for b in 0 ..< B {
                        let n = km == nil ? T : Int(lengths[b])
                        let d = abs(got[b, 0..., 0 ..< n, 0...] - ref[b, 0..., 0 ..< n, 0...]).max().item(Float.self)
                        XCTAssertLessThan(d, 1e-4, "window \(String(describing: window)) block \(block) masked \(km != nil)")
                    }
                }
            }
        }
    }

    func testBlockedPathOnlyForLongInputs() {
        XCTAssertFalse(EG2Attention.usesBlockedAttention(length: 512, block: 512, minLength: 1024))
        XCTAssertFalse(EG2Attention.usesBlockedAttention(length: 1023, block: 512, minLength: 1024))
        XCTAssertTrue(EG2Attention.usesBlockedAttention(length: 1024, block: 512, minLength: 1024))
        XCTAssertFalse(EG2Attention.usesBlockedAttention(length: 8192, block: 0, minLength: 1024))
    }

    func testLayerSlicedPLEMatchesFullProjection() {
        let cfg = try! JSONDecoder().decode(EmbeddingGemma2Config.self, from: Data(Self.tinyConfig.utf8))
        let ple = EG2PLE(cfg)
        let x = MLXRandom.normal([2, 5, cfg.hiddenSize])
        let full = ple(x)  // [B, T, L, pleDim]
        for i in [0, 1, cfg.numLayers - 1] {
            let d = abs(ple.layerInput(x, layer: i) - full[0..., 0..., i, 0...]).max().item(Float.self)
            XCTAssertLessThan(d, 1e-5, "layer \(i)")
        }
    }

    static let tinyConfig = """
    {"text_config":{"hidden_size":16,"intermediate_size":32,"num_hidden_layers":3,"num_attention_heads":2,
    "num_key_value_heads":1,"head_dim":8,"hidden_size_per_layer_input":8,"embedding_dim":12,"vocab_size":50,
    "rms_norm_eps":1e-6,"sliding_window":4,"layer_types":["sliding_attention","sliding_attention","full_attention"],
    "per_layer_config":{"2":{"head_dim":16,"num_key_value_heads":1}}}}
    """
}
