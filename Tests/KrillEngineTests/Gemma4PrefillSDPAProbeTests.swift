import XCTest
import MLX
import MLXFast
@testable import KrillCore

/// Probe: does the Gemma 4 prefill OOM come from the attention SDPA materializing
/// the full [heads, L, L] score matrix when given an additive *array* mask, and
/// does the `.causal` mask mode avoid it via the fused/flash kernel?
///
/// Pure synthetic shapes (no model load) at gemma-4-12b attention dims, so it is
/// fast and isolates the SDPA call. Gated on an env var so it never OOMs CI:
///   KRILL_RUN_SDPA_PROBE=1 swift test --filter Gemma4PrefillSDPAProbeTests
final class Gemma4PrefillSDPAProbeTests: XCTestCase {

    private func requireOptIn() throws {
        guard ProcessInfo.processInfo.environment["KRILL_RUN_SDPA_PROBE"] == "1" else {
            throw XCTSkip("KRILL_RUN_SDPA_PROBE != 1")
        }
    }

    /// gemma-4-12b: 16 query heads, 8 KV heads; full-attn head dim 256
    /// (config `head_dim`). Build [1, H, L, D] random bf16.
    private func qkv(L: Int, nq: Int = 16, nkv: Int = 8, d: Int = 256)
        -> (MLXArray, MLXArray, MLXArray) {
        let q = MLXRandom.normal([1, nq, L, d]).asType(.bfloat16)
        let k = MLXRandom.normal([1, nkv, L, d]).asType(.bfloat16)
        let v = MLXRandom.normal([1, nkv, L, d]).asType(.bfloat16)
        return (q, k, v)
    }

    private func mib(_ bytes: Int) -> String { String(format: "%.2f GB", Double(bytes) / 1_073_741_824) }

    /// Run one SDPA call, return (peakBytesDelta, millis) or nil if it threw/OOM'd.
    private func measure(
        _ label: String, q: MLXArray, k: MLXArray, v: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> (peak: Int, ms: Double, out: MLXArray)? {
        GPU.resetPeakMemory()
        let before = Memory.snapshot().peakMemory
        let t0 = Date()
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: mask)
        out.eval()
        let ms = Date().timeIntervalSince(t0) * 1000
        let peak = Memory.snapshot().peakMemory - before
        FileHandle.standardError.write(Data(
            "  \(label): peakΔ=\(mib(max(peak, 0)))  time=\(String(format: "%.1f", ms))ms\n".utf8))
        return (peak, ms, out)
    }

    /// CORRECTNESS: at a small L, `.causal` must equal `.array(additive causal)`.
    func testCausalEqualsAdditiveMaskSmallL() throws {
        try requireOptIn()
        let L = 512
        let (q, k, v) = qkv(L: L)
        let additive = createAdditiveCausalMask(L, dtype: .bfloat16)
        let aCausal = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: .causal)
        let aArray = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: .array(additive))
        let maxDiff = MLX.max(MLX.abs(aCausal - aArray)).item(Float.self)
        print("\n[correctness] L=\(L) maxAbsDiff(.causal vs .array) = \(maxDiff)")
        XCTAssertLessThan(maxDiff, 0.05,
            ".causal output diverged from explicit additive causal mask")
    }

    /// MEMORY (.causal): sweep L INCLUDING 16k and 32k. If `.causal` uses the
    /// fused/flash kernel, peakΔ stays bounded (no [16,L,L] score matrix) and
    /// 16k/32k run fine — proving `.causal` is the fix for full-attention layers.
    /// Run in isolation: KRILL_RUN_SDPA_PROBE=1 swift test --filter testCausalSweepBoundedMemory
    func testCausalSweepBoundedMemory() throws {
        try requireOptIn()
        print("\n===== SDPA .causal peak-memory sweep (16q/8kv/d256) =====")
        for L in [2048, 4096, 8192, 12288, 16384] {
            autoreleasepool {
                let (q, k, v) = qkv(L: L)
                FileHandle.standardError.write(Data("L=\(L): running .causal...\n".utf8))
                _ = measure(".causal", q: q, k: k, v: v, mask: .causal)
            }
        }
        print("========================================================\n")
    }

    /// MEMORY (.array): the same sweep with an explicit additive causal mask.
    /// Hypothesis: this materializes [16,L,L] and the process FATAL-OOMs around
    /// L~16k (a single >14.3GB buffer). Run ALONE so its crash does not mask the
    /// .causal result: KRILL_RUN_SDPA_PROBE=1 swift test --filter testArraySweepHitsOOM
    func testArraySweepHitsOOM() throws {
        try requireOptIn()
        print("\n===== SDPA .array(additive causal) peak-memory sweep (16q/8kv/d256) =====")
        for L in [2048, 4096, 8192, 12288, 16384] {
            autoreleasepool {
                let (q, k, v) = qkv(L: L)
                let additive = createAdditiveCausalMask(L, dtype: .bfloat16)
                print("L=\(L):  (additive mask = \(mib(L * L * 2)) bf16)")
                _ = measure(".array ", q: q, k: k, v: v, mask: .array(additive))
            }
        }
        print("=======================================================================\n")
    }
}
