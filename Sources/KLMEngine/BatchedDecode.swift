import Foundation
import MLX
import KLMCore
import KLMCache
import KLMSampler

/// Stage B - true KV-batched concurrent decode for plain-causal families
/// (Llama, Qwen 2.5/3 dense). Concurrent requests are prefilled individually
/// (prefill is already fast and correct at B=1), their per-row KV caches are
/// stacked LEFT-padded into one batched cache, and a single batched forward
/// decodes one token for every row per step. Per-row RoPE offsets keep each
/// row's positions natural despite the shared stacked layout, and a per-row
/// additive mask hides each row's left-pad prefix - so a batched row's output
/// matches running that prompt alone (no cross-row attention bleed).
///
/// v1 scope: fp16 KV only (no int8 / prefix cache on the batched path),
/// greedy or per-row sampling, no speculative decode. Finished rows stay in
/// the batch until all rows complete (they are simply not emitted) - a
/// correctness-first simplification; shrinking the batch mid-flight is a
/// follow-up.

/// Large negative additive-mask value, matching the codebase's causal-mask
/// convention (`createAdditiveCausalMask` uses the same magnitude). Added to a
/// left-pad key's attention score, it drives that key's softmax weight to
/// underflow to exactly zero in fp16, bf16, and fp32 alike (the valid scores
/// are O(10), so `exp(score - 10000)` is 0 in every supported dtype) - making
/// a batched row's attention identical to the unpadded single-row case.
private let maskNegInf: Float = -10000.0

/// Stack R per-row KV caches (one `[KVCache]` per row, each of shape
/// `[1, kvHeads, L_r, headDim]` per layer) into one batched `[KVCache]` per
/// layer of shape `[R, kvHeads, sMax, headDim]`, LEFT-padding shorter rows
/// with zeros so all rows align at the right edge (`sMax = max L_r`). The pad
/// region is masked during decode, so its zero K/V never contributes.
func stackCachesLeftPadded(perRow caches: [[KVCache]], lengths: [Int]) -> [KVCache] {
    let R = caches.count
    let numLayers = caches[0].count
    let sMax = lengths.max() ?? 0
    var batched: [KVCache] = []
    batched.reserveCapacity(numLayers)
    for layer in 0 ..< numLayers {
        // Snapshot every row's cache for this layer up front. A layer can be
        // legitimately empty for ALL rows (Gemma 4 KV-shared layers never
        // write their own cache - the donor layer holds the K/V and the
        // shared layer reuses it). Such a uniformly-empty layer gets a fresh
        // empty placeholder: the batched forward routes shared layers to the
        // donor's stacked cache and never reads or updates the placeholder.
        let snaps = (0 ..< R).map { caches[$0][layer].snapshot() }
        let present = snaps.lazy.filter { $0 != nil }.count
        if present == 0 {
            batched.append(KVCache())
            continue
        }
        // A PARTIALLY-empty layer (some rows have K/V, some don't) is a real
        // batch desync: stacking would silently drop a row and shift live rows
        // against the full-R mask/offsets - fail hard instead.
        precondition(
            present == R,
            "partially-empty KV layer \(layer): \(present)/\(R) rows cached (batch desync)")
        var ks: [MLXArray] = []
        var vs: [MLXArray] = []
        ks.reserveCapacity(R)
        vs.reserveCapacity(R)
        for r in 0 ..< R {
            let snap = snaps[r]!
            let pad = sMax - lengths[r]
            if pad > 0 {
                let zerosK = MLXArray.zeros(
                    [1, snap.keys.dim(1), pad, snap.keys.dim(3)], dtype: snap.keys.dtype)
                let zerosV = MLXArray.zeros(
                    [1, snap.values.dim(1), pad, snap.values.dim(3)], dtype: snap.values.dtype)
                ks.append(concatenated([zerosK, snap.keys], axis: 2))
                vs.append(concatenated([zerosV, snap.values], axis: 2))
            } else {
                ks.append(snap.keys)
                vs.append(snap.values)
            }
        }
        let bc = KVCache()
        bc.restore(keys: concatenated(ks, axis: 0), values: concatenated(vs, axis: 0))
        batched.append(bc)
    }
    return batched
}

/// Build the per-row additive decode mask `[R, 1, 1, totalLen]`: for each row
/// the first `sMax - L_r` columns (its left-pad prefix in the stacked cache)
/// are masked to a large negative, the rest are 0 (valid). Rebuilt each step
/// as `totalLen` grows; the masked prefix is fixed per row.
func buildBatchedDecodeMask(lengths: [Int], sMax: Int, totalLen: Int, dtype: DType) -> MLXArray {
    let R = lengths.count
    var flat = [Float](repeating: 0, count: R * totalLen)
    for r in 0 ..< R {
        let pad = sMax - lengths[r]
        if pad > 0 {
            for j in 0 ..< pad { flat[r * totalLen + j] = maskNegInf }
        }
    }
    return MLXArray(flat).reshaped(R, 1, 1, totalLen).asType(dtype)
}

extension InferenceEngine {
    /// Whether the loaded model supports the Stage B batched decode path.
    public var supportsBatchedDecode: Bool {
        loadedModelForBatching?.batchedDecodeForward != nil
    }

    /// Greedy batched decode of `promptTokens` (already tokenized), returning
    /// the generated token IDs per row - including the first (prefill-sampled)
    /// token, matching the serialized path's emission order. Stops a row at
    /// EOS or when it reaches `maxTokens` generated tokens. Returns nil if the
    /// loaded model is not batched-eligible.
    ///
    /// This is the correctness entry: a test asserts each row equals the
    /// serialized single-prompt run token-for-token (cross-row isolation).
    func batchedGreedyDecode(promptTokens: [[Int]], maxTokens: Int) -> [[Int]]? {
        guard let model = loadedModelForBatching,
              let batchedForward = model.batchedDecodeForward,
              let eosId = tokenizerEOS else { return nil }
        let R = promptTokens.count
        guard R > 0 else { return [] }
        let numLayers = model.numLayers
        let prefill = model.prefillForward ?? model.forward
        let sampler = Sampler(params: .greedy)

        // Per-row prefill: individual B=1 forwards fill each row's cache and
        // give the first sampled token. Prefill stays the unchanged path.
        var perRowCaches: [[KVCache]] = []
        var lengths: [Int] = []
        var current: [Int] = []
        for tokens in promptTokens {
            let caches = makeKVCaches(numLayers: numLayers)
            let input = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)
            let logits = prefill(input, caches)
            MLX.eval(logits)
            perRowCaches.append(caches)
            lengths.append(tokens.count)
            current.append(sampler.sample(logits))
        }

        let sMax = lengths.max() ?? 0
        let caches = stackCachesLeftPadded(perRow: perRowCaches, lengths: lengths)
        let kDtype = caches.first?.snapshot()?.keys.dtype ?? .float16

        var generated: [[Int]] = current.map { [$0] }
        var done = current.map { $0 == eosId }
        var step = 0
        while generated.contains(where: { $0.count < maxTokens }) && !done.allSatisfy({ $0 }) {
            let rowOffsets = (0 ..< R).map { lengths[$0] + step }
            let totalLen = sMax + step + 1
            let mask = buildBatchedDecodeMask(
                lengths: lengths, sMax: sMax, totalLen: totalLen, dtype: kDtype)
            let input = MLXArray(current.map { Int32($0) }).reshaped(R, 1)
            let logits = batchedForward(input, caches, mask, rowOffsets)
            MLX.eval(logits)
            for r in 0 ..< R {
                let tok = sampler.sample(logits[r ..< (r + 1)])
                current[r] = tok
                guard !done[r] else { continue }
                if tok == eosId {
                    done[r] = true
                } else if generated[r].count < maxTokens {
                    generated[r].append(tok)
                }
            }
            step += 1
            if step > maxTokens + 4 { break }   // safety bound
        }
        return generated
    }

    /// Dtype-independent cross-row contamination gate: run `row` (teacher-forced
    /// with its own greedy continuation) in TWO batches that share the same
    /// width and the same neighbor LENGTHS but use different neighbor CONTENT,
    /// and return the max-abs diff of `row`'s logits between them. Batched
    /// matmul computes each batch row from its own inputs only, so `row`'s
    /// logits must be BIT-IDENTICAL regardless of what the neighbors contain -
    /// any nonzero result is a genuine cross-batch indexing/bleed bug. Unlike
    /// `teacherForcedBatchedVsSerialMaxDiff` this is not confounded by the
    /// batched-vs-solo GEMM kernel-width rounding (it compares batched-vs-
    /// batched at a FIXED width), so it stays ~0 even for bf16 models. The
    /// neighbors share `row`'s length-region geometry (kept equal across the
    /// two batches), so the mask/pad layout `row` sees is identical in both -
    /// isolating neighbor content as the only variable.
    func crossRowContaminationMaxDiff(
        row: [Int], neighborsA: [[Int]], neighborsB: [[Int]], steps: Int
    ) -> Float? {
        guard let model = loadedModelForBatching,
              let batchedForward = model.batchedDecodeForward else { return nil }
        precondition(
            neighborsA.count == neighborsB.count
                && zip(neighborsA, neighborsB).allSatisfy { $0.count == $1.count },
            "contamination probe requires matching neighbor counts & lengths across A/B")
        let prefill = model.prefillForward ?? model.forward
        let cont = serialGreedyDecode(promptTokens: row, maxTokens: steps + 1) ?? []
        let usableSteps = min(steps, cont.count - 1)
        guard usableSteps >= 1 else { return nil }

        // Run `row` at index 0 of [row] + neighbors; return its per-step logits.
        func rowZeroLogits(_ neighbors: [[Int]]) -> [MLXArray] {
            let prompts = [row] + neighbors
            let R = prompts.count
            var perRowCaches: [[KVCache]] = []
            var lengths: [Int] = []
            for tokens in prompts {
                let bc = makeKVCaches(numLayers: model.numLayers)
                let pl = prefill(MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count), bc)
                MLX.eval(pl)
                perRowCaches.append(bc)
                lengths.append(tokens.count)
            }
            let sMax = lengths.max() ?? 0
            let caches = stackCachesLeftPadded(perRow: perRowCaches, lengths: lengths)
            let kDtype = caches.first?.snapshot()?.keys.dtype ?? .float16
            var out: [MLXArray] = []
            for k in 0 ..< usableSteps {
                let rowOffsets = (0 ..< R).map { lengths[$0] + k }
                let mask = buildBatchedDecodeMask(
                    lengths: lengths, sMax: sMax, totalLen: sMax + k + 1, dtype: kDtype)
                // Row 0 gets its real continuation; neighbors get a filler token
                // (any valid id - their logits are never read).
                let toks = [cont[k]] + neighbors.map { $0.first ?? 0 }
                let input = MLXArray(toks.map { Int32($0) }).reshaped(R, 1)
                let bl = batchedForward(input, caches, mask, rowOffsets)
                MLX.eval(bl)
                out.append(bl[0 ..< 1].reshaped(-1))
            }
            return out
        }

        let a = rowZeroLogits(neighborsA)
        let b = rowZeroLogits(neighborsB)
        var maxDiff: Float = 0
        for k in 0 ..< usableSteps {
            maxDiff = Swift.max(maxDiff, MLX.max(MLX.abs(a[k] - b[k])).item(Float.self))
        }
        return maxDiff
    }

    /// Rigorous multi-step correctness gate: teacher-force each row with its
    /// OWN reference continuation and compare, at every step, the batched
    /// per-row logits against that prompt's solo logits. Returns the per-row
    /// max-abs logit difference over all `steps`. Teacher-forcing (identical
    /// inputs) sidesteps greedy tie-flip divergence, so a near-zero diff
    /// proves there is no cross-row bleed and no per-step position/mask error
    /// across the whole decode - the property exact-token matching cannot
    /// assert under fp16 batched-GEMM rounding.
    func teacherForcedBatchedVsSerialMaxDiff(
        promptTokens: [[Int]], steps: Int
    ) -> [Float]? {
        guard let model = loadedModelForBatching,
              let batchedForward = model.batchedDecodeForward else { return nil }
        let R = promptTokens.count
        let prefill = model.prefillForward ?? model.forward

        // Reference continuations (the teacher-forcing tokens) per row.
        var cont: [[Int]] = []
        for tokens in promptTokens {
            cont.append(serialGreedyDecode(promptTokens: tokens, maxTokens: steps + 1) ?? [])
        }
        let usableSteps = min(steps, (cont.map { $0.count }.min() ?? 1) - 1)
        guard usableSteps >= 1 else { return nil }

        // Serial teacher-forced logits: [R][usableSteps], each [vocab].
        var serialLogits: [[MLXArray]] = []
        for (i, tokens) in promptTokens.enumerated() {
            let sc = makeKVCaches(numLayers: model.numLayers)
            let pl = prefill(MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count), sc)
            MLX.eval(pl)
            var rowLogits: [MLXArray] = []
            for k in 0 ..< usableSteps {
                let l = model.forward(MLXArray([Int32(cont[i][k])]).reshaped(1, 1), sc)
                MLX.eval(l)
                rowLogits.append(l.reshaped(-1))
            }
            serialLogits.append(rowLogits)
        }

        // Batched teacher-forced: prefill all, stack, feed each row its own
        // continuation token per step under per-row offsets + mask.
        var lengths: [Int] = []
        var perRowCaches: [[KVCache]] = []
        for tokens in promptTokens {
            let bc = makeKVCaches(numLayers: model.numLayers)
            let pl = prefill(MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count), bc)
            MLX.eval(pl)
            perRowCaches.append(bc)
            lengths.append(tokens.count)
        }
        let sMax = lengths.max() ?? 0
        let caches = stackCachesLeftPadded(perRow: perRowCaches, lengths: lengths)
        let kDtype = caches.first?.snapshot()?.keys.dtype ?? .float16
        var maxDiff = [Float](repeating: 0, count: R)
        for k in 0 ..< usableSteps {
            let rowOffsets = (0 ..< R).map { lengths[$0] + k }
            let mask = buildBatchedDecodeMask(
                lengths: lengths, sMax: sMax, totalLen: sMax + k + 1, dtype: kDtype)
            let input = MLXArray((0 ..< R).map { Int32(cont[$0][k]) }).reshaped(R, 1)
            let bl = batchedForward(input, caches, mask, rowOffsets)
            MLX.eval(bl)
            for r in 0 ..< R {
                let br = bl[r ..< (r + 1)].reshaped(-1)
                let d = MLX.max(MLX.abs(br - serialLogits[r][k])).item(Float.self)
                maxDiff[r] = Swift.max(maxDiff[r], d)
            }
        }
        return maxDiff
    }

    /// Reference canonical B=1 greedy decode from raw tokens (same model, NO
    /// batching). The Stage B test asserts the batched path reproduces this
    /// exactly, both at R=1 (the batched forward equals the canonical one) and
    /// for each row of an R=N batch (no cross-row interference). Returns nil
    /// if no model is loaded.
    func serialGreedyDecode(promptTokens: [Int], maxTokens: Int) -> [Int]? {
        guard let model = loadedModelForBatching, let eosId = tokenizerEOS else { return nil }
        let caches = makeKVCaches(numLayers: model.numLayers)
        let sampler = Sampler(params: .greedy)
        let prefill = model.prefillForward ?? model.forward
        let input = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, promptTokens.count)
        var logits = prefill(input, caches)
        MLX.eval(logits)
        var tok = sampler.sample(logits)
        var gen = [tok]
        var done = tok == eosId
        while gen.count < maxTokens && !done {
            let inp = MLXArray([Int32(tok)]).reshaped(1, 1)
            logits = model.forward(inp, caches)
            MLX.eval(logits)
            tok = sampler.sample(logits)
            if tok == eosId { done = true } else { gen.append(tok) }
        }
        return gen
    }
}
