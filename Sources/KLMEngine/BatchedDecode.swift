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

/// Quantized analogue of ``stackCachesLeftPadded``: stack R per-row
/// `[QuantizedKVCache]` (each layer `[1, kvHeads, L_r, headDim]` uint8 storage
/// plus per-position fp16 scales/zeros) into one batched `[QuantizedKVCache]`
/// per layer of shape `[R, kvHeads, sMax, headDim]`, LEFT-padding shorter rows.
/// The six tensors (uint8 keys/values + fp16 key/value scales/zeros) are each
/// concatenated along the sequence axis per row, then batched across rows on
/// axis 0 - so dequantization inside attention reads each row's own values
/// only. The pad region is filled with quantized zeros (`q*scale + zero` with
/// all three zero dequantizes to exactly 0) and is masked during decode, so it
/// never contributes - matching the fp16 zero-pad contract.
func stackCachesLeftPaddedQuantized(
    perRow caches: [[QuantizedKVCache]], lengths: [Int]
) -> [QuantizedKVCache] {
    let R = caches.count
    let numLayers = caches[0].count
    let sMax = lengths.max() ?? 0
    var batched: [QuantizedKVCache] = []
    batched.reserveCapacity(numLayers)
    for layer in 0 ..< numLayers {
        // As with the fp16 stack, a layer can be uniformly empty for ALL rows
        // (Gemma 4 KV-shared layers never write their own cache - the donor
        // holds the K/V). Such a layer gets a fresh empty placeholder the
        // batched forward routes around; a PARTIALLY-empty layer is a batch
        // desync and fails hard.
        let snaps = (0 ..< R).map { caches[$0][layer].quantizedSnapshot() }
        let present = snaps.lazy.filter { $0 != nil }.count
        if present == 0 {
            batched.append(QuantizedKVCache())
            continue
        }
        precondition(
            present == R,
            "partially-empty quantized KV layer \(layer): \(present)/\(R) rows cached (batch desync)")
        var qks: [MLXArray] = []; var qvs: [MLXArray] = []
        var kss: [MLXArray] = []; var kzs: [MLXArray] = []
        var vss: [MLXArray] = []; var vzs: [MLXArray] = []
        for r in 0 ..< R {
            let s = snaps[r]!
            let pad = sMax - lengths[r]
            if pad > 0 {
                let H = s.keys.dim(1), D = s.keys.dim(3)
                // Pad with quantized zeros: uint8 0 storage + 0 scale + 0 zero
                // dequantizes to 0 at every padded position (mask hides it too).
                let padQ = MLXArray.zeros([1, H, pad, D], dtype: s.keys.dtype)
                let padS = MLXArray.zeros([1, H, pad, 1], dtype: s.keyScales.dtype)
                qks.append(concatenated([padQ, s.keys], axis: 2))
                qvs.append(concatenated([padQ, s.values], axis: 2))
                kss.append(concatenated([padS, s.keyScales], axis: 2))
                kzs.append(concatenated([padS, s.keyZeros], axis: 2))
                vss.append(concatenated([padS, s.valueScales], axis: 2))
                vzs.append(concatenated([padS, s.valueZeros], axis: 2))
            } else {
                qks.append(s.keys); qvs.append(s.values)
                kss.append(s.keyScales); kzs.append(s.keyZeros)
                vss.append(s.valueScales); vzs.append(s.valueZeros)
            }
        }
        let bc = QuantizedKVCache()
        bc.restoreQuantized(QuantizedKVSnapshot(
            keys: concatenated(qks, axis: 0), values: concatenated(qvs, axis: 0),
            keyScales: concatenated(kss, axis: 0), keyZeros: concatenated(kzs, axis: 0),
            valueScales: concatenated(vss, axis: 0), valueZeros: concatenated(vzs, axis: 0)))
        batched.append(bc)
    }
    return batched
}

/// Type-dispatching stack: route a batched set of per-row caches to the fp16
/// or the quantized stacker by their concrete type. The batcher allocates every
/// row/layer from one factory, so inspecting any element decides the path. Lets
/// the (type-agnostic) decode loop hold `[KVCacheProtocol]` while the stacking
/// stays type-correct.
func stackCachesLeftPaddedAny(
    perRow caches: [[KVCacheProtocol]], lengths: [Int]
) -> [KVCacheProtocol] {
    if caches.first?.first is QuantizedKVCache {
        let q = caches.map { $0.map { $0 as! QuantizedKVCache } }
        return stackCachesLeftPaddedQuantized(perRow: q, lengths: lengths)
    }
    let f = caches.map { $0.map { $0 as! KVCache } }
    return stackCachesLeftPadded(perRow: f, lengths: lengths)
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

/// Build the per-row additive **multi-query verify** mask `[R, 1, L, totalLen]`
/// (`totalLen = sMax + L`) for a batched speculative-verify forward that feeds
/// `L` new tokens per row at once. For row `r`, query position `j` (0-based over
/// the L new tokens, absolute position `lengths[r] + j`):
///   - cached columns `[0, sMax)`: the left-pad prefix `[0, sMax - lengths[r])`
///     is masked (same as the single-query decode mask);
///   - new columns `[sMax, sMax+L)`: block-causal — query `j` may attend to new
///     keys `0..j` and is masked from `j+1..L-1` (its own future drafts).
/// This makes the L-wide forward identical to L sequential single-token batched
/// steps (each new token sees the cache + the accepted prefix, not future drafts).
func buildBatchedVerifyMask(lengths: [Int], sMax: Int, newLen L: Int, dtype: DType) -> MLXArray {
    let R = lengths.count
    let totalLen = sMax + L
    var flat = [Float](repeating: 0, count: R * L * totalLen)
    for r in 0 ..< R {
        let pad = sMax - lengths[r]
        for j in 0 ..< L {
            let base = (r * L + j) * totalLen
            for k in 0 ..< pad { flat[base + k] = maskNegInf }            // left-pad
            for nk in (j + 1) ..< L { flat[base + sMax + nk] = maskNegInf } // future drafts
        }
    }
    return MLXArray(flat).reshaped(R, 1, L, totalLen).asType(dtype)
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

    /// Batched **n-gram speculative** decode: like ``batchedGreedyDecode`` but each
    /// row proposes prompt-lookup draft tokens and they are verified in one wide
    /// `[R, W]` batched forward (W = 1 + max draft length), committing multiple
    /// tokens per row per round. This trades a per-round re-stack for fewer rounds
    /// and a wider forward that fills the GPU better at moderate R (the occupancy
    /// gap diagnosed in `docs/CONCURRENT_THROUGHPUT.md`).
    ///
    /// Greedy-only (argmax verify), fp16 KV. Per-row output equals that row's
    /// batched greedy output up to fp16 verify-vs-decode tie-flips. Returns nil if
    /// the model is not batch-eligible. v1 keeps finished rows in the batch (not
    /// committed/emitted) — the same simplification as ``batchedGreedyDecode``.
    func batchedNgramSpecDecode(promptTokens: [[Int]], maxTokens: Int) -> [[Int]]? {
        guard let model = loadedModelForBatching,
              let batchedForward = model.batchedDecodeForward,
              let stopId = tokenizerEOS else { return nil }
        let R = promptTokens.count
        guard R > 0 else { return [] }
        let numLayers = model.numLayers
        let prefill = model.prefillForward ?? model.forward
        let sampler = Sampler(params: .greedy)

        // Per-row prefill + a per-row proposer seeded with the prompt + first token.
        var perRowCaches: [[KVCache]] = []
        var lengths: [Int] = []
        var current: [Int] = []        // each row's lastToken, NOT yet in its cache
        var proposers: [NgramProposer] = []
        for tokens in promptTokens {
            let caches = makeKVCaches(numLayers: numLayers)
            let pl = prefill(MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count), caches)
            MLX.eval(pl)
            let first = sampler.sample(pl)
            perRowCaches.append(caches)
            lengths.append(tokens.count)
            current.append(first)
            let p = NgramProposer(config: .init(), eosIds: [stopId])
            p.reset(prompt: tokens)
            p.append([first])
            proposers.append(p)
        }
        var generated: [[Int]] = current.map { [$0] }
        var done = current.map { $0 == stopId }
        var rounds = 0
        let roundCap = maxTokens + 8   // safety bound

        while (0 ..< R).contains(where: { !done[$0] && generated[$0].count < maxTokens }) {
            rounds += 1
            if rounds > roundCap { break }
            let active = (0 ..< R).filter { !done[$0] && generated[$0].count < maxTokens }
            if active.isEmpty { break }

            // Propose per active row; verify width W = 1 + max draft length.
            var drafts = [[Int]](repeating: [], count: R)
            for r in active { drafts[r] = proposers[r].propose() }
            let maxK = active.map { drafts[$0].count }.max() ?? 0
            let W = maxK + 1

            // Per-row verify input [R, W] = [current_r, draft_r..., pad]. Padding
            // (and inactive rows) repeat the last token; their predictions are
            // ignored and their KV is never committed.
            var flat = [Int32](repeating: 0, count: R * W)
            for r in 0 ..< R {
                let seq = [current[r]] + drafts[r]
                for j in 0 ..< W { flat[r * W + j] = Int32(seq[min(j, seq.count - 1)]) }
            }

            let sMax = lengths.max() ?? 0
            let caches = stackCachesLeftPadded(perRow: perRowCaches, lengths: lengths)
            let kDtype = caches.first?.snapshot()?.keys.dtype ?? .float16
            let mask = buildBatchedVerifyMask(lengths: lengths, sMax: sMax, newLen: W, dtype: kDtype)
            let bl = batchedForward(
                MLXArray(flat).reshaped(R, W), caches, mask, lengths)   // [R, W, vocab]
            MLX.eval(bl)
            let pred = argMax(bl, axis: -1).asType(.int32)              // [R, W]
            MLX.eval(pred)
            let predFlat = pred.reshaped(-1).asArray(Int32.self).map(Int.init)
            func p(_ r: Int, _ j: Int) -> Int { predFlat[r * W + j] }

            for r in active {
                let k = drafts[r].count
                // Accept drafts up to the first mismatch: p(r,i) is the model's
                // greedy token after input position i, which must equal draft_r[i].
                var accepted: [Int] = []
                var i = 0
                while i < k && p(r, i) == drafts[r][i] { accepted.append(drafts[r][i]); i += 1 }
                let allAccepted = (i == k)
                let cacheEntries: Int          // new KV positions to keep for row r
                if allAccepted {
                    accepted.append(p(r, k))   // bonus (input was draft_r[k-1], or current if k==0)
                    cacheEntries = k + 1       // current + all k drafts are now context
                } else {
                    accepted.append(p(r, i))   // target replacement at the mismatch
                    cacheEntries = i + 1       // current + the i accepted drafts
                }

                // Commit the kept new KV (stacked positions [sMax, sMax+cacheEntries))
                // into row r's own per-row cache, preserving per-row isolation.
                for layer in 0 ..< numLayers {
                    guard let snap = caches[layer].snapshot() else { continue }
                    let ks = snap.keys[r ..< (r + 1), 0..., sMax ..< (sMax + cacheEntries), 0...]
                    let vs = snap.values[r ..< (r + 1), 0..., sMax ..< (sMax + cacheEntries), 0...]
                    _ = perRowCaches[r][layer].update(keys: ks, values: vs)
                }
                lengths[r] += cacheEntries
                if k > 0 {
                    proposers[r].recordOutcome(
                        acceptedDraft: allAccepted ? k : accepted.count - 1, proposed: k)
                }
                proposers[r].append(accepted)

                // Emit, stopping the row at the first stop id or maxTokens.
                for tok in accepted {
                    if tok == stopId { done[r] = true; break }
                    generated[r].append(tok)
                    if generated[r].count >= maxTokens { break }
                }
                current[r] = accepted.last ?? stopId
                if current[r] == stopId { done[r] = true }
            }
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

    /// Correctness gate for the batched **multi-token verify** primitive (the
    /// foundation of batched speculative decode): feed each row its `L`
    /// teacher-forcing continuation tokens in ONE `[R, L]` batched forward (with
    /// `buildBatchedVerifyMask`), and return the per-row max abs logit diff vs the
    /// serial single-token reference at every position. A near-zero diff proves
    /// the L-wide block-causal forward equals L sequential single-token steps —
    /// i.e. per-row RoPE (`offset + j`) and the block-causal mask are correct, so
    /// a speculative verify can commit multiple tokens per row in one step.
    func batchedVerifyVsSerialMaxDiff(promptTokens: [[Int]], steps L: Int) -> [Float]? {
        guard let model = loadedModelForBatching,
              let batchedForward = model.batchedDecodeForward else { return nil }
        let R = promptTokens.count
        let prefill = model.prefillForward ?? model.forward

        var cont: [[Int]] = []
        for tokens in promptTokens {
            cont.append(serialGreedyDecode(promptTokens: tokens, maxTokens: L + 1) ?? [])
        }
        let usable = min(L, (cont.map { $0.count }.min() ?? 1) - 1)
        guard usable >= 1 else { return nil }

        // Serial teacher-forced logits: [R][usable], each [vocab].
        var serialLogits: [[MLXArray]] = []
        for (i, tokens) in promptTokens.enumerated() {
            let sc = makeKVCaches(numLayers: model.numLayers)
            let pl = prefill(MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count), sc)
            MLX.eval(pl)
            var rowLogits: [MLXArray] = []
            for k in 0 ..< usable {
                let l = model.forward(MLXArray([Int32(cont[i][k])]).reshaped(1, 1), sc)
                MLX.eval(l)
                rowLogits.append(l.reshaped(-1))
            }
            serialLogits.append(rowLogits)
        }

        // Batched: prefill all, stack, ONE [R, usable] verify forward.
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
        // Per-row base offset; applyRoPEPerRow rotates new position j to offset+j.
        let rowOffsets = lengths
        let mask = buildBatchedVerifyMask(
            lengths: lengths, sMax: sMax, newLen: usable, dtype: kDtype)
        let flatTokens = (0 ..< R).flatMap { r in (0 ..< usable).map { Int32(cont[r][$0]) } }
        let input = MLXArray(flatTokens).reshaped(R, usable)
        let bl = batchedForward(input, caches, mask, rowOffsets)   // [R, usable, vocab]
        MLX.eval(bl)

        var maxDiff = [Float](repeating: 0, count: R)
        for r in 0 ..< R {
            for k in 0 ..< usable {
                let br = bl[r ..< (r + 1), k ..< (k + 1)].reshaped(-1)
                let d = MLX.max(MLX.abs(br - serialLogits[r][k])).item(Float.self)
                maxDiff[r] = Swift.max(maxDiff[r], d)
            }
        }
        return maxDiff
    }

    /// Whether the loaded model supports int8-quantized batched decode (the
    /// Stage C4 quantized path). Set for families whose attention accepts
    /// `KVCacheProtocol` (currently Gemma 4).
    public var supportsQuantizedBatchedDecode: Bool {
        loadedModelForBatching?.batchedDecodeForwardQuantized != nil
    }

    /// int8 analogue of ``crossRowContaminationMaxDiff``: the dtype-independent
    /// cross-row isolation gate, run on the QUANTIZED batched path. A row's
    /// logits must be BIT-IDENTICAL regardless of neighbor CONTENT (same batch
    /// width + neighbor lengths), because each batch element is quantized and
    /// matmul-ed from its own inputs only - any nonzero result is a genuine
    /// cross-row bleed or a quantized-stacking indexing bug. int8 quantization
    /// is lossy but per-row, so this stays exactly 0 (it compares
    /// batched-vs-batched at a FIXED width, so neither the int8 rounding nor the
    /// bf16 GEMM-width rounding confounds it).
    func quantizedCrossRowContaminationMaxDiff(
        row: [Int], neighborsA: [[Int]], neighborsB: [[Int]], steps: Int
    ) -> Float? {
        guard let model = loadedModelForBatching,
              let batchedForward = model.batchedDecodeForwardQuantized else { return nil }
        precondition(
            neighborsA.count == neighborsB.count
                && zip(neighborsA, neighborsB).allSatisfy { $0.count == $1.count },
            "contamination probe requires matching neighbor counts & lengths across A/B")
        let prefill = model.prefillForward ?? model.forward
        // Row's own greedy continuation, computed on the int8 serial path so the
        // teacher-forcing tokens match what this row would generate alone.
        let cont = quantizedSerialGreedyDecode(promptTokens: row, maxTokens: steps + 1) ?? []
        let usableSteps = min(steps, cont.count - 1)
        guard usableSteps >= 1 else { return nil }

        func rowZeroLogits(_ neighbors: [[Int]]) -> [MLXArray] {
            let prompts = [row] + neighbors
            let R = prompts.count
            var perRowCaches: [[QuantizedKVCache]] = []
            var lengths: [Int] = []
            for tokens in prompts {
                let bc = makeQuantizedKVCaches(numLayers: model.numLayers)
                let pl = prefill(MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count), bc)
                MLX.eval(pl)
                perRowCaches.append(bc)
                lengths.append(tokens.count)
            }
            let sMax = lengths.max() ?? 0
            let caches = stackCachesLeftPaddedQuantized(perRow: perRowCaches, lengths: lengths)
            var out: [MLXArray] = []
            for k in 0 ..< usableSteps {
                let rowOffsets = (0 ..< R).map { lengths[$0] + k }
                let mask = buildBatchedDecodeMask(
                    lengths: lengths, sMax: sMax, totalLen: sMax + k + 1, dtype: .float16)
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

    /// int8 analogue of ``teacherForcedBatchedVsSerialMaxDiff``: teacher-force
    /// each row with its own int8 serial continuation and compare, per step, the
    /// QUANTIZED batched per-row logits against that prompt's int8 SOLO logits.
    /// At R=1 (no left-pad, the stacked cache IS the single row) the K is
    /// quantized post-RoPE identically in both paths, so the diff is ~0
    /// bit-exact - proving per-row quantized stacking + decode correctness. At
    /// R>1 the same bf16 batched-GEMM kernel-width rounding the fp16 path sees
    /// applies (so the test uses the loose explosion-guard bound there); the
    /// real correctness signal is the bit-exact contamination gate above.
    func quantizedTeacherForcedBatchedVsSerialMaxDiff(
        promptTokens: [[Int]], steps: Int
    ) -> [Float]? {
        guard let model = loadedModelForBatching,
              let batchedForward = model.batchedDecodeForwardQuantized else { return nil }
        let R = promptTokens.count
        let prefill = model.prefillForward ?? model.forward

        var cont: [[Int]] = []
        for tokens in promptTokens {
            cont.append(quantizedSerialGreedyDecode(promptTokens: tokens, maxTokens: steps + 1) ?? [])
        }
        let usableSteps = min(steps, (cont.map { $0.count }.min() ?? 1) - 1)
        guard usableSteps >= 1 else { return nil }

        // int8 serial teacher-forced logits per row.
        var serialLogits: [[MLXArray]] = []
        for (i, tokens) in promptTokens.enumerated() {
            let sc = makeQuantizedKVCaches(numLayers: model.numLayers)
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

        // int8 batched teacher-forced.
        var lengths: [Int] = []
        var perRowCaches: [[QuantizedKVCache]] = []
        for tokens in promptTokens {
            let bc = makeQuantizedKVCaches(numLayers: model.numLayers)
            let pl = prefill(MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count), bc)
            MLX.eval(pl)
            perRowCaches.append(bc)
            lengths.append(tokens.count)
        }
        let sMax = lengths.max() ?? 0
        let caches = stackCachesLeftPaddedQuantized(perRow: perRowCaches, lengths: lengths)
        var maxDiff = [Float](repeating: 0, count: R)
        for k in 0 ..< usableSteps {
            let rowOffsets = (0 ..< R).map { lengths[$0] + k }
            let mask = buildBatchedDecodeMask(
                lengths: lengths, sMax: sMax, totalLen: sMax + k + 1, dtype: .float16)
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

    /// int8 reference B=1 greedy decode (same model, quantized caches, NO
    /// batching). Mirrors ``serialGreedyDecode`` but allocates
    /// `QuantizedKVCache`s, so it is the solo reference the quantized batched
    /// gates teacher-force against.
    func quantizedSerialGreedyDecode(promptTokens: [Int], maxTokens: Int) -> [Int]? {
        guard let model = loadedModelForBatching, let eosId = tokenizerEOS,
              model.batchedDecodeForwardQuantized != nil else { return nil }
        let caches = makeQuantizedKVCaches(numLayers: model.numLayers)
        let sampler = Sampler(params: .greedy)
        let prefill = model.prefillForward ?? model.forward
        let input = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, promptTokens.count)
        var logits = prefill(input, caches)
        MLX.eval(logits)
        var tok = sampler.sample(logits)
        var gen = [tok]
        var done = tok == eosId
        while gen.count < maxTokens && !done {
            logits = model.forward(MLXArray([Int32(tok)]).reshaped(1, 1), caches)
            MLX.eval(logits)
            tok = sampler.sample(logits)
            if tok == eosId { done = true } else { gen.append(tok) }
        }
        return gen
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

    /// Robust, prompt-independent correctness probe for shared-prefix (partial)
    /// KV reuse. Builds the KV two ways over the fp16 cache and returns the max
    /// per-element |diff| of the resulting per-layer caches plus the max |cold|
    /// value (for relative scale):
    ///   - COLD: prefill `prefix + suffix` in one forward.
    ///   - PARTIAL: prefill the FULL `prefix + suffix` (the engine stores the
    ///     whole prompt), snapshot, restore into fresh caches, truncate back to
    ///     `prefix.count` (so the truncate really trims the stored suffix), then
    ///     re-forward only `suffix` - exactly the restore/truncate/suffix-forward
    ///     the engine's partial-reuse path runs when a later request shares this
    ///     prefix.
    /// SCOPE: this measures the cache the DONOR (non-shared) layers write. It
    /// gates the restore + truncate + suffix-forward orchestration: a bad restore
    /// or a wrong truncate length corrupts the donor K/V at O(1) relative. It
    /// does NOT exercise Gemma 4's shared-layer suffix-Q RoPE offset - that fix
    /// only rotates the KV-shared layers' QUERY, which is never written to any
    /// cache (and those layers' own caches are empty, so they are skipped here).
    /// The offset fix is gated separately by the first-decoded-token check in
    /// `Gemma4PartialReuseLiveTests`.
    /// A mathematically exact path yields identical caches. The residual is the
    /// model's own GEMM rounding: fp16 families (Llama/Qwen) round at ~1e-3
    /// relative, so their greedy output stays byte-identical; Gemma 4 computes in
    /// bf16, so a length-dependent suffix GEMM rounds at a few percent relative,
    /// which is numerically correct but can flip a downstream greedy tie. Returns
    /// nil if no model is loaded.
    func partialPrefillCacheMaxDiff(
        prefix: [Int], suffix: [Int]
    ) -> (maxDiff: Float, maxCold: Float)? {
        guard let model = loadedModelForBatching, !suffix.isEmpty else { return nil }
        let prefill = model.prefillForward ?? model.forward
        let full = prefix + suffix

        let coldCaches = makeKVCaches(numLayers: model.numLayers)
        _ = prefill(MLXArray(full.map { Int32($0) }).reshaped(1, full.count), coldCaches)

        // Prime on the FULL prompt so the restore carries a stored suffix that
        // `truncate(to: prefix.count)` actually trims (mirrors a stored entry
        // longer than the shared prefix).
        let primeCaches = makeKVCaches(numLayers: model.numLayers)
        _ = prefill(MLXArray(full.map { Int32($0) }).reshaped(1, full.count), primeCaches)
        let partialCaches = makeKVCaches(numLayers: model.numLayers)
        for (i, c) in primeCaches.enumerated() {
            if let snap = c.snapshot() {
                partialCaches[i].restore(keys: snap.keys, values: snap.values)
            }
        }
        for c in partialCaches { c.truncate(to: prefix.count) }
        _ = prefill(MLXArray(suffix.map { Int32($0) }).reshaped(1, suffix.count), partialCaches)

        var maxDiff: Float = 0
        var maxCold: Float = 0
        for i in 0 ..< coldCaches.count {
            guard let cs = coldCaches[i].snapshot(), let ps = partialCaches[i].snapshot(),
                  cs.keys.dim(2) == ps.keys.dim(2) else { continue }
            maxDiff = Swift.max(maxDiff,
                MLX.max(MLX.abs(cs.keys - ps.keys)).item(Float.self),
                MLX.max(MLX.abs(cs.values - ps.values)).item(Float.self))
            maxCold = Swift.max(maxCold,
                MLX.max(MLX.abs(cs.keys)).item(Float.self),
                MLX.max(MLX.abs(cs.values)).item(Float.self))
        }
        return (maxDiff, maxCold)
    }
}
