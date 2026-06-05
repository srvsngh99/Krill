# KrillLM Backlog

Deferred-but-tracked engineering items. Each entry is something we consciously chose
*not* to do in the PR that surfaced it, with enough context to pick it up later. This is
the in-repo companion to the owner's out-of-repo board
(`~/.claude/plans/krillm-pending-work.md`), scoped to code-level follow-ups.

---

## Gemma 4 partial-prefix (shared-prefix) KV reuse

**Status:** SERIAL fp16 path DONE (2026-06-04, PR #156). int8-KV SERIAL path
DONE (2026-06-05, this PR). Only the concurrent batched path remains OPEN
(see below).

Shared-prefix (longest-common-prefix) KV reuse is bit-exact for standard
per-layer caches (Llama, Qwen, Mistral, Phi, dense MoE). It is **now also
bit-exact for Gemma 4 on the default fp16 serial path.** The blocker was Gemma
4's cross-layer KV sharing: a shared layer holds an empty OWN cache and reuses
the donor's K/V, so the attention offset `cache?.sequenceLength` resolved to 0.
That is correct in a COLD prefill (the span starts at position 0, so offset 0 ==
true positions) but wrong for a PARTIAL-PREFIX resume, where the span is the
diverging SUFFIX at true positions `[LCP, count)` — offset 0 rotated the suffix
Q at `0..suffixLen-1`, misaligning it with the restored donor K (rotated at
their true positions) → RoPE mismatch → divergent greedy output.

**Fix (`Gemma4Model.swift`, `Gemma4Attention.callAsFunction`):** for a
multi-token forward (`L > 1`) whose donor cache already holds more than this
span (`donorLen > L`), the shared layer rotates Q at base `donorLen - L`. The
donor (a non-shared layer that ran earlier in the same forward) appended this
L-token span to its cache, so `donorLen - L` recovers the span's true base
position: 0 for a cold full prefill (unchanged), `LCP` for a partial resume.
The L==1 decode step and the single-token full-MATCH re-forward keep the legacy
offset-0 path untouched, so this changes behavior ONLY for a multi-token
partial-prefix resume.

Verified end-to-end on gemma-4-e2b: a 562-token shared-prefix request drops from
1001 ms (cold) to 158 ms (partial reuse) - Ollama-parity territory. The existing
Gemma 4 smoke + 11 batched-decode gates (incl. full-match replay) stay green
(cold/decode paths are provably unchanged).

**CORRECTNESS STANDARD (bf16). Gemma 4 partial reuse is numerically correct but
NOT strictly byte-exact, and this is fundamental, not a bug.** Gemma 4 computes
in bf16; a partial reuse forwards only the suffix, whose shorter GEMM rounds a
few percent differently than the cold full forward (measured ~0.05 max relative
on e2b vs ~0.0005 for fp16 Qwen). The reused KV cache is correct to within that
rounding, but it can flip a downstream greedy near-tie into an equally-valid
different continuation - the same bf16 behavior `BatchedDecodeLiveTests`'
teacher-forced gate already accepts (tolerance bound, not token equality). So the
DENSE families keep the strict byte-exact gate, and Gemma 4 is gated on the
robust, prompt-independent invariants instead (`Gemma4PartialReuseLiveTests`,
2026-06-06):
- `testCacheMatchesColdWithinBf16` - the restored+truncated+suffix-forwarded
  cache (what the DONOR, non-shared layers write) matches a cold prefill within a
  bf16 relative bound. A bad restore or wrong truncate length corrupts it at O(1)
  relative. It does NOT cover the shared-layer Q offset (Q is never cached; the
  empty KV-shared layers are skipped). New helper `partialPrefillCacheMaxDiff`.
- `testPartialReuseEngagesAndIsFaster` / `...Int8` - gates the shared-layer Q
  offset + engine wiring: the first decoded token (flows through the KV-shared
  layers; the pre-fix offset-0 path corrupted it grossly, not via a bf16
  tie-flip) matches cold and prefill is far faster.

(An earlier framing of these tests asserted full greedy-token equality and
passed only because the chosen prompt had no near-tie in-window; the robust
invariants above are stronger and prompt-independent.)

**int8-KV serial path - DONE (2026-06-05, this PR).** The shared-layer offset
fix is dtype-agnostic, so the only missing pieces were on the cache side:
`PrefixCache.storeQuantized` now retains the entry's tokens (it previously
discarded them, so a quantized entry could only serve a byte-identical full hit,
never a shared-prefix match), and a new `lookupLongestPrefixQuantized` mirrors
the fp16 LCP lookup over int8 storage. The engine gained a parallel int8 partial
branch (restore the quantized donor snapshots, truncate to the shared length,
forward the suffix). Quantization is per-token (each token carries its own
scale/zero), so a restored-then-truncated prefix is bit-identical to a cold int8
prefill of those tokens (the int8 quantization itself is exact-per-token; the
bf16 standard above still applies to the underlying compute). Gate:
`Gemma4PartialReuseLiveTests.testPartialReuseEngagesAndIsFasterInt8` plus
`QuantizedPrefixCacheTests` LCP units. The int8 batched full-match gates stay
green.

**Remaining (separate follow-up):**
- **Concurrent batched path** - `InferenceEngine.swift` `makeBatchedPrefillRow`
  still gates `family != "gemma4"`. Gemma 4 batches on the int8 quantized closure
  (`makeBatchedPrefillRowQuantized`), and the batched ragged-decode passes
  all-zero `rowOffsets` to shared layers; a batched partial resume needs the
  per-row shared-layer base wired through there too.

---

## Extract a shared MoE `SwitchGLU` / `QuantizedSwitchedLinear` module

**Status:** DONE — `Sources/KLMCore/MoESwitchGLU.swift` (`MoEQuantizedSwitchedLinear`
+ `MoESwitchGLU`, parameterized by `MoEActivation.{swiglu,geglu}`). All six families
(Qwen3-MoE, Gemma 4, Mixtral, Qwen2-MoE, OLMoE, DeepSeek-V2) refactored onto it; the
per-family copies are deleted (net −421 lines). Bit-exact gated: the four synthetic
mlx-lm logit-parity fixtures (Mixtral/OLMoE/Qwen2-MoE/DeepSeek-V2) pass post-refactor,
the quantized SwitchGLU sorted-vs-unsorted cover runs for both `.swiglu` and `.geglu`
(`MoESortPathTests`), and gemma-4-26b-a4b (GeGLU) + Qwen3-Coder-30B-A3B (SwiGLU) both
generate coherent output on the real checkpoints.

---

### Original write-up (kept for context)

**Status:** deferred (owner decision, native-MoE workstream).

**Context.** The `gatherQuantizedMM`-based expert dispatch (a stacked
`[E, O, I_packed]` quantized switched-linear per projection + a SwitchGLU that runs the
decode/prefill sort path) is currently **copy-pasted per family**:

- `Qwen3QuantizedSwitchedLinear` / `Qwen3SwitchGLU` - `Sources/KLMCore/Qwen3MoEModel.swift`
- `Gemma4QuantizedSwitchedLinear` / `Gemma4SwitchGLU` - `Sources/KLMCore/Gemma4Model.swift`
- one copy per new MoE family added in the native-MoE workstream (Mixtral, Qwen2-MoE,
  OLMoE, DeepSeek).

The only real difference between copies is the activation (SiLU/SwiGLU vs Gemma's GeGLU)
and a couple of router details. The `(token, expert)` sort path is *already* shared
(`Sources/KLMCore/MoESortPath.swift`) - this item is about sharing the rest.

**Proposed work.** Factor out a single generic `MoESwitchGLU` + `QuantizedSwitchedLinear`
in a dedicated file, parameterized by an activation closure (and any small per-family
knobs), then refactor Qwen3 + Gemma4 + the four new families onto it. Each refactor must
be gated by a **bit-exact** logit/output comparison against the pre-refactor module
(these are shipped, perf-critical paths - Qwen3-MoE and Gemma 4 both beat Ollama, and the
#85/#87 gather+sort wins must not regress).

**Why deferred.** The native-MoE workstream prioritizes breadth (port the four remaining
sidecar families) over consolidation. Copy-paste keeps each family PR self-contained and
zero-risk to the shipped Qwen3/Gemma4 paths. Consolidate once all families have landed
and their numerics are pinned by parity tests.

---

## DeepSeek-V3 absorbed-MLA attention + real-checkpoint verification

**Status:** DONE (synthetic parity); only the 671B real-checkpoint run is RAM-blocked here.

**Context.** The native DeepSeek runtime (`Sources/KLMCore/DeepSeekModel.swift`) serves
**DeepSeek-V2 / V2-Lite** end-to-end: MLA with the standard `kv_b_proj` decompression,
YaRN RoPE, the always-on shared expert, the `first_k_dense_replace` dense prefix, and the
V2 softmax / `group_limited_greedy` router. This is numerically verified against mlx-lm
(`DeepSeekParityTests`, V2 fixture: argmax + cosine > 0.9999 on identical 4-bit weights).

**What landed.** The absorbed V3 representation is now implemented and parity-verified:
- `DeepSeekV3Attention` + `DeepSeekQuantizedMultiLinear` (per-head `embed_q` / `unembed_out`
  via `quantizedMatmul` with the `transpose` flag, mirroring mlx-lm's `mla.MultiLinear`).
- A **latent KV cache** (the standard `KVCache` stores `kv_latent` as keys + `k_pe` as
  values, both with one head -- no new cache type needed; it already handles V2's
  differing key/value dims).
- The split attention: rope-slice `pe_scores` folded into the additive attention bias, the
  L==1 decode path (query projected into the latent, attention over the cached latent,
  `unembed_out` back to value space) and the L>1 prefill path (expand latent to per-head
  nope-keys / values).
- The `noaux_tc` group gate (`DeepSeekMoEGate`: sigmoid + `e_score_correction_bias` for
  selection, group top-2-sum, `norm_topk_prob`) is now exercised end-to-end (V2 never hit
  it -- `n_group=1`).
- `DeepSeekDecoderLayer` picks V2 vs V3 attention by `config.usesAbsorbedMLA`; the loader no
  longer rejects V3.

Verified by `tools/verify_deepseek_parity.py <dir> v3` -> `DeepSeekParityTests`
(`KLM_DEEPSEEK_V3_PARITY_DIR`): logit parity vs mlx-lm (argmax + cosine > 0.9999), plus a
decode-matches-prefill self-consistency test for the L==1 latent-cache path. The parity
tool randomizes the V3 router (its gate weight initializes to zero, which would make every
routed score tie at 0.5 and decide selection by tie-break artifact rather than numerics).
Only the 671B real V3 is RAM-blocked on a small host -- run the real-checkpoint parity on a
bigger box.
