# Krill Backlog

Deferred-but-tracked engineering items. Each entry is something we consciously chose
*not* to do in the PR that surfaced it, with enough context to pick it up later. This is
the in-repo companion to the owner's out-of-repo board
(`~/.claude/plans/krill-pending-work.md`), scoped to code-level follow-ups.

---

## `krill launch` roster: remaining agents + live verification

**Status:** core DONE (PRs #161-#164). `krill launch <agent>` ships with
claude, codex, opencode, hermes, pi, copilot, droid. Follow-ups consciously
deferred:

- **Live-verify hermes / pi / copilot / droid.** Only claude/codex/opencode
  were installed to test end-to-end (codex did a real `/v1/responses`
  round-trip). The other four follow each tool's documented OpenAI-compatible
  config but were not run against a real binary; verify their exact env/config
  schema once installed and tweak the `AgentProfile` literal if a version
  drifted. Profiles live in `Sources/KLMCLI/AgentProfiles.swift`.
- **`codex-app` (Codex desktop).** The GUI reads the real `~/.codex/config.toml`
  (it does not inherit a shell `CODEX_HOME`), so wiring it means merging a
  provider + profile into the user's file AND setting their default provider -
  invasive to do silently. Needs a careful TOML block-merge mode (append-if-
  absent + `.bak`) and `open -a Codex` launch semantics. Documented as manual
  in `docs/CONNECT_CODING_AGENTS.md` for now.
- **`openclaw`.** Config surface not verified (reportedly Pi-stack based);
  shipping guessed wiring would be worse than the documented manual path.
- **Streaming granularity (optional).** `/v1/responses` and `/v1/messages` are
  buffered (one delta) like the chat tool path; token-incremental Responses
  deltas would mirror `handleStreamingCompletion`. Codex tolerates the current
  granularity, so this is polish, not a blocker.

## Gemma 4 partial-prefix (shared-prefix) KV reuse

**Status:** DONE - all paths. SERIAL fp16 (PR #156), int8-KV SERIAL (PR #157),
honest bf16 gate (PR #158), and the CONCURRENT BATCHED path (fp16 +
int8-quantized, this PR) all reuse shared prefixes for Gemma 4.

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

**Concurrent batched path - DONE (this PR).** It turned out NOT to need any
batched-decode change: the batched per-row prefill runs through the SERIAL
forward (`prefillForward`), so the shared-layer suffix-Q offset fix already
applies. So the only changes were to drop the `family != "gemma4"` gate in
`makeBatchedPrefillRow` (fp16) and add the same partial branch to
`makeBatchedPrefillRowQuantized` (int8, the closure Gemma 4 actually uses). The
stacked decode reads the resulting per-row caches unchanged. Gates:
`BatchedDecodeLiveTests.testBatchedPartialPrefixReuseMatchesColdDecode` (now runs
Gemma 4 too, first-token gate; dense stays bit-for-bit) and
`testInt8BatchedPartialPrefixReuseEngages`. Both use an isolated per-test prefix
cache - the default on-disk cache hydrates full-match lookups from disk, so a
stale entry from a prior run could mask a real partial-reuse miss.

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

---

## Native Swift+MLX `krill quantize` (drop the Python shell-out)

**Status:** DONE. `krill quantize` is now pure Swift+MLX
(`Sources/KLMCore/CheckpointQuantizer.swift`) - the python3/mlx_lm.convert
shell-out is gone, so the shipped binary has no Python anywhere. Output is
**byte-identical to the canonical MLX op** (affine: 1007/1007 vs mlx-community
4-bit, `tools/verify_native_quantize_parity.sh`; nvfp4: 765/765 vs
`mx.quantize(mode:"nvfp4")`, `tools/verify_native_quantize_nvfp4.sh`).
The dense (no-reference) pass supports dense text families
(Llama/Qwen/Mistral/Phi/GLM/Glm4) by quantizing every 2-D divisible `.weight`.
**MoE / vision / Gemma are now supported via `--reference <a 4-bit build>`**: the
quantizer learns the EXACT quantized-module set from the reference checkpoint's
`.scales` (the generalization of `tools/requant_gemma4_nvfp4.py`), so it
reproduces the proven mlx-community coverage for any family without
reverse-engineering each loader predicate - DeepSeek's float router gate, the
Qwen2.5-VL float vision tower, and the Gemma PLE/tied-head all fall out
automatically. Stacked 3-D experts are quantized affine at the config's
top-level group, mirroring the MoE runtime which reconstructs born-quantized
experts affine from the top-level group (and reads no override for them); since
affine supports only group 32/64/128, an nvfp4 MoE (group 16) is rejected up
front (use `--mode affine` group 64 or `--mode mxfp4`/`mxfp8` group 32 for MoE).
`--protect <substr>` raises chosen 2-D modules to a higher precision (the Gemma
vision/audio projectors auto-protect at 8-bit affine, the color-fidelity fix),
recorded as per-module overrides the loader resolves via `q.effective(path)`. Gated byte-identical vs a fresh
`mx.quantize` recomputed from the same bf16 source at each module's effective
precision (`tools/verify_native_quantize_reference.sh`; gemma-4-12b nvfp4 + 8-bit
projectors). `--mode` affine/nvfp4/mxfp4/mxfp8 (the float formats auto-pick their
required group size). `--dtype` default fp16. Remaining: real-MoE byte-gating
needs a bf16 MoE source downloaded (the path is unit-tested synthetically).
Original write-up kept below for context.

**Original write-up (deferred):** The only Python touchpoint left in the
*shipped binary*. Everything in the inference / serving / model-load /
embedding / audio / vision path is already pure Swift+MLX with no sidecar; the
historical mlx-lm bridge was fully retired.

**Context.** `Sources/KLMCLI/QuantizeCommand.swift` implements `krill quantize`
by shelling out to `python3 -c "import mlx_lm; mlx_lm.convert(...)"`. It is an
optional, offline, manually-invoked model-prep utility -- it is NEVER auto-called
by load or serve (only registered as a CLI subcommand in `KrillCLI.swift`), and
it degrades gracefully (checks for python3 + mlx-lm, prints install hints if
absent). So it is not a runtime sidecar, but it is Python usage in the product
binary and the goal is fully-native, no Python anywhere outside tests/benchmarks.

**Proposed work.** Reimplement the convert/quantize in Swift+MLX:
- Load the HF safetensors + config via `swift-transformers` (already a dep).
- Quantize the eligible linear weights with mlx-swift's quantization primitives
  (`MLX.quantized` / `QuantizedLinear`), honoring `--bits` / `--group-size` and
  the same skip rules mlx-lm applies (e.g. do not quantize tiny / non-divisible
  tensors; keep embeddings/norms as configured).
- Write the MLX-format weights + an updated `config.json` (with the
  `quantization` block) into the registry path, matching the on-disk layout the
  native loader already reads.
- Gate it: quantize a small model both ways (this native path vs
  `mlx_lm.convert`) and assert the produced weights load and yield logit parity
  within the bf16/quant tolerance the loader already tolerates.

**Why deferred.** Inference is already 100% native; this is the last cosmetic
Python dependency and it sits on a cold, optional prep path, so it is low-urgency
relative to runtime work. Picking it up makes the shipped binary Python-free.

---

## Batcher-side stall monitor + spec->overlap handoff (concurrent n-gram default-on)

**Status:** deferred follow-up to PR #232 (single-stream adaptive n-gram spec).

**Context.** PR #232 made n-gram (prompt-lookup) speculative decode the default
on the SINGLE stream, with a stall monitor that hands off to the byte-exact
overlap pipeline loop when the lookup stops paying off (non-echo workloads). The
CONCURRENT batcher's n-gram spec path was deliberately left as an explicit
opt-in (`batcherNgramSpec`, default false): its spec round
(`ContinuousBatcher.decodeSpecRound`) bypasses the +9-11% CPU/GPU overlap
pipeline (PR #231) and has no equivalent stall->overlap handoff, so enabling it
by default would regress non-echo CONCURRENT throughput. Today the batcher's
default path keeps its byte-exact overlap win for everyone.

**Proposed work.** Give the batcher the same adaptiveness the single stream has:
- Per-row stall monitors (each row already owns a `NgramProposer`; PR #232 added
  the `recordRound` / sticky `stalled` machinery -- reuse it).
- When all (or enough) live rows in an epoch have stalled, switch that epoch from
  `decodeSpecRound` back to the overlap pipeline path (the epoch already
  re-stacks the live rows each round, so the switch is at an epoch boundary --
  no mid-forward surgery). Mixed epochs (some rows still echoing) keep the spec
  round, which degenerates to W=1 for the stalled rows.
- Then flip the batcher default to on, gated by the same `ngram_spec` config.

**Why deferred.** The single-stream win is the high-value, low-risk half and is
shipped. The batcher needs the epoch-level path-switch logic above to avoid
regressing the proven concurrent overlap, so it is a separate, carefully-gated
piece of work.
