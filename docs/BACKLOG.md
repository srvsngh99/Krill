# KrillLM Backlog

Deferred-but-tracked engineering items. Each entry is something we consciously chose
*not* to do in the PR that surfaced it, with enough context to pick it up later. This is
the in-repo companion to the owner's out-of-repo board
(`~/.claude/plans/krillm-pending-work.md`), scoped to code-level follow-ups.

---

## Gemma 4 partial-prefix (shared-prefix) KV reuse

**Status:** OPEN (surfaced 2026-06-04 by the agentic/RAG prefix-cache work, PRs
#148 serial / #151 concurrent batched).

Shared-prefix (longest-common-prefix) KV reuse is bit-exact for standard
per-layer caches (Llama, Qwen, Mistral, Phi, dense MoE) but is **excluded for
Gemma 4** (`InferenceEngine.swift:984` serial, `:2012` batched:
`family != "gemma4"`). The exclusion is **verified necessary**: temporarily
enabling it reuses (prefill 403 ms -> 10 ms) but produces **different greedy
output than a cold prefill** (non-bit-exact), so the naive gate flip ships WRONG
answers. Consequence: on a Gemma-4 agentic/RAG workload Ollama's prefix cache
currently wins (it caches the shared context; KrillLM re-prefills it). Full-MATCH
reuse still works for Gemma 4.

Prime suspect: Gemma 4's cross-layer KV sharing (`num_kv_shared_layers`) - shared
layers reuse a donor layer's K/V and derive their RoPE offset from the donor
cache length AFTER the donor appended the suffix (`Gemma4Model.swift:~445`), so a
multi-token suffix span is mis-rotated. Full-match (1-token re-forward) survives;
the S>1 partial span does not.

**Full plan + root-cause analysis + validation gates:**
`~/.claude/plans/krillm-gemma4-partial-prefix-reuse-handoff.md`. Do not flip the
gate without a byte-exact Gemma-4 reuse-vs-cold gate green.

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
