# KrillLM Backlog

Deferred-but-tracked engineering items. Each entry is something we consciously chose
*not* to do in the PR that surfaced it, with enough context to pick it up later. This is
the in-repo companion to the owner's out-of-repo board
(`~/.claude/plans/krillm-pending-work.md`), scoped to code-level follow-ups.

---

## Extract a shared MoE `SwitchGLU` / `QuantizedSwitchedLinear` module

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

**Status:** deferred (RAM-blocked here; distinct attention representation).

**Context.** The native DeepSeek runtime (`Sources/KLMCore/DeepSeekModel.swift`) serves
**DeepSeek-V2 / V2-Lite** end-to-end: MLA with the standard `kv_b_proj` decompression,
YaRN RoPE, the always-on shared expert, the `first_k_dense_replace` dense prefix, and the
V2 softmax / `group_limited_greedy` router. This is numerically verified against mlx-lm
(`DeepSeekParityTests`, V2 fixture: argmax + cosine > 0.9999 on identical 4-bit weights).

The **V3 router gating** (`noaux_tc`: sigmoid scores + `e_score_correction_bias` for
selection + group top-2-sum + `norm_topk_prob`) is implemented in the shared
`DeepSeekMoEGate` and structurally tested.

**What's left.** mlx-lm's `deepseek_v3` uses an *absorbed* MLA representation that is
structurally different from V2: per-head `MultiLinear` `embed_q` / `unembed_out` weights
(instead of `kv_b_proj`), a **latent KV cache** (it caches `kv_latent` + `k_pe`, not full
keys/values), and a split attention with separate `pe_scores` plus distinct L==1 (decode)
vs L>1 (prefill) code paths. A real mlx-community DeepSeek-V3 checkpoint ships
`embed_q`/`unembed_out`, so the V2 loader rejects it with a clear message
(`loadDeepSeek` -> `usesAbsorbedMLA`). Implementing it requires a `MultiLinear` (per-head)
layer, a latent-KV cache variant (the current `KVCache` stores `[B, H, L, headDim]`
keys/values), and the absorbed forward with the L==1 / L>1 branches and manual rope-pe
attention.

It is verifiable against mlx-lm on a tiny synthetic V3 fixture
(`tools/verify_deepseek_parity.py <dir> v3`), but the 671B real V3 is RAM-blocked on this
24 GB host - run the real-checkpoint parity on a bigger box.
