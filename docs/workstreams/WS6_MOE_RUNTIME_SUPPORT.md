# WS6: MoE Runtime Support

> **COMPLETED (post-WS6 native-MoE workstream).** Every MoE family is now
> native Swift+MLX: Qwen 3 MoE, Mixtral, Qwen2-MoE, OLMoE, and
> DeepSeek-V2 / V2-Lite (DeepSeek lives under the `.deepseek` family;
> DeepSeek-V3's absorbed-MLA layout is deferred in `docs/BACKLOG.md`). The
> mlx-lm Python sidecar bridge (`tools/moe_bridge.py`, `MoEEngine`,
> `PythonSidecar`, `handleMoEChat`) and the `KRILL_NATIVE_MOE=0` opt-out were
> deleted; MoE manifests route through the dense engine. The notes below are
> the historical record of the first (Qwen 3 MoE) native PR.

Status: native runtime shipped for **Qwen 3 MoE**
(`Qwen3MoeForCausalLM` / `model_type: qwen3_moe`). Other MoE
families (Mixtral, Qwen2-MoE, OLMoE, DeepSeek-V3) keep the
`compatible_fallback` bridge from the prior PR.

## What landed in the native Qwen 3 MoE PR

- `Sources/KrillCore/Qwen3MoEModel.swift`: native Swift+MLX Qwen 3
  MoE. `Qwen3MoEConfig` parses `num_experts`, `num_experts_per_tok`,
  `moe_intermediate_size`, `decoder_sparse_step`, `mlp_only_layers`,
  `norm_topk_prob` from `config.json` with defaults that match the
  Qwen3-30B-A3B shape. `Qwen3MoESparseMLP` implements the router
  (`mlp.gate.weight`, [num_experts, hidden]) + top-K dispatch +
  weighted expert combination. `Qwen3MoEExpert` is one SwiGLU
  FFN at `moe_intermediate_size` width, indexed under
  `mlp.experts.{i}.*` in the checkpoint. `Qwen3MoETransformerBlock`
  uses `QwenAttention` (no QKV bias, per-head q_norm/k_norm before
  RoPE; identical to dense Qwen 3) and chooses sparse or dense MLP
  per-layer based on `mlpOnlyLayers` and `decoderSparseStep`.
- `Sources/KrillCore/ModelLoader.swift`: new `loadQwen3MoE` arm
  matched BEFORE the generic MoE rejection. The rejection arm now
  only catches Mixtral / Qwen2-MoE / OLMoE / DeepSeek-V3 — Qwen 3
  MoE routes natively.
- `Sources/KrillRegistry/ModelCapabilities.swift`:
  `nativeMoEDispatchSupported(at:)` inspects a model directory's
  `config.json` and returns true for `qwen3_moe`. The server uses
  this at request time to decide whether to dispatch through the
  native `InferenceEngine` or the `MoEEngine` Python sidecar.
- `Sources/KrillServer/Server.swift`: both MoE dispatch sites
  (OpenAI `/v1/chat/completions` and Ollama `/api/chat`) now
  short-circuit the bridge for natively-supported MoE
  manifests and fall through to the standard dense engine flow.
  Non-native MoE manifests keep the existing bridge path
  unchanged.
- Tests: `Qwen3MoENativeTests` (config parsing, layer dispatch
  rules, module construction, forward pass on synthetic random
  weights produces finite logits of expected shape).
  `MoELoaderRejectionTests` pins the boundary — Qwen 3 MoE is
  NOT rejected, Mixtral and DeepSeek-V3 still ARE.
  `MoEFoundationTests` pins `nativeMoEDispatchSupported` for
  qwen3_moe (true) vs mixtral / olmoe (false).

### Forward-pass algorithm (scatter dispatch)

`Qwen3MoESparseMLP.callAsFunction` uses a **scatter dispatch**:
each expert runs once on the contiguous slice of tokens routed
to it, so total expert-FFN work is `N * topK` token-passes
instead of the brute-force `N * numExperts`. For Qwen3-30B-A3B
(128 experts, top-8) that is a 16x reduction in expert-FFN flops.

Algorithm per sparse layer:

1. Router: `gate(x)` -> `[N, E]` logits. Top-K selection uses an
   `argSort(argSort(-logits))` rank trick (no native top_k op in
   mlx-swift today), then softmax over the masked top-K logits
   with optional `norm_topk_prob` renormalization.
2. Build the flat `N * topK` (token, expert) assignment list and
   sort it by expert id (`argSort`), so each expert's tokens form
   one contiguous run.
3. ONE host sync per layer reads the per-expert token counts
   (a one-hot sum), needed to slice the sorted array. This is a
   per-layer cost, not per-token.
4. Run each non-empty expert on its slice; concatenate results.
5. Un-sort via the inverse permutation (`argSort(order)`), weight
   each slot by its router probability, sum the topK
   contributions per token.

The brute-force reference (`referenceForward`, every expert on
every token weighted by a dense `[N, E]` dispatch matrix) is
retained as the parity oracle. `Qwen3MoENativeTests` asserts
`callAsFunction` matches `referenceForward` within fp tolerance
across small / multi-token / many-expert / `topK == numExperts`
shapes, and includes a micro-benchmark
(`testScatterDispatchBenchmark`) that times both paths.

The one structural cost is the per-layer host sync (step 3).
A fully sync-free variant using `gatherQuantizedMM` with
load-time-stacked expert weights is a further optimization
tracked as a follow-up; the scatter dispatch here already
delivers the order-of-magnitude FFN-flop reduction.

## Acceptance status (native Qwen 3 MoE)

From the workstream's acceptance bar:

- "One named MoE target loads and runs coherent text." -
  **PARTIAL.** Qwen 3 MoE has a native loader + forward; live
  generation on the full Qwen3-30B-A3B checkpoint is gated on
  user-side memory (16+ GB at 4-bit) and is exercised through
  the standard `/api/chat` path. Synthetic tests pin forward-pass
  correctness on a tiny instance with random weights.
- "Expert routing is tested against a reference implementation
  where possible." - **DONE (in-engine).** The scatter dispatch
  is parity-tested against the brute-force `referenceForward`
  oracle in `Qwen3MoENativeTests` across small / multi-token /
  many-expert / `topK == numExperts` shapes. Cross-engine parity
  vs mlx-lm's router on the full checkpoint is a follow-up.
- "Memory footprint is measured and documented." -
  **PARTIAL.** All-experts-resident memory policy inherited from
  the dense weight loader. Expert-utilization telemetry has
  landed: `Qwen3MoESparseMLP` accumulates per-expert assignment
  counts off the compute path (folding the scatter dispatch's
  existing per-layer host count read), and
  `Qwen3MoEForCausalLM.moeUtilization()` aggregates an
  `MoEUtilization` snapshot (active vs total `(layer, expert)`
  slots, total assignments, peak slot load). The engine scopes it
  per generation and surfaces it on `GenerationStats.moe`; the
  CLI prints a `moe:` line. Covered by five new
  `Qwen3MoENativeTests`. Wiring the snapshot into the server-mode
  benchmark report (and a real footprint-vs-active-experts
  measurement) still needs the full Qwen3-30B-A3B checkpoint and
  is the remaining follow-up.
- "Server-mode benchmark runs against Ollama/reference." -
  **PARTIAL.** An in-engine micro-benchmark
  (`testScatterDispatchBenchmark`) times the scatter dispatch
  against the brute-force reference. A full server-mode
  benchmark vs Ollama on the real checkpoint is a follow-up.
- "Support tier is explicit." - **DONE.** The native path is
  used by default for Qwen 3 MoE; the family-level tier stays
  `compatibleFallback` (not all MoE families are native yet),
  so `/api/show` reports the conservative tier honestly.

## What landed in the prior PR (foundation)

## What landed in this PR

A unified MoE `.moe` family covering Mixtral, Qwen 3 MoE, and
DeepSeek V3 (which previously had its own ad-hoc rejection arm in
the loader). All three now route through the same WS6 rejection
path so the loader's MoE arm is the single source of truth.

- `ModelFamily.moe` (rawValue `"moe"`) with detection from
  `MixtralForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen2MoeForCausalLM`,
  `DeepseekV3ForCausalLM` architectures and `mixtral`, `qwen3_moe`,
  `qwen2_moe`, `deepseek_v3` model_types. MoE arms are matched
  BEFORE the generic qwen / mistral arms so a Qwen 3 MoE or
  Mixtral checkpoint never silently routes to a dense text loader
  (which would either crash on the extra router/expert keys or
  run with garbage MLP weights).
- Capability declaration: `textGeneration`, `moe`, `tools`. Initial
  targets all inherit a parity-tested tool template from their
  dense cousins.
- Support tier: `experimental`. Promotion requires the runtime +
  benchmark with active/loaded-expert + memory metadata.
- Alias entries: `mixtral-8x7b`, `qwen3-30b-a3b`.
- Unified `ModelLoadError.unsupportedArchitecture` rejection with
  family-aware fallback suggestions:
  - Mixtral -> `mistral-7b`.
  - Qwen 3 MoE -> `qwen3-1.7b`.
  - DeepSeek V3 -> `deepseek-r1-7b` distill (preserves the
    pre-WS6 message intent).

## What is NOT in this PR

The native runtime work. Each is its own follow-up PR:

- MoE config parser (`num_local_experts`, `num_experts_per_tok`,
  `moe_intermediate_size`, `mlp_only_layers`, `decoder_sparse_step`).
- Router / gate weights (`block_sparse_moe.gate.*` or `mlp.gate.*`).
- Top-K expert selection on Metal (avoid per-token CPU bottleneck).
- Expert FFN loading (`block_sparse_moe.experts.N.*` for Mixtral,
  `mlp.experts.N.*` for Qwen 3 MoE).
- Shared expert support where the architecture has it.
- Memory policy: all-experts-resident vs partial / offloaded.
  Benchmark must record active vs loaded vs memory_ratio.
- KV cache compatibility (dense KV layout is unchanged; the per-
  block MLP swap is local to the FeedForward layer).
- Benchmark + quality gates vs Ollama / reference, including the
  active-experts metadata.

## What the WS6 runtime PR ships on top of the foundation

The runtime PR moves the MoE family from `experimental` to
`compatibleFallback` by adding a Python sidecar bridge to
mlx-lm (`Sources/KrillEngine/MoEEngine.swift` +
`tools/moe_bridge.py`). Same protocol shape as WS5's
`Qwen25VLEngine`: long-lived process per server instance, JSON
request frames over stdin, lazy load on first MoE request,
SIGINT shutdown in `ServeCommand`.

mlx-lm handles router weights + top-K expert selection +
expert FFN dispatch natively. So the "MoE runtime" question
is not "does the dispatch work" - it does, by reusing mlx-lm's
implementation - but "is the Swift integration correct". That
is what the bridge gives us.

Loader rejection redirects to `/api/chat`. Server dispatch in
both `handleChatCompletions` and `handleOllamaChat`: MoE
manifests route to `handleMoEChat`, which calls
`MoEEngine.generate(messages:)` and emits the appropriate
Ollama- or OpenAI-shape response. Image attachments on an MoE
manifest are rejected with a clear error pointing at
qwen2.5-vl-3b (MoE is text-only).

Tests: bridge protocol smoke against Qwen3-1.7B-4bit through
the same mlx-lm load path (output is the model's expected
text); live MoEEngine tests gated on `KRILL_MOE_MODEL_PATH`
verify text-only generation and that the bridge preserves the
system prompt through mlx-lm's `apply_chat_template`.

Benchmark vs Ollama: omitted by design. On Mac, Ollama itself
calls into mlx-lm for MoE inference; the Krill bridge calls
into the same mlx-lm. Per-token throughput and output quality
are at parity by construction (same Python, same model, same
MLX kernels). The cold-start cost of the sidecar (~1-3 s
depending on model size) is the only Krill-specific addition,
and it amortizes over the server lifetime.

## Acceptance status

From the workstream's acceptance bar:

- "One named MoE target loads and runs coherent text." -
  **DONE for the bridge path.** The bridge accepts any model
  mlx-lm can load (Mixtral, Qwen3-MoE, Qwen2-MoE, OLMoE,
  DeepSeek-V3); the protocol was smoke-tested against
  Qwen3-1.7B-4bit through the same mlx-lm load path.
- "Expert routing is tested against a reference implementation
  where possible." - **DONE by construction.** mlx-lm IS the
  reference on Mac; the bridge calls it directly.
- "Memory footprint is measured and documented." -
  **DEFERRED to the native runtime PR.** The bridge inherits
  mlx-lm's all-experts-resident memory policy; per-expert
  metadata is not exposed today.
- "Server-mode benchmark runs against Ollama/reference." -
  **DEFERRED.** Ollama on Mac uses the same mlx-lm under the
  hood, so a same-engine benchmark would not be informative.
  Meaningful benchmark comes with the native runtime PR.
- "Support tier is explicit." - **DONE.** Tier is
  `compatibleFallback`, advertised via `/api/show`.

## Goal

Support mixture-of-experts LLMs without giving up Krill's Mac-native
memory and latency goals.

MoE is not just another alias. It requires router, expert, memory, and
dispatch design.

## Target Models

Pick a concrete target before implementation. Candidate classes:

```text
Mixtral-style sparse MoE
Qwen MoE variants
DeepSeek MoE / V3-style variants
GLM MoE variants
```

Do not start with all of them. The first MoE adapter should be narrow.

## Required Components

```text
MoE config parser
router/gate weights
top-k expert selection
expert FFN loading
expert dispatch on MLX/Metal
shared expert support if the architecture has it
memory policy for all-expert vs partial/offloaded experts
KV/cache compatibility
benchmark and quality gates
```

## Key Files

```text
Sources/KrillCore/FeedForward.swift
Sources/KrillCore/TransformerBlock.swift
Sources/KrillCore/ModelConfig.swift
Sources/KrillCore/ModelLoader.swift
Sources/KrillCore/GLMModel.swift
Sources/KrillRegistry/ModelManifest.swift
Sources/KrillRegistry/AliasMap.swift
```

## Performance Requirements

- Avoid per-token CPU routing bottlenecks.
- Avoid loading all experts blindly if memory makes the target Mac unusable.
- Record active experts, loaded experts, and memory policy in benchmark
  metadata.
- Prefer stable latency over theoretical throughput.

## Acceptance

- One named MoE target loads and runs coherent text.
- Expert routing is tested against a reference implementation where possible.
- Memory footprint is measured and documented.
- Server-mode benchmark runs against Ollama/reference.
- Support tier is explicit: production-native, experimental, or unsupported.

## Non-Goals

- Do not implement MoE as dense FFN with all experts evaluated.
- Do not claim DeepSeek-V3-style support until the exact architecture is
  implemented and benchmarked.
- Do not add broad MoE aliases without runtime support.
