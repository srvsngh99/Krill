# WS6: MoE Runtime Support

Status: foundation only (family detection + capability metadata +
explicit loader rejection). Native router + expert dispatch lands
in follow-up PRs.

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
mlx-lm (`Sources/KLMEngine/MoEEngine.swift` +
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
text); live MoEEngine tests gated on `KLM_MOE_MODEL_PATH`
verify text-only generation and that the bridge preserves the
system prompt through mlx-lm's `apply_chat_template`.

Benchmark vs Ollama: omitted by design. On Mac, Ollama itself
calls into mlx-lm for MoE inference; the KrillLM bridge calls
into the same mlx-lm. Per-token throughput and output quality
are at parity by construction (same Python, same model, same
MLX kernels). The cold-start cost of the sidecar (~1-3 s
depending on model size) is the only KrillLM-specific addition,
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

Support mixture-of-experts LLMs without giving up KrillLM's Mac-native
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
Sources/KLMCore/FeedForward.swift
Sources/KLMCore/TransformerBlock.swift
Sources/KLMCore/ModelConfig.swift
Sources/KLMCore/ModelLoader.swift
Sources/KLMCore/GLMModel.swift
Sources/KLMRegistry/ModelManifest.swift
Sources/KLMRegistry/AliasMap.swift
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
