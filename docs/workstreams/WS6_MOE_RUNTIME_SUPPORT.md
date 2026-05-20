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

## Acceptance status

From the workstream's acceptance bar:

- "One named MoE target loads and runs coherent text." -
  **PENDING** (follow-up PR).
- "Expert routing is tested against a reference implementation." -
  **PENDING**.
- "Memory footprint is measured and documented." - **PENDING**.
- "Server-mode benchmark runs against Ollama/reference." -
  **PENDING**.
- "Support tier is explicit." - **DONE** (family declared at
  `experimental` in the registry; the rejection error message
  names the tier and points users at the working dense
  alternatives).

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
