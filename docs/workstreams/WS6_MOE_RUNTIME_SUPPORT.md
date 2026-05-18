# WS6: MoE Runtime Support

Status: planned

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
