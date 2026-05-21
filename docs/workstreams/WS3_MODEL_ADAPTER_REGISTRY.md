# WS3: Model Adapter And Capability Registry

Status: capability metadata layer landed; server-side `ModelAdapter`
routing landed; load-time adapter polymorphism (`detect` / `load` /
tokenizer / cache) remains deferred.
Detailed usage: [../MODEL_REGISTRY.md](../MODEL_REGISTRY.md)

## What landed in the foundational PR (#25)

- `Capability` enum (textGeneration, visionInput, audioInput,
  embeddings, tools, structuredOutput, moe, reranker) and
  `SupportTier` enum (productionNative, compatibleFallback,
  experimental, unsupported) in `Sources/KLMRegistry/ModelCapabilities.swift`.
- Per-family capability set + support tier lookups, plus a stable
  `ollamaTag` mapping so the discovery endpoints emit Ollama-compatible
  capability identifiers.
- `InferenceEngine.capabilities` reads from the registry and revokes
  vision/audio at the engine layer when the loaded checkpoint has no
  multimodal forward. `supportsNativeImage` / `supportsAudio` now
  derive from the capability set instead of hardcoded family checks.
- `OllamaCompat.capabilities(for:)` reads from the registry; `/api/show`
  payload now exposes a `support_tier` field alongside the capability
  list.
- Tests pin the capability set per family and the tools-template
  helper consistency.

## What landed in the WS3 follow-up

The foundational PR deferred the `ModelAdapter` contract because its
first real users (a second native VLM, a native MoE runtime) were not
yet in tree. WS5 and WS6 have since landed, so the server-side adapter
is now warranted and shipped:

- `ModelAdapter` in `Sources/KLMRegistry/ModelAdapter.swift`: a
  `Sendable` value type that is the single declarative source of
  truth for a family's server-side contract -
  - `chatRouting` (`.denseEngine` / `.visionBridge` /
    `.mixtureOfExperts`),
  - `requiresImageInput` (text-only turns refused before a multi-GB
    sidecar starts),
  - `chatTemplate` (`ChatTemplatePolicy`: `hermes` / `gemma4` /
    `llama` / `qwen`),
  - `capabilities` / `supportTier` delegated to `ModelCapabilities`
    so each fact still has exactly one table.
- `Server.swift`: the duplicated `manifest.family == .qwen25vl` /
  `.moe` branches in `handleChatCompletions` and `handleOllamaChat`
  are replaced by one `dispatchFamilyChat` helper driven by
  `ModelAdapter.chatRouting`. Adding a bridge-backed family is now a
  registry change, not a new server branch.
- `ToolCalling.ToolFormat.forFamily` delegates the family→template
  decision to `ModelAdapter.chatTemplate`; the server module only
  maps the registry's module-neutral `ChatTemplatePolicy` onto its
  concrete renderer/parser.
- Tests pin `chatRouting`, `requiresImageInput`, and `chatTemplate`
  for every `ModelFamily`, and assert the `ToolFormat.forFamily`
  mapping is behaviour-preserving.

## What this still does NOT do

- No load-time adapter polymorphism. The WS3 design sketch also lists
  `detect`, `load`, `tokenizerPolicy`, and `cachePolicy`. Folding
  those in would force a rewrite of every loader and engine for an
  abstraction whose hot decode path must stay zero-cost. They remain
  in `ModelLoader` / the per-family engines and can adopt `ModelAdapter`
  incrementally.
- `Server.swift` still has a couple of family-keyed branches that are
  model-mechanics decisions (audio token IDs) rather than routing or
  capability decisions, and those stay family-keyed by design.

## Goal

Make model support explicit, scalable, and testable.

KrillLM should know the difference between:

```text
family detected
weights loadable
text generation supported
vision supported
audio supported
embeddings supported
MoE supported
production-native
fallback-only
experimental
unsupported
```

## Current Problem

Support is split across aliases, family detection, model loaders, server
guards, and docs. That works for a small set of families, but it will not
scale to many model types without unclear claims and risky conditionals.

## Proposed Shape

Introduce an adapter contract such as:

```text
ModelAdapter
  id
  families
  detect(config, files)
  capabilities
  load(config, weights)
  tokenizerPolicy
  chatTemplatePolicy
  cachePolicy
  benchmarkProfile
```

Capabilities should be explicit:

```text
textGeneration
visionInput
audioInput
embeddings
reranking
tools
structuredOutput
moe
serverSafe
productionNative
fallbackOnly
experimental
```

## Key Files

```text
Sources/KLMRegistry/AliasMap.swift
Sources/KLMRegistry/ModelManifest.swift
Sources/KLMCore/ModelLoader.swift
Sources/KLMEngine/InferenceEngine.swift
Sources/KLMServer/Server.swift
Sources/KLMServer/ServerMultimodal.swift
docs/ARCHITECTURE.md
docs/ADDING_MODELS.md
```

## Implementation Phases

1. Define support-tier and capability types.
2. Populate current families without changing behavior.
3. Replace ad hoc `family == "gemma4"` style checks where safe.
4. Expose capability information in `/api/tags`, `/api/show`, and docs.
5. Add tests that unsupported media fails before generation.
6. Make model promotion require capability and benchmark metadata.

## Acceptance

- Every alias has declared capabilities and support tier.
- Server errors explain unsupported model/media combinations.
- `/api/show` exposes enough capability metadata for clients.
- Docs can be generated or checked against the registry.
- Adding a new family starts by adding an adapter, not scattered switches.

## Non-Goals

- Do not rewrite all model code at once.
- Do not make the adapter abstraction slower on the hot decode path.
- Do not promote existing experimental paths to production by metadata only.
