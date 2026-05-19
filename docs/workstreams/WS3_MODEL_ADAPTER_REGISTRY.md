# WS3: Model Adapter And Capability Registry

Status: foundational metadata layer landed; full ModelAdapter
polymorphism deferred.
Detailed usage: [../MODEL_REGISTRY.md](../MODEL_REGISTRY.md)

## What landed in this PR

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

## What this does NOT do

- No runtime `ModelAdapter` polymorphic contract. Adding one in the
  same PR would force every loader rewrite for an abstraction whose
  first user is not in tree yet. Once WS5 lands a second native VLM,
  the template-selection branch is the right place to introduce
  adapter dispatch.
- The Server.swift still has a couple of family-keyed branches that
  are model-mechanics decisions (chat template, audio token IDs)
  rather than capability decisions, and those stay family-keyed.

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
