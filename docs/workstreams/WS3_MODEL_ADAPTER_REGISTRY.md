# WS3: Model Adapter And Capability Registry

Status: planned

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
