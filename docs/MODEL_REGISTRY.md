# Model Capability And Support-Tier Registry (WS3)

Status: foundational metadata layer landed. Full adapter polymorphism
(per-family `ModelAdapter` runtime contract) is intentionally NOT in
this PR; that is a follow-up once the metadata layer is in use across
call sites.

## Why this exists

KrillLM previously answered "can this model handle X" by sprinkling
`family == .gemma4` checks across the engine, the server's media
gating, and the Ollama-compatibility shim. That works for one
multimodal family. It does not scale.

`ModelCapabilities` is the single source of truth for:

- What every supported model family is allowed to do at runtime.
- What support tier each family ships at.
- What Ollama-compatible capability tags the discovery endpoints
  emit.

When something is wrong about a family's support claim, only one place
needs to change.

## Capability set

`Capability` (`Sources/KLMRegistry/ModelCapabilities.swift`):

```text
textGeneration   - causal LM
visionInput      - image input via native multimodal forward
audioInput       - audio input via native multimodal forward
embeddings       - vector embeddings via the embedding engine
tools            - parity-tested native tool chat template
structuredOutput - reserved; not declared until per-family parity gates
moe              - reserved; declared once WS6 lands a runtime
reranker         - reserved; declared once WS7's reranker subtrack lands
```

## Support tier

`SupportTier`:

```text
production_native   - Swift + MLX/Metal path, tests, docs, benchmark gate
compatible_fallback - bridge / slower reference path. NOT a perf claim
experimental        - runs on known fixtures, lacks full gates
unsupported         - explicit error before execution
```

Every existing family in this build is `production_native`. Promotion
requires tests + benchmarks + docs. The registry MUST NOT silently
upgrade a family.

## Wiring

| Caller                          | What it consults                          |
| ------------------------------- | ----------------------------------------- |
| `InferenceEngine.capabilities`  | `ModelCapabilities.capabilities(for:)` + checkpoint facts (e.g. text-only Gemma 4 dumps revoke vision/audio) |
| `InferenceEngine.supportsNativeImage` / `supportsAudio` | derived from `capabilities` |
| `Server` pre-generation media gate | calls `engine.supportsNativeImage` / `engine.supportsAudio` |
| `OllamaCompat.capabilities(for:)` (used by `/api/show`, `/api/tags`) | `ModelCapabilities.capabilities(for:).map(\.ollamaTag)` |
| `OllamaCompat.showPayload`      | adds `support_tier` key derived from `ModelCapabilities.supportTier(for:)` |

Existing call sites that asked "is this Gemma 4?" to decide whether to
accept image / audio now read through `capabilities`. Other Gemma-4
specific branches (e.g. chat-template selection, audio token IDs) remain
family-keyed; those are model-mechanics decisions, not capability
decisions, so the family check is still the right primitive.

## What this PR does NOT do

- It does not introduce a runtime `ModelAdapter` polymorphic dispatch.
  Adding one in the same PR would force every loader rewrite for an
  abstraction whose first user is not in tree yet.
- It does not change the chat template selection path. That stays
  family-keyed for now; once a second native VLM lands (WS5), the
  template-selection branch is the right place to introduce adapter
  dispatch.
- It does not change per-family chat templates or any model-level
  Swift code.

## What it unblocks

- WS4 (new dense text families) can add an entry to the capability
  switch + an alias and inherit `production_native` once the
  acceptance gates pass.
- WS5 (second native vision family) gets a clean home for its
  `visionInput` declaration without piling onto the engine's Gemma 4
  branch.
- WS6 (MoE) and WS7 (reranker) already have their capability cases
  reserved; landing them requires no registry shape change.
- Server can grow `/api/show` capability metadata that clients can
  programmatically read instead of inferring from family strings.
