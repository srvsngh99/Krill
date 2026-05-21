# Model Capability And Support-Tier Registry (WS3)

Status: metadata layer landed; the server-side `ModelAdapter` runtime
contract (chat routing + chat-template policy) has since landed on top
of it. Load-time adapter polymorphism (`detect` / `load` / tokenizer /
cache) remains a deliberate follow-up — see "What this does NOT do".

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
| `Server.dispatchFamilyChat`     | `ModelAdapter(family:).chatRouting` — picks the dense / VLM-bridge / MoE handler |
| `ToolCalling.ToolFormat.forFamily` | `ModelAdapter(family:).chatTemplate` — picks the tool chat template |

Existing call sites that asked "is this Gemma 4?" to decide whether to
accept image / audio read through `capabilities`. The server's chat
*routing* and *tool-template* selection — formerly duplicated
`manifest.family == .qwen25vl` / `.moe` branches — now read through
`ModelAdapter` (see below). The remaining family-keyed branches (audio
token IDs) are low-level model-mechanics decisions, so the family check
is still the right primitive there.

## The `ModelAdapter` runtime contract

`ModelAdapter` (`Sources/KLMRegistry/ModelAdapter.swift`) is the single
declarative source of truth for a family's *server-side* contract:

- `chatRouting` — `.denseEngine` / `.visionBridge` / `.mixtureOfExperts`.
- `requiresImageInput` — whether a text-only turn is refused up front.
- `chatTemplate` — `ChatTemplatePolicy` (`hermes` / `gemma4` / `llama` /
  `qwen`); KLMServer maps this onto its concrete renderer/parser.
- `capabilities` / `supportTier` — delegated to `ModelCapabilities`, so
  each fact still has exactly one table and `ModelAdapter` is the one
  type a caller consults.

Adding a bridge-backed or specially-routed family is now a registry
change (a `switch` case in `ModelAdapter`), not a new `Server.swift`
branch.

## What this does NOT do

- It does not introduce *load-time* adapter polymorphism. The WS3
  design sketch also lists `detect`, `load`, `tokenizerPolicy`, and
  `cachePolicy`; folding those in would force a rewrite of every loader
  and engine for an abstraction whose hot decode path must stay
  zero-cost. They remain in `ModelLoader` / the per-family engines.
- It does not change per-family chat *rendering* or any model-level
  Swift code — only which template/handler a family selects.

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
