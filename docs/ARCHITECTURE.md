# Krill architecture

Krill is a native Swift + MLX inference runtime for Apple Silicon. The `krill`
binary contains the model runtimes, tokenizer, caches, sampler, grammar engine,
HTTP server, coding-agent harness, and terminal UI. Production inference does
not call Python or `mlx-vlm`.

The changing model roster is intentionally not duplicated here. The
authoritative sources are `AliasMap.swift` for the built-in pull shortcuts and
`ModelCapabilities.swift` for family capabilities and support tiers. The
[README model section](../README.md#models) is the user-facing summary.

## Package boundaries

```text
KrillCLI (executable and command wiring)
├── KrillTUI          full-screen chat/agent presentation logic
├── KrillServer       OpenAI, Ollama, and Anthropic HTTP adapters
│   ├── KrillEngine
│   ├── KrillTooling
│   └── KrillHarness
├── KrillEngine       load/swap lifecycle, prefill, decode, batching
│   ├── KrillCore     text, vision, audio, embedding, and reranker models
│   ├── KrillCache    fp16/int8 KV caches and persistent prefix cache
│   ├── KrillTokenizer
│   ├── KrillSampler
│   └── KrillGrammar
├── KrillHarness      coding-agent loop, permissions, tools, research
├── KrillAgent        hardware-aware model recommender/operator
└── KrillRegistry     model catalog, manifests, Hugging Face puller, config

KrillCore
├── KrillKernels      fused Metal/MLX kernels
└── KrillRuntime      Metal availability and runtime validation
```

These boundaries keep model math independent from the CLI and wire protocols.
`KrillHarness` is also independent of MLX, so its agent loop can be exercised
with a mock generator.

## Inference path

```text
messages + attachments + generation options
                 │
                 ▼
       family-aware prompt/template
                 │
                 ├── dedicated native VLM driver when position/grid state
                 │   cannot fit the generic forward interface
                 ▼
   exact or longest-shared-prefix KV lookup
                 │
                 ▼
       suffix-only or chunked prefill
                 │
                 ├── native image/audio encoder when present
                 ▼
       sampling / grammar mask / stop set
                 │
                 ├── standard pipelined decode
                 ├── prompt-lookup or draft-model speculation
                 └── continuous ragged batching for eligible models
                 ▼
      AsyncStream<TokenEvent> + final stats
```

The in-memory prefix-cache tier can restore the longest common token prefix and
prefill only the divergent suffix. Exact hits can also be hydrated from the
persistent disk tier. Model id and a hash of all non-text conditioning are part
of cache identity, preventing an image/audio KV state from being reused for a
different attachment. Both fp16 and int8 KV paths support shared-prefix reuse.

Long prompts use chunked prefill to avoid a quadratic attention allocation.
Gemma 4 sliding-attention layers can use rotating KV storage, while
full-attention layers retain the full history. See
[`INFERENCE_ENGINE.md`](INFERENCE_ENGINE.md) for the detailed flow.

## Native model runtimes

The native causal-LM set includes dense Llama, Qwen, Mistral, Gemma, Phi and
GLM variants; Qwen3.5 hybrid linear/full attention; and multiple switched-MoE
families including Qwen, Mixtral, OLMoE, and DeepSeek. Separate native paths
serve embedding encoders and cross-encoder rerankers.

Multimodal routing is also in-process:

| Family/path | Native media implementation | Routing detail |
|---|---|---|
| Gemma 4 e2b/e4b | SigLIP2 vision + USM Conformer audio | Generic multimodal forward; image and audio can be combined |
| Gemma 4 unified | Raw-patch vision + raw-audio frame projectors | Generic multimodal forward on the unified decoder |
| Qwen2.5-VL | ViT/merger + 3D mRoPE | Dedicated driver carries image grid and decode position |
| Qwen3.5-VL | Native vision + hybrid text decoder + 3D mRoPE | Dedicated driver; SSM state precludes prefix restoration |
| LLaVA-1.5 | CLIP + projector + Llama decoder | Generic 1D-RoPE multimodal decode |
| Llama 3.2 Vision | Tiled vision + cross-attention | Dedicated driver; supports multiple images |
| LocateAnything | MoonViT + connector + Qwen2.5 decoder | Dedicated native-resolution grid driver |
| Unlimited-OCR | SAM/CLIP DeepEncoder + DeepSeek-MoE decoder | Dedicated OCR prompt/splice path |
| Whisper | Native mel frontend, encoder, decoder, tokenizer | Used for local speech-to-text in voice mode |

Capabilities are determined twice: the registry declares what a family can do,
and the loaded checkpoint can remove capabilities when the necessary tower or
sub-configuration is absent. The server gates media before generation using
that effective capability set.

## Serving and agent surfaces

The server defaults to `127.0.0.1:57455`. Its OpenAI Chat/Completions/Responses,
Ollama, and Anthropic Messages adapters translate into the same internal
generation and tool-call path. Model load/unload, embeddings, reranking, health,
status, and metrics endpoints live alongside those generation surfaces; see
[`SERVER_API.md`](SERVER_API.md) for the current endpoint contract.

`krill run` uses the same engine for one-shot generation and the chat TUI.
`krill code` turns on the in-process agent harness. `krill launch <agent>` starts
or reuses the local server and points supported external coding agents at the
appropriate protocol adapter; it does not insert a separate inference proxy.

## Distribution and release metadata

`VERSION` is the repository release source of truth. CI verifies that it agrees
with the Swift `KrillVersion`, the newest `RELEASES.md` and `CHANGELOG.md`
entries, and the versioned fields in `Formula/krill.rb`.

The formula in this repository is a tested release snapshot. The installable,
canonical formula is published in
[`srvsngh99/homebrew-krill`](https://github.com/srvsngh99/homebrew-krill); release
automation must update both copies from the same asset digest.

## External packages

| Package | Purpose |
|---|---|
| mlx-swift | MLX arrays, neural-network modules, and Metal execution |
| swift-transformers + Jinja | Hugging Face tokenizer and chat-template rendering |
| swift-argument-parser | CLI parsing |
| swift-nio | HTTP server |
| swift-crypto | download and content hashing |
| swift-log | structured logs |

Exact dependency constraints live in [`Package.swift`](../Package.swift).
