# Inference engine internals

`Sources/KrillEngine/InferenceEngine.swift` owns the loaded model, tokenizer,
cache policy, grammar masks, and streaming generation lifecycle. Model-family
math is constructed by `KrillCore.loadModel`, which returns a `LoadedModel`
containing the common forward closures and optional family-specific seams.

## Model lifecycle

```text
load        read config → detect architecture → instantiate → load weights/tokenizer
swap        load replacement into temporaries → publish it atomically
unload      release model, tokenizer, and model-specific state
load draft  construct a SpeculativeDecoder without changing the target model
```

A failed swap leaves the prior model active. Effective capabilities combine the
registry family declaration with checkpoint facts discovered by the loader; a
text-only checkpoint therefore cannot accidentally advertise image or audio.

## Generation dispatch

The public `generate` entry points produce an `AsyncStream<TokenEvent>` and a
thread-safe accessor for the completed `GenerationStats`.

Most text and 1D-position multimodal models use the generic pipeline. Dedicated
native drivers handle state that does not fit its common forward signature:

- Qwen2.5-VL and Qwen3.5-VL carry image-grid-aware 3D mRoPE positions.
- Llama 3.2 Vision carries cross-attention KV state and a sparse per-image mask.
- LocateAnything carries the native-resolution MoonViT grid.
- Specialized OCR and Gemma 4 media paths build their required placeholder or
  splice layout before prefill.

No driver shells out to Python. Gemma 4 audio is decoded and preprocessed once
before prefill: e2b/e4b use the native USM log-mel path, while the unified model
projects decoded waveform frames directly. Native images are preprocessed by
their corresponding vision family.

## Generic request pipeline

1. Resolve the family chat template, thinking mode, stop-token set, media
   placeholders, context limit, and requested output grammar.
2. Allocate the family-appropriate fp16 or int8 KV caches. Gemma sliding layers
   may use rotating caches; KV-sharing layers preserve their donor topology.
3. Look for an exact prefix-cache entry. An exact hit restores KV, backs up one
   token, and forwards that token to recover logits without duplicating a row.
4. On an exact miss, look in memory for the longest common token prefix for the
   same model and media hash. Restore and truncate the donor cache, then prefill
   only the divergent suffix. Unsafe spans or incompatible cache geometry fall
   back to a cold prefill.
5. Prefill the remaining prompt. Long prompts are split according to the engine
   setting exposed as `KRILL_PREFILL_CHUNK`; last-token-only forward closures
   avoid an unnecessary vocabulary projection for earlier prompt positions.
6. Snapshot eligible KV state into the bounded prefix cache. Writes to the disk
   tier are asynchronous.
7. Sample and stream until a stop id, stop string, cancellation, or token limit.
   Grammar-constrained requests mask logits before sampling.
8. Emit the terminal event and publish prompt/decode timing, token counts, TTFT,
   cache/speculation data, and derived throughput.

## KV and prefix caches

The ordinary KV layout is `[batch, kvHeads, sequence, headDim]`. `KVCache`
amortizes decode-time concatenation by collecting pending slices and compacting
them in batches. `QuantizedKVCache` stores int8 values with scale/zero metadata.
`RotatingKVCache` bounds storage for sliding-window attention.

`PrefixCache` has two bounded tiers:

- An in-memory LRU (eight entries by default) supports both exact matches and
  longest-common-prefix reuse because it retains each entry's tokens.
- A persistent safetensors tier under `~/.krill/cache` supports exact matches
  across processes. Hydrated disk entries intentionally do not participate in
  longest-common-prefix scanning.

The key covers model id, KV dtype, prompt tokens, and non-text media identity.
The disk budget defaults to 2 GB and the per-entry memory cap defaults to 4 GB;
both are configurable. Prefix reuse is skipped when a family has non-restorable
state (for example Qwen3.5's SSM path) or when cache-span guards reject it.

## Decode strategies

The default single-request path overlaps token detokenization/stream delivery
with the next forward where possible. Greedy generation can additionally use:

- **Prompt-lookup (n-gram) speculation:** propose repeated runs from the prompt
  or generated context, verify them with the target, and hand back to ordinary
  decode when a rolling acceptance monitor says lookup is no longer useful.
- **Draft-model speculation:** a separately loaded small model proposes a run;
  the target verifies it in one forward, rolls caches back at the first
  rejection, and emits a bonus token when every proposal is accepted.

Non-greedy sampling, penalties, int8 restrictions, grammar requirements, and
explicit request options determine whether speculation is eligible.

For concurrent server traffic, `ContinuousBatcher` groups ready rows into
ragged epochs. Per-row masks and RoPE offsets isolate left-padded sequences.
Eligible model families expose batched fp16 and, where implemented, int8 or
windowed decode closures; other families fall back to serialized generation.

## Sampling and structured output

`KrillSampler` applies penalties and temperature/top-k/top-p filtering before
categorical sampling; non-positive temperature is greedy argmax.
`KrillGrammar` compiles JSON, JSON Schema, regular-expression, CFG, and forced
tool-call constraints into token masks. The engine memoizes decoded vocabulary
pieces and the most recent compiled masks for the loaded model.

## Correctness gates

Most tests use deterministic synthetic fixtures. Tests that need real weights
are opt-in through environment paths and run on Apple Silicon. The maintained
matrix and commands are documented in [`TESTING.md`](TESTING.md); a scheduled
workflow exercises a small real text checkpoint, while larger multimodal and
performance gates remain explicit because of download size and runtime.
