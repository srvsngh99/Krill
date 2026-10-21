# Neural Engine embeddings (`embed_device`)

Krill can run the EmbeddingGemma 2 **text** embedder on the Apple Neural Engine (ANE) through
Core ML instead of on the GPU through MLX. The ANE is about 3.5x faster per query, uses about
13x less energy per query, takes the load off the GPU (a chat model keeps the GPU to itself) and keeps
the always-on query server small. It is opt-in: the default is the GPU, exactly as before.

```
# once: build the package from the bf16 EmbeddingGemma 2 you already have
krill pull embeddinggemma-2                  # source weights (1.5 GB)
krill pull embeddinggemma-2-mlx-4bit-dyn-text  # MLX fallback (0.5 GB)
make ane-package                             # ~4 min, writes embeddinggemma-2-ane-lin8 (~860 MB)

# then choose the device
echo 'embed_device = "ane"' >> ~/.krill/config.toml    # or KRILL_EMBED_DEVICE=ane, or /config embed_device=ane
krill serve
curl -si localhost:57455/v1/embeddings -d '{"model":"embeddinggemma-2-mlx-4bit-dyn-text","input":"hi","task":"SearchQuery"}' | grep -i x-krill
# x-krill-embed-device: ane
```

## The setting

| | |
|---|---|
| Key | `embed_device` in `config.toml`; `KRILL_EMBED_DEVICE` in the environment; `/config embed_device=ane` in the TUI |
| Values | `gpu` (default), `ane`, `auto` |
| Precedence | request `device` field > `KRILL_EMBED_DEVICE` > `embed_device` in config.toml > `gpu` |
| Takes effect | when `krill serve` starts (no hot reload) |

Krill has no per-model config section, so there is no per-model layer in that order.

- **`gpu`**: the MLX path. Nothing changes.
- **`ane`**: every EmbeddingGemma 2 text request (`embeddinggemma-2`, `embeddinggemma-2-mlx-4bit-dyn-text`,
  `embeddinggemma-2-8bit`, ... any `embeddinggemma-2*` alias) is served by the ANE package. Other embedders
  (BGE, nomic, ...) are never routed.
- **`auto`**: `ane` when this machine has Core ML (Apple Silicon macOS) and the package is installed, otherwise `gpu`, silently.

If `ane` is set but the package is missing (or Core ML is unavailable) the server **starts anyway, prints a
`WARNING:` line in its banner and the log, and serves on the GPU**. It never fails a request because of the setting. The
response header (below) and `/v1/status` always say what is really happening.

A request can pin its own device with `"device": "gpu" | "ane" | "auto"`. Use `"device": "gpu"` for a bulk corpus job:
**corpus embedding should use the full-precision GPU build** (`embeddinggemma-2`, bf16, fp32 compute). The ANE is
for queries and low-latency serving; its vectors match bf16 closely (see Accuracy) but are not bit-identical.
Nothing in Krill routes to the ANE unless you set `ane` or `auto`.

### What you can see

- Response header `x-krill-embed-device: gpu | ane | mixed` on `/v1/embeddings`, `/api/embed` and `/api/embeddings`
  (`mixed` = some inputs of the request went to MLX, see below).
- `GET /v1/status` has an `embed_device` object: `setting`, `effective`, `ane_package`, `core_ml`.
- The `krill serve` banner prints `Embeddings: embed_device = ane -> ane (Neural Engine package: ...)`.
- A debug log line `[embed] <model>: n input(s) served by <device>`.

## What goes where

| Input | `ane` / `auto` |
|---|---|
| Text up to 512 tokens (after `max_length` truncation, special tokens included) | Neural Engine |
| Text over 512 tokens | MLX, using the build you asked for (so a request for `embeddinggemma-2` falls back to bf16) |
| Image, audio, video (alone or mixed with text) | MLX; text items in the same request still run on the ANE |

`task` prefixes, `instruction`, `dimensions` (Matryoshka 768/512/256/128, re-normalised), `max_length`, `keep_alive`
and the NaN guard behave exactly as on the GPU path.

## The package

`embeddinggemma-2-ane-lin8` lives in the model store next to the other models:

```
embeddinggemma-2-ane-lin8/
  krill_ane.json            backend "coreml", variant, buckets, table, fallback_model (default embeddinggemma-2-mlx-4bit-dyn-text)
  s64.mlmodelc  s128  s256  s512     compiled Core ML models, batch 1, int8 per-channel weights (132 MB each)
  embed_tokens.f16          fp16 [262144, 512] token embeddings, memory-mapped (256 MB on disk; only touched rows are resident)
  tokenizer.json  tokenizer_config.json  config_sentence_transformers.json
```

Each bucket is one fixed-shape model; Krill picks the smallest that fits, pads and masks. The host looks up the
token rows in the mapped table and multiplies by `sqrt(512)`; the model does the 24 layers, masked mean pooling, the
512 to 768 projection and L2 normalisation. The RMSNorm is pre-scaled by 1/64 so that no activation overflows fp16.

**Build it** (offline, Python; the runtime is Swift only):

```
make ane-package                 # lin8 (default). ANE_VARIANT=fp16 for the larger fp16 variant
python3 tools/ane/build_ane_embedder.py --help
```

`make ane-package` creates a venv at `~/.krill/ane-venv`, installs the pins in `tools/ane/requirements.txt`
(torch 2.14.1, coremltools 9.0, numpy, safetensors), converts the four buckets from the installed bf16
`embeddinggemma-2` (about 1.8 GB RAM, ~1 minute per bucket), compiles them with `xcrun coremlcompiler`, writes
the package and registers it. `--source`, `--out`, `--fallback` and `--buckets` adjust the paths.
fp16 variant (`embeddinggemma-2-ane-fp16`, ~1 GB) is optional and is used only if lin8 is absent. 6-bit palettised
weights were measured and rejected (visible loss in retrieval).

## Memory and lifetime

- Buckets load lazily: S=64 on the first query, 128/256/512 only when a longer input arrives. Each prediction runs in
  an `autoreleasepool` with reused input and output buffers (a Core ML loop that allocates an `MLMultiArray` per call
  without one leaks 70 to 500 KB per request).
- The ANE engine uses the same keep-alive as every other embedder (`keep_alive` in config, `KRILL_KEEP_ALIVE`, per-request
  `keep_alive`; default 5m). Past the TTL the buckets, the mapped table, the tokenizer and any MLX fallback are released in one
  atomic check-and-unload (`KeepAliveController.evictIfDue`), so a request that starts at that moment keeps its model.
- Core ML compiles a bucket for the ANE the first time it is loaded on a machine (seconds), then reloads it from the
  system cache in under 100 ms.

## Measured (M4 Pro, macOS 26.6, release build, through HTTP)

| | Neural Engine (lin8) | GPU, MLX 4-bit |
|---|---|---|
| Single query, p50 / p95 | 4.0 ms / 4.7 ms (750 queries, median of 3 runs) | 13.6 ms (Core ML spike power run) |
| Energy per query (over idle) | about 6 mJ (S=64) | about 80 mJ |
| Server process, lifetime peak | 156 MB (queries, 200 passages up to 512 tokens, batches of 8) | 765 MB (same text through the GPU path) |
| Server process while loaded | about 63 to 67 MB | 400+ MB |
| Resting, 10 s after the TTL | 47 to 52 MB (4 MB bare process) | back to the bare process |
| `aned` + `ANECompilerService` growth | under 1 MB | n/a |
| 20,000 mixed requests | no growth (slope -0.4 MB per 1,000) | |
| First request after server start | 0.8 s (ANE compile cache warm), 7.4 s (cold: the model was never compiled on this Mac) | |

The first request also pays for the tokenizer and the mapped table; the next is 4 ms. The same figure applies to the first
request after the keep-alive TTL unloaded the model.

### Accuracy

2,000 Kreach passages (including the 500 longest, cut to 512 tokens) and the 750 queries, vectors compared with the
bf16 build: queries cosine min 0.9993 (mean 0.9998), passages min 0.9989, p01 0.9994. Dense retrieval on the Kreach set,
ANE queries against bf16 passages versus MLX 4-bit queries (paired bootstrap, 10,000 resamples): English hit@5 0.442 vs
0.440 and MRR@10 0.357 vs 0.350, Indic hit@5 0.178 vs 0.170 and MRR@10 0.105 vs 0.102; no significant loss in any
of the four. The ANE vectors are closer to bf16 than the 4-bit GPU vectors are. Same request on `gpu` and `ane`
(4-bit alias): cosine 0.9992 to 0.9995.


## Troubleshooting

- `Embeddings: embed_device = ane -> gpu` in the banner: the package is not installed (`krill list` should show
  `embeddinggemma-2-ane-lin8`) or this is not Apple Silicon macOS.
- 500 `needs the MLX model ... for long text and media`: install the fallback, `krill pull embeddinggemma-2-mlx-4bit-dyn-text`
  (or whichever alias you request).
- Core ML refuses a package after a macOS update: rebuild with `make ane-package`.
