# Neural Engine embeddings and rerankers (`embed_device`)

Krill can run EmbeddingGemma 2 and the other embedders and cross-encoder rerankers listed under
[Supported models](#supported-models) on the Apple Neural Engine (ANE) through Core ML instead of on the GPU through MLX.
The ANE is faster per query, uses far less energy, takes the load off the GPU (a chat model keeps the GPU to itself)
and keeps the always-on query server small. It is opt-in: the default is the GPU, exactly as before.

```
# once per model: build the package from weights you already pulled
krill pull bge-base-en                       # source weights
make ane-package MODEL=bge-base-en ANE_VARIANT=auto   # converts, parity-gates against the GPU, installs bge-base-en-ane-<variant>
# EmbeddingGemma 2: krill pull embeddinggemma-2 embeddinggemma-2-mlx-4bit-dyn-text; make ane-package

# then choose the device (one setting for embedders and rerankers)
echo 'embed_device = "ane"' >> ~/.krill/config.toml    # or KRILL_EMBED_DEVICE=ane, or /config embed_device=ane
krill serve
curl -si localhost:57455/v1/embeddings -d '{"model":"bge-base-en","input":"hi"}' | grep -i x-krill
# x-krill-embed-device: ane
```

## The setting

| | |
|---|---|
| Key | `embed_device` in `config.toml`; `KRILL_EMBED_DEVICE` in the environment; `/config embed_device=ane` in the TUI |
| Values | `gpu` (default), `ane`, `auto` |
| Precedence | request `device` field > `KRILL_EMBED_DEVICE` > `embed_device` in config.toml > `gpu` |
| Takes effect | when `krill serve` starts (no hot reload) |
| Covers | `/v1/embeddings`, `/api/embed`, `/api/embeddings` **and** `/v1/rerank` (there is no separate `rerank_device`; the request `device` field works on both) |

Krill has no per-model config section, so there is no per-model layer in that order.

- **`gpu`**: the MLX path. Nothing changes.
- **`ane`**: a request for `<model>` is served by the installed package `<model>-ane-lin8` (or `-fp16` if that is all
  there is). A model with **no package** is served on the GPU and the server logs **one warning per model** (never a failed
  request). Every quantized EmbeddingGemma 2 text alias (`embeddinggemma-2-mlx-4bit-dyn-text`, `embeddinggemma-2-8bit`, ...)
  shares the one `embeddinggemma-2-ane-*` package. The full-precision `embeddinggemma-2` build is the corpus/reference
  embedder and **stays on the GPU whatever the setting**; only a request that sends `"device": "ane"` (or `"auto"`) moves it.
- **`auto`**: `ane` for a model whose package is installed on a machine with Core ML (Apple Silicon macOS), otherwise `gpu`, silently.

A request can pin its own device with `"device": "gpu" | "ane" | "auto"`. Use `"device": "gpu"` for a bulk corpus job:
**corpus embedding should use the full-precision GPU build** (`embeddinggemma-2`, bf16, fp32 compute); ANE vectors match the
GPU closely (see Accuracy and the table) but are not bit-identical. Nothing in Krill routes to the ANE unless you set `ane` or `auto`.

### What you can see

- Response header `x-krill-embed-device: gpu | ane | mixed` on `/v1/embeddings`, `/api/embed`, `/api/embeddings` and
  `/v1/rerank` (`mixed` = some inputs of the request went to MLX, see below).
- `GET /v1/status` has an `embed_device` object: `setting`, `effective`, `ane_package` (first), `ane_packages`
  (`[{model, package}]`), `core_ml`.
- The `krill serve` banner prints `Embeddings and rerank: embed_device = ane -> ane (Neural Engine packages: ...)`.
- A debug log line `[embed] <model>: n input(s) served by <device>` and, once per model, `[embed] embed_device = ane but no ANE
  package is installed for <model> (...)`.

## What goes where

| Input | `ane` / `auto` |
|---|---|
| Text up to the package's largest bucket (512 tokens; after `max_length` truncation, special tokens included) | Neural Engine |
| Longer text | MLX, using the GPU build of the same model you asked for (a request for `embeddinggemma-2` falls back to bf16) |
| Image, audio, video (EmbeddingGemma 2 only; alone or mixed with text) | MLX; text items in the same request still run on the ANE |
| Rerank pair (query + document) | Neural Engine; the pair is cut to the largest bucket (documents are truncated, never the query) |

`task` prefixes, `instruction`, `dimensions` (Matryoshka models only; re-normalised), `max_length`, `keep_alive`, the NaN
guard, tokenisation (EmbeddingGemma 2's code-point BPE; Krill's own tokenizer for every other model), the sentence-transformers
`prompts` table and, for decoder embedders, the single trailing EOS behave exactly as on the GPU path. Pooling follows
Krill's own per-model rule on both devices (plain BERT/RoBERTa models are mean pooled even where the upstream card says CLS).

## The package

A package is a model-store directory named `<model>-ane-<variant>`, variant `lin8` (int8 per-channel weights, default
when it passes the parity gate) or `fp16`:

```
bge-base-en-ane-fp16/
  krill_ane.json            manifest (format 2, below)
  s64.mlmodelc  s128  s256  s512     compiled Core ML models, batch 1 (buckets are 64/128/256/512, capped at the model's context)
  embed_tokens.f16          fp16 [vocab, hidden] word-embedding table, memory-mapped (only the rows in use become resident)
  tokenizer.json ... config.json config_sentence_transformers.json 1_Pooling/   copied from the source model
```

Each bucket is one fixed-shape model; Krill picks the smallest that fits, pads and masks. The host looks up the
token rows in the mapped table (times `embed_scale`) and the model does everything else: position/type embeddings,
the layers, pooling (mean, CLS, last token, or the cross-encoder classifier head) and L2 normalisation. RMSNorm and
LayerNorm are pre-scaled so that no activation can overflow fp16.

### Manifest format 2 (`krill_ane.json`)

Format 1 (EmbeddingGemma 2 only, PR 343) is still read; its missing fields default to the EmbeddingGemma 2 values.

| Field | Meaning |
|---|---|
| `format` | `2` (`1` accepted) |
| `kind` | `embedder` or `reranker` |
| `model` | registry name the package was built from |
| `pooling` | `mean`, `cls`, `last_token`, `classifier`; informational, baked into the graph |
| `gather` | `host` (mapped fp16 table); `model` is reserved and refused |
| `tokenizer` | `codepoint_bpe` (EG2) or `krill` |
| `output_name`, `output_dim`, `embed_scale` | Core ML output, its width (1 for a reranker), host multiplier on gathered rows |
| `max_tokens`, `buckets` | largest bucket and the batch-1 bucket list |
| `context_tokens` | the model's own context (what the GPU path truncates to) |
| `append_eos` | decoder embedders: exactly one trailing EOS |
| `mrl_dims` | matryoshka widths or null |
| `table` | `{file, vocab, dim, dtype}` |
| `fallback_model` | the GPU build of the same model, for text past the buckets |

Prefixes and task prompts are not duplicated in the manifest: the package carries the model's own sentence-transformers
config and `EmbeddingPromptTable` reads it like it does for the GPU build.

## Build a package

Offline, Python; the runtime is Swift only.

```
krill pull bge-base-en
make release                                   # the parity gate serves the package with this binary
make ane-package MODEL=bge-base-en ANE_VARIANT=auto ANE_CORPUS=/path/corpus.json
python3 tools/ane/build_ane_package.py --help
```

`make ane-package` creates a venv at `~/.krill/ane-venv` (pins in `tools/ane/requirements.txt`: torch 2.14.1,
coremltools 9.0, numpy, safetensors), reads the weights from the model store (`KRILL_MODELS_DIR` or `~/.krill/models`),
converts the buckets (1 to 15 minutes), compiles them with `xcrun coremlcompiler` and **parity-gates the package before it
installs it**: `krill serve` loads the new package next to the source model and the same server answers each request on `ane` and
on `gpu`. Embedders need cosine >= 0.999 as the **minimum** over 200 queries (one per request) and 200 passages (batches of 8);
rerankers need Spearman >= 0.999 over 1,000 scored pairs (the max absolute logit difference is printed too).
`--variant auto` tries `lin8` and falls back to `fp16` when int8 misses the gate. A model that fails the gate writes nothing
and prints the numbers. `--corpus` is a JSON file `{"queries": [...], "passages": [...]}`; `--reference hf` compares with HF
transformers fp32 instead of the GPU build (for a checkpoint the GPU path cannot load).

The encoders in `tools/ane/encoders.py` are torch ports of the Krill GPU modules (`Sources/KrillCore/*EmbeddingModel.swift`)
and use the HF checkpoint key names: BERT/RoBERTa/XLM-R (+ cross-encoder head), MPNet, nomic-bert (v1/v1.5 and the v2 MoE,
all eight experts computed densely with the top-2 weights applied, as the GPU path does), gte-v1.5, jina-bert (ALiBi), ModernBERT and
Qwen2. They are hand-written rather than traced from `transformers` because nomic, gte and jina are custom-code architectures
that `transformers` cannot load offline, and HF masks/SDPA do not trace to a fixed-shape fp16 graph. Each was checked against
HF where HF has the architecture (cosine 1.0000 on BERT, MPNet, ModernBERT; 3e-6 on the cross-encoder logit).
EmbeddingGemma 2 keeps its own port (`eg2_ane.py`) and the format-1 package.

## Supported models

Measured on an M4 Pro, macOS 26.6, release build, through HTTP on the shared dev Mac (see "How these were measured").
"p50 / p95" are single-query milliseconds, GPU then ANE; "peak" is the lifetime peak `phys_footprint` of the server
process (150 single queries, 200 passages of up to 512 tokens in batches of 8), GPU then ANE; "rest" is the process 14 s
after an 8 s keep-alive expired (a bare process is about 4 MB). A model ships as supported only if parity passes, ANE p50 beats GPU p50
and the ANE peak is lower. Every package has 100% of its compute ops placed on the ANE by `MLComputePlan`
(all four buckets, `CPU_AND_NE`). Sizes are the whole package (four buckets plus the word-embedding table).

| Model | Variant, size | Parity (ANE vs GPU build through Krill) | p50 ms | p95 ms | Peak MB | Rest MB | Verdict |
|---|---|---|---|---|---|---|---|
| `all-minilm-l6-v2` | fp16, 110 MB | cosine min 0.99997 / mean 0.99998 / p01 0.99997 (queries); min 0.99998 (passages) | 3.0 / 1.1 | 6.4 / 1.4 | 383 / 28 | 26 / 14 | supported |
| `bge-small-en` | fp16, 196 MB | min 0.99999 (q), 0.99998 (p) | 5.0 / 1.8 | 9.1 / 2.3 | 556 / 30 | 27 / 18 | supported |
| `bge-base-en` | fp16, 730 MB | min 0.99999 (q), 0.99999 (p) | 8.9 / 2.6 | 14.7 / 3.4 | 1044 / 34 | 30 / 22 | supported |
| `all-mpnet-base-v2` | fp16, 738 MB | min 0.99998 (q), 0.99991 (p) | 10.5 / 2.7 | 16.9 / 3.4 | 999 / 50 | 29 / 48 | supported |
| `gte-base-en-v1.5` | fp16, 955 MB | min 0.99997 (q), 0.99995 (p) | 11.3 / 3.2 | 18.3 / 3.9 | 1076 / 34 | 25 / 32 | supported |
| `jina-embeddings-v2-base-en` | fp16, 963 MB | min 0.99990 (q), 0.99991 (p) | 9.9 / 3.0 | 15.8 / 3.5 | 1188 / 48 | 27 / 26 | supported |
| `nomic-embed-text` (v1.5) | fp16, 955 MB | min 0.99962 (q), 0.99939 (p) | 7.6 / 3.2 | 13.5 / 4.1 | 1004 / 34 | 24 / 27 | supported |
| `nomic-embed-text-v1` | fp16, 955 MB | min 0.99959 (q), 0.99942 (p) | 6.4 / 3.5 | 13.0 / 6.3 | 1030 / 35 | 25 / 27 | supported |
| `nomic-embed-text-v2-moe` | fp16, 2,669 MB | min 0.99954 (q), 0.99969 (p) | 26.6 / 6.1 | 29.8 / 6.8 | 2873 / 226 | 75 / 98 | supported (all 8 experts computed densely, top-2 weights) |
| `bge-reranker-base` | lin8, 746 MB | Spearman 0.99962 on 730 scored pairs (0.99195 on all 1,000 including the floor plateau); max abs logit diff 0.452 | 135.6 / 80.8 | 156.8 / 84.6 | 1794 / 217 | 72 / 24 | supported |
| `gte-large-en-v1.5` | fp16, 3,288 MB | min 0.99997 (q), 0.99937 (p) | 17.0 / 23.8 | 18.7 / 24.4 | 2800 / 42 | 30 / 33 | **GPU-only**: parity passes and the peak is 66x lower, but ANE p50 is slower than the GPU's |
| `gte-modernbert-base` | none | fp16 cosine min 0.99923 (q) but 0.99731 (p) against HF fp32; lin8 0.36 | | | | | **GPU-only**: fails the parity gate. The error is fp16 accumulation on long inputs, not overflow (pre-scales 1, 1/8 and 1/32 give the same 0.9988 vs fp32 at 512 tokens). Krill's own GPU path also cannot load the pulled checkpoint in this release (`keyNotFound final_norm.weight`; its keys carry a `model.` prefix), so the gate used `--reference hf` |
| `gte-qwen2-1.5b` | not built | | | | | | **GPU-only for now**: not attempted. The encoder is ported (`Qwen2` in `encoders.py`, last-token pooling, one-hot select) but the 6 GB fp32 download stages in the system temp on the internal disk, which was under the 15 GB free-space limit set for this work |
| `gte-qwen2-7b`, `e5-mistral-7b-instruct`, `sfr-embedding-mistral` | none | | | | | | **GPU-only**: 7B decoders are too large for the ANE |
| `bge-reranker-large`, `bge-reranker-v2-m3` | none | | | | | | **GPU-only for now**: not attempted (560M and 568M parameters; the same builder handles them) |

`lin8` (int8 per-channel) was tried first for every model with `--variant auto`. It missed the 0.999 minimum-cosine gate for every
embedder it was gated on (gte-large was built fp16 only; for example bge-base-en passages min 0.9982, gte-base 0.9988, MiniLM 0.9964, jina 0.9495) and passed only for the
reranker, so embedders ship as `fp16`. Everything above is a model alias of its own: `all-minilm` and `nomic-embed-text-v1.5`
share weights with `all-minilm-l6-v2` and `nomic-embed-text` but need their own package (`make ane-package MODEL=all-minilm`).

### How these were measured

One server process per device (`KRILL_EMBED_DEVICE=ane` or unset, keep-alive 8 s, port 57492), 12 warm-up requests, then the
median of three runs of 150 single-query requests; `bench.lock` was held only while measuring. The Mac is shared, so each run logs
free memory, the Swapouts delta, the load average and any foreign process above 50% CPU or 1 GB. Swapouts never rose (0 MB in
every run) and load stayed under 7, but a foreign process over 1 GB (the browser, the two always-on `krill` services, a Python job) was present
in every run, which the rule counts as contended; each measurement was therefore repeated six times and the median taken from the
first three. The ANE advantage is 1.7x to 4.4x, far outside that noise; the one verdict near the line (gte-large, ANE 1.4x slower)
was stable across all six runs.

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
- 500 `needs the MLX model ... for long text and media`: install the fallback, the GPU build of the same model
  (`krill pull <model>`; for EmbeddingGemma 2 `krill pull embeddinggemma-2-mlx-4bit-dyn-text`).
- `embed_device = ane` but a model is served on the GPU: the log has one `no ANE package is installed for <model>` line;
  build it with `make ane-package MODEL=<model>`. `x-krill-embed-device` always says what served the request.
- `make ane-package` ends with `PARITY FAILED`: the numbers are printed and nothing was installed. Try `--variant fp16`
  (or `auto`); a model that still fails stays on the GPU.
- Core ML refuses a package after a macOS update: rebuild with `make ane-package`.
