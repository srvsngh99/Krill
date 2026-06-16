# Running the Gemma-4-12B coder fine-tune natively in MLX

`yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1` is a coding/reasoning
fine-tune of `google/gemma-4-12B-it`. It is the **same `gemma4_unified`
architecture KrillLM already serves natively** (`family: .gemma4Unified`,
`Gemma4UnifiedModel`) - only the weights differ. So no engine changes are
needed; the entire task is getting the weights into the MLX layout the Swift
loader expects.

## Why not GGUF

The fine-tune is most visible as `*-GGUF` (llama.cpp k-quants: Q2_K … Q8_0).
We deliberately **do not** ingest the GGUF:

- GGUF k-quants are lossy. Even Q8_0 cannot recover the bf16 the fine-tuner
  trained from, so converting from GGUF would bake that loss into the MLX
  weights - the opposite of "best performance".
- KrillLM has no GGUF runtime, and `gemma4` GGUF is not something MLX loads.

Instead we start from the **NVFP4 safetensors** publication of the same model,
e.g. `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4`. That is a
clean 4-bit-float (`nvfp4-pack-quantized`, compressed-tensors) checkpoint of the
identical architecture - no GGUF anywhere. ("MTP" in the name is branding; the
shard contains no multi-token-prediction tensors, just the standard
`gemma4_unified` weights.)

## The pipeline (two offline steps, pure numpy + MLX - no torch, no GGUF)

```
NVFP4 safetensors (compressed-tensors)
        │  tools/convert_gemma4_compressed_nvfp4_to_bf16.py
        ▼
bf16 safetensors (MLX key scheme)
        │  tools/requant_gemma4_nvfp4.py  --protect o_proj
        ▼
MLX nvfp4 (uniform 4-bit + 8-bit-protected attn o_proj + 8-bit vision/audio projectors)
        │  krillm run <dir>
        ▼
served natively
```

### Step 0 - download the NVFP4 source

```sh
huggingface-cli download sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4 \
    --local-dir ~/models/coder-nvfp4-src
```

### Step 1 - decompress NVFP4 → bf16

`tools/convert_gemma4_compressed_nvfp4_to_bf16.py` reads the compressed-tensors
shard at the byte level (the F8_E4M3 / U8 / BF16 dtypes have no numpy
equivalent), reconstructs each quantized Linear

```
value  = E2M1[code & 0x7] * (-1 if code & 0x8 else +1)   # FP4 E2M1
weight = value * weight_scale / weight_global_scale       # NVFP4 two-level scale
```

(FP4 LUT, nibble order and the two-level formula are taken verbatim from
`vllm-project/compressed-tensors`), and rewrites the HF keys into the
MLX/KrillLM scheme used by the requant oracle:

| compressed-tensors (HF) | MLX / KrillLM |
| --- | --- |
| `model.language_model.<x>` | `language_model.model.<x>` |
| `model.embed_vision.<x>` | `embed_vision.<x>` |
| `model.embed_audio.<x>` | `embed_audio.<x>` |
| `model.vision_embedder.<x>` | `vision_embedder.<x>` |

```sh
# validate the decoder first (no full run): re-encodes one module, checks the
# bytes match the stored weight_packed, and that the global-amax block scale == 448
python3 tools/convert_gemma4_compressed_nvfp4_to_bf16.py \
    --src ~/models/coder-nvfp4-src --out /tmp/x --self-check

python3 tools/convert_gemma4_compressed_nvfp4_to_bf16.py \
    --src ~/models/coder-nvfp4-src --out ~/models/gemma-4-12b-coder-bf16
```

### Step 2 - requantize bf16 → MLX nvfp4 (the proven recipe)

`tools/requant_gemma4_nvfp4.py` is the same tool that produced the canonical
`gemma-4-12b-nvfp4` (uniform nvfp4 + 8-bit `o_proj` + 8-bit vision/audio
projectors = the both-axes win). The new `--src-bf16-dir` flag points it at a
converted fine-tune instead of the cached canonical bf16:

```sh
python3 tools/requant_gemma4_nvfp4.py \
    --src-bf16-dir ~/models/gemma-4-12b-coder-bf16 \
    --out          ~/models/gemma-4-12b-coder-nvfp4 \
    --protect      o_proj
```

The module set to quantize is learned from the cached
`mlx-community/gemma-4-12B-it-4bit` oracle (same architecture), so the coverage
matches the canonical 12B exactly. Output config carries the top-level `nvfp4`
block + per-module 8-bit overrides - the exact format the Swift loader resolves
via `q.effective(path)`.

The fine-tune's own `chat_template.jinja` + tokenizer are carried through both
steps, so its `<|turn>` / `<|channel>thought` prompt format and reasoning
channels are preserved (KrillLM already strips Gemma channels via
`ReasoningParser.stripGemmaChannels`).

## Serving

```sh
krillm run ~/models/gemma-4-12b-coder-nvfp4          # by path - no registration
krillm serve --model ~/models/gemma-4-12b-coder-nvfp4
```

To run it under a friendly name (`krillm run gemma-4-12b-coder`) you need it in
the local registry, which today means publishing the converted dir to a HF repo
and `krillm pull`-ing it (then optionally adding an `AliasMap` entry so it is a
known built-in). `krillm create`'s `FROM` only accepts an already-installed
registry model, not a local path - so publish-then-pull is the supported route
to a named built-in. That publish step is an explicit, public action; do it
deliberately.

## Verifying

After conversion, confirm coherence with a coding prompt and check the reasoning
channel is stripped from the visible output. A garbled response would indicate a
decode/remap bug (the converter's `--self-check` guards against this up front).
```
