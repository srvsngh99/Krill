# gemma-4-12b-agentic — Gemma-4-12B agentic fine-tune (Krill NVFP4)

`gemma-4-12b-agentic` is the community **agentic** fine-tune of `google/gemma-4-12B-it`
(`yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2`), optimized for
tool-use / τ²-bench, served natively in MLX on Krill's `gemma4_unified` runtime.
Sibling of the coder fine-tune (see `GEMMA4_12B_CODER_FINETUNE.md`).

```bash
krill pull gemma-4-12b-agentic
krill run  gemma-4-12b-agentic "Write a Python LRU cache."
KRILL_ENABLE_THINKING=1 krill run gemma-4-12b-agentic "..."   # reasoning channel
```

Registry alias → `srv-sngh/gemma-4-12B-agentic-fable5-composer2.5-v2-nvfp4`
(`Sources/KrillRegistry/AliasMap.swift`).

## How it was converted

Unlike the coder (which shipped as compressed-tensors NVFP4), the agentic v2 was
only published as GGUF upstream. We converted from a community **bf16 safetensors**
re-host — *not* from GGUF, to avoid k-quant quality loss:

1. **Download** the bf16 safetensors source (HF key layout, single `model.safetensors`).
2. **Reshard + key-remap** to MLX/oracle layout with an index
   (`model.<x>` → Krill's `language_model.model.<x>` / `embed_vision.<x>` scheme).
3. **Requant** with the proven recipe via `tools/requant_gemma4_nvfp4.py`:
   bulk **NVFP4** (group_size 16, 4-bit) + **8-bit affine** protected `o_proj` and
   vision/audio projectors. → 6.8 GB, loads in ~1.7 s.

## Compatibility

MLX checkpoint; requires the `gemma4_unified` architecture + the mixed-NVFP4 config
(top-level nvfp4 + per-module 8-bit overrides). Runs on **Krill**; not drop-in for
vanilla `mlx_lm`/`mlx_vlm`, llama.cpp/Ollama (GGUF), or transformers/vLLM.

## Benchmarks

HumanEval/MBPP/GSM8K (incl. EvalPlus) comparison vs the base and coder models lives
on the Hugging Face model card:
<https://huggingface.co/srv-sngh/gemma-4-12B-agentic-fable5-composer2.5-v2-nvfp4>.
