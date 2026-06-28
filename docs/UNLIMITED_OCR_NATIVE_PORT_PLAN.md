# Unlimited-OCR — Native Krill Port Plan

Status: proposed (plan-first; no engine code until signed off). Date: 2026-06-28.
Owner: Sourav. Target: `krill run unlimited-ocr` serving document/OCR parsing
natively on Apple Silicon, in a fast Mac-native quantized format.

> **Release coupling:** per decision on 2026-06-28, **v0.15.0 is held until this
> port lands.** The ready web-search + fixes work waits behind this. That is a
> deliberate, accepted cost — flagged here so it stays visible.

---

## 1. Outcome we want

- `krill pull unlimited-ocr` + `krill run unlimited-ocr` parses images / multi-page
  PDFs to text natively — no Python, no `trust_remote_code`, no SGLang.
- Runs in a **fast, Mac-native quantized format** (Krill MLX, mixed-precision),
  not the 6.67 GB bf16 source.
- Text decode is native Swift+MLX; the vision encoder runs natively too (it is the
  *point* of an OCR model — see §5 for the native-vs-bridge decision).
- Parity-validated against the reference (HF/transformers) the way the DeepSeek-V2
  and Ornith runtimes were.

## 2. What the model actually is (verified from config + modeling files)

`model_type: unlimited-ocr`, `architectures: ['UnlimitedOCRForCausalLM']`, MIT,
custom-code. Built on **DeepSeek-OCR**. Three parts:

### 2.1 Language backbone — DeepSeek-MoE (small)
From `language_config` (`DeepseekV2Model`):
- `hidden_size 1280`, `num_hidden_layers 12`, `num_attention_heads 10`,
  `num_key_value_heads 10`, `intermediate_size 6848`.
- MoE: `n_routed_experts 64`, `num_experts_per_tok 6`, `n_shared_experts 2`,
  `moe_intermediate_size 896`, `first_k_dense_replace 1`, `n_group 1`.
- **`kv_lora_rank: None` → standard multi-head attention, NOT MLA.** This is the
  pivotal fact (see §4).
- `max_position_embeddings 32768`.

### 2.2 Vision encoder — "DeepEncoder" (`deeplip_b_l`)
A hybrid two-tower ViT (`deepencoder.py`):
- **SAM-ViT-B**: width 768, 12 layers, 12 heads, windowed attention with global
  attn at layers `[2, 5, 8, 11]`, `downsample_channels [512, 1024]`, image 1024.
- **CLIP-L/14-224**: width 1024, 24 layers, 16 heads, patch 14.
- A 16× downsample between the towers; `LayerNormfp32`, `quick_gelu`, learned
  position embeddings with `get_abs_pos` interpolation.

### 2.3 Projector — `MlpProjector`
`input_dim 2048 → n_embed 1280`, `projector_type linear` (config) — maps fused
vision features into the LM embedding space. (The module also supports
downsample/MLP-GELU variants; this checkpoint uses the linear one.)

### 2.4 Multimodal plumbing
- Tiled "long-horizon" parsing: `candidate_resolutions [[1024,1024]]`,
  `tile_tag 2D`, `global_view_pos head` (gundam/base modes).
- Image tokens spliced into the LM input embeddings at vision-token positions.
- Its own processor / conversation template (`processor_config.json`,
  `conversation.py`).

## 3. What Krill already has (large reuse)

- **DeepSeek-MoE runtime** — `KrillCore/DeepSeekModel.swift` + `MoESwitchGLU.swift`:
  parity-validated V2/V2-Lite/V3 with the router gate, **shared experts**,
  **`first_k_dense_replace`**, fine-grained group gating, YaRN. Binds on
  `model_type: deepseek_v2/_v3` / arch containing `deepseek`
  (`ModelLoader.swift:354`, `ModelManifest.swift:238`). The MoE FFN stack we need
  is **already here and tested**.
- **Quantization pipeline** — `CheckpointQuantizer.swift` + `tools/requant_gemma4_nvfp4.py`
  (generic `--protect <substr>` per-module bit overrides). This is the §6 path.
- **Vision-model precedent** — the multimodal load/serve plumbing shape exists
  from the Gemma-4 / Ornith vision work (image-token splice, processor wiring).
- **Registry/catalog** — alias + no-rebuild `krill catalog` registration
  (`ModelManifest.swift`, `AliasMap.swift`).

## 4. Gaps to close (the actual work)

| # | Gap | Size | Notes |
|---|---|---|---|
| G1 | **Non-MLA attention branch for DeepSeek-MoE.** `DeepSeekAttention` hard-assumes MLA (`kvLoraRank: Int`, mandatory `kv_a_proj_with_mqa`). This checkpoint is standard MHA. | **Small–med** | Add a `kv_lora_rank == nil` path (plain q/k/v proj + RoPE) reusing the existing MoE FFN, or compose Qwen/Llama-style attention + DeepSeek MoE. |
| G2 | **DeepEncoder vision tower (SAM-ViT-B + CLIP-L + 16× downsample).** | **Large** | Net-new. Two ViT variants; SAM windowed attention + the conv downsample are the tricky parts. ViT blocks themselves are standard. |
| G3 | **MlpProjector (linear variant).** | **Small** | Single Linear 2048→1280; trivial once G2 emits features. |
| G4 | **OCR preprocessing + tiling.** resolution tiling, 2D tile tags, global-view placement, normalization. | **Med** | Must match the reference pixel pipeline exactly or OCR quality silently degrades. |
| G5 | **Multimodal splice + processor/template** for `model_type: unlimited-ocr` → reuse deepseek language loader for the `language_config` sub-block; wire image tokens. | **Med** | New loader rule + arch detection entry. |
| G6 | **Mac-native fast format (convert + quantize).** | **Med** | §6. |

## 5. Decision: native vision vs. mlx_vlm bridge

Ornith punts its vision tower to `mlx_vlm` and serves only text natively. For an
**OCR** model, vision *is* the product, so the bar is different:

- **Option A — native DeepEncoder port (G2 in full).** Best end state: pure Swift,
  no Python at runtime, fast. Highest effort. **Recommended target.**
- **Option B — mlx_vlm bridge first.** Get end-to-end OCR working fast by running
  the vision tower via mlx_vlm (as Ornith does), native DeepSeek-MoE text decode,
  then replace vision with the native G2 port in a follow-up. Lower risk, faster
  first light, but ships a Python dependency in the interim.

This is a real fork worth your call (see §9).

## 6. Fast Mac-native format (your "convert to a faster format" ask)

Source is **6.67 GB bf16** safetensors (3B; the bulk is the 64 experts). Plan,
reusing the existing pipeline:

1. **Convert** bf16 safetensors → MLX (`mlx_lm.convert` / Krill `CheckpointQuantizer`).
2. **Mixed-precision quantize** (à la `requant_gemma4_nvfp4.py`): protect the
   quality-critical modules at 8-bit, push the bulk to 4-bit/nvfp4:
   - Protect: **vision encoder + projector** (OCR is pixel-detail sensitive),
     attention `o_proj`, router/gate.
   - 4-bit: the **MoE experts** (residency-dominant, tolerant).
3. Land a Krill-format blob (e.g. `srv-sngh/Unlimited-OCR-mixed-nvfp4`), register
   in the catalog. Target footprint ~2–3.5 GB depending on the protect set.
4. Validate quantized OCR quality against the bf16 reference on a fixed page set
   (CER/exact-match), not just perplexity — OCR regressions hide in layout.

Disk is fine (85 GB free vs ~7 GB source + intermediates).

## 7. Phased plan (each phase independently verifiable, parity-gated)

1. **Research + tensor map.** Enumerate checkpoint keys; map every weight to a
   Swift module; pin the exact preprocessing (resize/normalize/tile) from
   `conversation.py`/`processor_config.json`. Deliverable: a key-map doc. *No code.*
2. **Language backbone (G1, G5-partial).** Load the `language_config` MoE via the
   DeepSeek runtime + new non-MLA branch. Gate: **text-only** forward matches the
   reference logits on a fixed prompt (tiny-fixture parity, like DeepSeek-V2-Lite).
3. **Vision tower (G2, G3).** Port DeepEncoder + projector. Gate: vision features
   match the reference to tolerance on a fixed image.
4. **End-to-end multimodal (G4, G5).** Splice + preprocessing + template. Gate:
   full OCR output matches reference on a small page/PDF set (CER).
5. **Quantize to Mac-native format (G6).** Convert + mixed-precision; re-run the
   §6.4 quality gate quantized.
6. **Register + ship.** Catalog entry, `krill pull/run unlimited-ocr`, docs,
   then unblock and cut the release.

## 8. Risks & unknowns

- **R1 SAM windowed attention + downsample** are the least standard pieces; easy to
  get subtly wrong → silent OCR-quality loss. Mitigation: per-stage feature parity
  (Phase 3 gate) before wiring the LM.
- **R2 Preprocessing drift.** OCR is acutely sensitive to resize/normalization/tile
  geometry; a mismatch degrades accuracy without crashing. Mitigation: byte-exact
  reproduction + CER gate.
- **R3 Non-MLA DeepSeek branch** may surface assumptions baked into the MLA-only
  attention/KV-cache path. Mitigation: Phase 2 text-only parity isolates it.
- **R4 Quant sensitivity** of the vision tower. Mitigation: protect vision at 8-bit;
  quality gate, not just size.
- **R5 Effort/schedule.** This is an Ornith-scale port (its own release). Holding
  v0.15.0 behind it means web search + fixes stay unshipped for that whole window.

## 9. Decisions — LOCKED (2026-06-28)

1. **Vision strategy → native DeepEncoder (Option A).** The release must be
   MLX-only, so no mlx_vlm/Python in the shipped binary. mlx_vlm is permitted only
   as a *parity oracle* during development, never shipped.
2. **Release coupling → v0.15.0 held for this port.** v0.15.0 ships web search +
   fixes **and** native Unlimited-OCR together.
3. **Quant target → nvfp4 mixed-precision ONLY.** No int4 interim release. Protect
   SAM/CLIP/projector + attention `o_proj` + router at 8-bit; MoE experts at
   4-bit/nvfp4. CER quality gate against the bf16 reference.

Phase 1 (research + tensor key-map) is **complete** — see
[`UNLIMITED_OCR_TENSOR_MAP.md`](UNLIMITED_OCR_TENSOR_MAP.md).

## 10. Honest bottom line

The language half is mostly **already built** (DeepSeek-MoE runtime), so this is
not a from-scratch LLM port — it is primarily a **vision-encoder port + multimodal
plumbing + a small non-MLA attention branch + a quantization pass**. Still an
Ornith-scale effort concentrated in the DeepEncoder. Worth doing (real new OCR
capability, reputable MIT model); the main cost is the release it blocks.
