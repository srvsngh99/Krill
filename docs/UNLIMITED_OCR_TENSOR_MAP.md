# Unlimited-OCR — Tensor Key Map & Spec (Phase 1 deliverable)

Status: research complete. Date: 2026-06-28. Companion to
[`UNLIMITED_OCR_NATIVE_PORT_PLAN.md`](UNLIMITED_OCR_NATIVE_PORT_PLAN.md).

End goal (locked): ship a **native MLX nvfp4** Unlimited-OCR in **v0.15.0** — no
Python / no `trust_remote_code` / no mlx_vlm in the release. mlx_vlm is used only
as a parity oracle during development.

Source: `baidu/Unlimited-OCR`, 1 safetensors shard, **2710 tensors**, 6.67 GB bf16.
61 unique key patterns (layer index collapsed to `N`). Every pattern is mapped
below to its target Swift module, with reuse vs net-new called out.

---

## 1. Top-level layout

```
lm_head.weight                       → LM head
model.embed_tokens.weight            → token embeddings
model.norm.weight                    → final RMSNorm
model.layers.N.*                     → DeepSeek-MoE decoder (12 layers)   [LANGUAGE]
model.sam_model.*                    → SAM-ViT-B tower                    [VISION]
model.vision_model.*                 → CLIP-L/14 tower                    [VISION]
model.projector.layers.{weight,bias} → MlpProjector (linear 2048→1280)    [VISION→LM]
model.image_newline                  → learned tile row-separator token
model.view_seperator                 → learned global/local view separator
```

## 2. Language decoder — `model.layers.N.*` (REUSE Krill DeepSeek-MoE)

| Key pattern | Module | Status |
|---|---|---|
| `self_attn.q_proj.weight` | Q proj | **G1: non-MLA branch** |
| `self_attn.k_proj.weight` | K proj | G1 |
| `self_attn.v_proj.weight` | V proj | G1 |
| `self_attn.o_proj.weight` | O proj | reuse |
| `input_layernorm.weight` / `post_attention_layernorm.weight` | RMSNorm | reuse |
| `mlp.gate.weight` | MoE router | **reuse** (`DeepSeekMoEGate`) |
| `mlp.experts.N.{gate,up,down}_proj.weight` | 64 routed experts | **reuse** (`MoESwitchGLU`) |
| `mlp.shared_experts.{gate,up,down}_proj.weight` | 2 shared experts | **reuse** |
| `mlp.{gate,up,down}_proj.weight` (no `experts`) | dense MLP for the first `first_k_dense_replace=1` layer | **reuse** |
| `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight` | embed / final norm / head | reuse |

**Key finding:** attention is plain `q/k/v/o_proj` (standard MHA) — **no**
`q_a_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`, confirming `kv_lora_rank: None`. The
entire MoE stack (router, 64 routed + 2 shared experts, dense-prefix layer)
maps 1:1 onto Krill's existing parity-validated DeepSeek-MoE modules. **The only
language gap is G1: a non-MLA attention path** (Krill's `DeepSeekAttention` hard-
assumes MLA). Dims: hidden 1280, 12 layers, 10 heads, 6 experts/token.

## 3. SAM-ViT-B tower — `model.sam_model.*` (NET-NEW, G2a)

| Key pattern | Module |
|---|---|
| `patch_embed.proj.{weight,bias}` | Conv2d patch embed (1024 img) |
| `pos_embed` | absolute pos embed (interp via `get_abs_pos`) |
| `blocks.N.norm1/norm2.{weight,bias}` | LayerNorm |
| `blocks.N.attn.qkv.{weight,bias}` / `attn.proj.{weight,bias}` | fused-QKV attention |
| `blocks.N.attn.rel_pos_h` / `rel_pos_w` | **decomposed relative position** (ViTDet windowed attn) |
| `blocks.N.mlp.lin1/lin2.{weight,bias}` | MLP |
| `neck.N.{weight,bias}` | channel-reduction neck (`downsample_channels [512,1024]`) |
| `net_2.weight`, `net_3.weight` | downsample convs (16× reduction) |

Net-new. 12 blocks, width 768, global attn at layers `[2,5,8,11]` (others
windowed). The **decomposed rel-pos + windowing** and the **neck/downsample convs**
are the trickiest pieces and the top parity risk (R1).

## 4. CLIP-L/14 tower — `model.vision_model.*` (NET-NEW, G2b — mostly standard ViT)

| Key pattern | Module |
|---|---|
| `embeddings.class_embedding` / `patch_embedding.weight` / `position_embedding.weight` | CLIP embeddings (patch 14, 224) |
| `pre_layrnorm.{weight,bias}` | pre-LN (note HF's `layrnorm` spelling) |
| `transformer.layers.N.layer_norm1/2.{weight,bias}` | LayerNorm |
| `transformer.layers.N.self_attn.qkv_proj.{weight,bias}` / `out_proj.{weight,bias}` | fused-QKV attention |
| `transformer.layers.N.mlp.fc1/fc2.{weight,bias}` | MLP (`quick_gelu`) |

Net-new but standard ViT-L (24 layers, width 1024, 16 heads). `LayerNormfp32` +
`quick_gelu` activation. Lower risk than SAM.

## 5. Projector & glue (NET-NEW, G3 — trivial)

- `model.projector.layers.{weight,bias}` — single `Linear(2048 → 1280)`. Input
  2048 = concat of the two 1024-d vision streams (SAM neck + CLIP) → LM hidden 1280.
- `model.image_newline`, `model.view_seperator` — learned vectors inserted during
  tile layout (2D tile tags / global-view-at-head). Wire in the splice step (G5).

## 6. Preprocessing (pin EXACTLY — `processor_config.json`) (G4)

- `image_mean = image_std = [0.5,0.5,0.5]`, `normalize: true` → scale to [-1, 1].
- `patch_size: 16`, `downsample_ratio: 4`, `candidate_resolutions: [[1024,1024]]`.
- `image_token: "<image>"`, `pad_token: "<｜▁pad▁｜>"`, `add_special_token: false`.
- Tiling: `tile_tag: 2D`, `global_view_pos: head` (gundam = tiled + global view at
  head; base = single view). `image_newline` between tile rows; `view_seperator`
  between global and local views.
- **R2:** OCR accuracy is acutely sensitive to resize/normalize/tile geometry —
  reproduce byte-exact and gate on CER, not perplexity.

## 7. Chat template (`conversation.py`) (G5)

DeepSeek / DeepSeekV2 separator style (FastChat-derived). Roles USER/ASSISTANT.
Special tokens: `<｜sft▁begin｜>`, `<｜sft▁end｜>`, `<｜end▁of▁sentence｜>` (EOS).
`sft_format: "unlimitedocr"`. Implement as a Krill prompt builder (no Jinja
dependency), mirroring how other families are handled in `applyChatTemplate`.

## 8. Net-new vs reuse — summary

- **Reuse (no/low work):** entire MoE FFN stack + router + shared experts + dense
  prefix + embeddings/head/norm (Krill DeepSeek-MoE), quantization pipeline,
  registry/catalog, multimodal splice precedent.
- **G1 (small):** non-MLA standard-attention branch in the DeepSeek decoder.
- **G2 (large):** SAM-ViT-B (G2a, hard: rel-pos windowing + downsample) + CLIP-L
  (G2b, standard ViT).
- **G3 (trivial):** linear projector.
- **G4 (med):** exact OCR preprocessing + tiling.
- **G5 (med):** image-token splice, learned newline/separator, `unlimited-ocr`
  loader rule + arch detection, chat template.
- **G6 (med):** convert bf16 → MLX → **nvfp4 mixed** (protect SAM/CLIP/projector +
  attn `o_proj` + router at 8-bit; experts at 4-bit/nvfp4); CER quality gate.

## 9. Next (Phase 2)

Load `language_config` via the DeepSeek runtime + the new G1 non-MLA branch;
text-only forward-parity against the HF reference on a fixed prompt (tiny quant
fixture), exactly as DeepSeek-V2-Lite was validated. No vision yet.
