# Gemma-4-12B — Open Issues

Defects found during the full benchmark run on 2026-06-09 (KrillLM main `c5387b0`,
release build, registered canonical `gemma-4-12b` = nvfp4-oproj8 requant). Both are in
the **shipped nvfp4 model**, not the engine. Full scorecard:
`~/.claude/plans/gemma12b-full-bench-2026-06-09.md`.

---

## Issue 1 — nvfp4 vision degradation: pure red misread as "brown"  [RESOLVED]

**Status:** RESOLVED (2026-06-09). Root cause confirmed (red-channel attenuation
from nvfp4-quantizing the vision projector), fixed by auto-protecting the
vision/audio projectors at 8-bit in `tools/requant_gemma4_nvfp4.py`, re-registered
both `gemma-4-12b` / `gemma-4-12b-nvfp4` blobs from the corrected requant, and
gated by `tools/verify_gemma4_vision_color.py` (now 6/6 colors correct, text
unchanged: Paris / 391 / Buenos dias). Detail retained below for the record.

**Severity:** medium (functional, color-specific). Multimodal still works; one channel is off.

### Symptom
The registered `gemma-4-12b` reads a solid pure-red image as **"Brown"**, while blue and
green are correct. Reproduced at 96x96 and 448x448, temp 0, single word forced.

```
RED   (255,0,0) -> 'Brown'   WRONG
BLUE  (0,0,255) -> 'Blue'     ok
GREEN (0,255,0) -> 'Green'    ok
```

### Repro
```bash
# serve the registered model
KRILL_NO_AUTO_DAEMON=1 .build/release/krillm serve --model gemma-4-12b --port 57455
# POST a solid-red PNG (base64) in the `images` array of /api/generate, prompt
# "What color is this image? Reply with only the color name." -> 'Brown'
```

### Root cause (confirmed)
The nvfp4 requant quantized the **vision projector**. The blob's
`model.safetensors.index.json` shows these carry `.scales` (i.e. nvfp4-quantized):
- `vision_embedder.patch_dense.{weight,scales,bias}`
- `embed_vision.embedding_projection.{weight,scales}`
- (`embed_audio.embedding_projection` is likewise quantized)

Text MMLU is unaffected (77.6%) because text never touches the vision path. The PR #171
multimodal correctness gate ran on the **non-requant** unified checkpoint, so this
degradation was never gated against. nvfp4's float grid still mostly works (blue/green
correct) but loses enough precision on the red patch-embedding to shift it toward brown.

### Fix direction
1. In `tools/requant_gemma4_nvfp4.py`, **protect the vision (and audio) projectors** —
   keep `vision_embedder.patch_dense`, `embed_vision.embedding_projection`,
   `embed_audio.embedding_projection` at 8-bit or bf16 (byte-passthrough), mirroring the
   existing attn `o_proj`@8-bit protection. These are small tensors, so the size cost is
   negligible.
2. Re-register `gemma-4-12b` from the corrected requant.
3. **Add a color/vision parity gate that runs on the REQUANT checkpoint** (not just the
   source): solid R/G/B/C/M/Y images -> correct color name. This is the gate gap that let
   the defect ship.

### Acceptance
RED/BLUE/GREEN (and CMY) all return the correct color name on the registered model;
text MMLU unchanged (~77.6%); vision parity gate green on the requant ckpt.

---

## Issue 2 — thinking-channel markers leak into responses

**Severity:** medium (UX/formatting; affects every response).

### Symptom
Responses are prefixed with the literal CoT-channel markers on **both text and image
paths**:

```
'<|channel>thought\n<channel|>Blue'                       # image prompt
'<|channel>thought <channel|>The ocean is the lifeblood…' # text prompt (cold krillm run)
```

The actual answer follows the markers, but the markers themselves should never reach the
user.

### Context
Gemma-4-12B opens a CoT/thinking channel (known: mlx-swift 0.31.4 makes it open long CoT
on hard prompts). Ollama suppresses this by default and exposes `think:false`. KrillLM
emits the thinking-channel markers inline in the user-visible `response`.

### Fix direction
- Strip the thinking-channel span (`<|channel>thought ... <channel|>`) from the
  user-visible response before returning it, OR
- Default thinking off for this model (with an opt-in to surface it), matching Ollama's
  `think:false` default behavior.
- Apply on both the server (`/api/generate`, `/v1/chat/completions`) and the `krillm run`
  CLI path (the cold-run output showed the leak too).

### Acceptance
A plain prompt returns the answer with no `<|channel>` markers on both the server and CLI
paths; thinking content is either suppressed or surfaced through a dedicated field/flag,
not inlined into `response`.

---

## Notes
- Neither blocks the benchmark conclusions: single-stream decode parity, prefill 1.40x,
  cold 1.7-1.9x, concurrency 1.50x@N16, tools 4/4 parity, multimodal KrillLM-only.
- Fixing both makes the capability lead clean (correct vision + clean output), which is
  the genuine "miles ahead" axis vs Ollama's text-only MLX gemma tag.
