# GLM-4 (Glm4ForCausalLM) native runtime — status & follow-ups

Native Swift+MLX runtime for the GLM-4-0414 / GLM-Z1 generation (model_type
`glm4`), distinct from the legacy ChatGLM runtime in `GLMModel.swift`.

## Shipped

- `Sources/KLMCore/Glm4Model.swift` — `Glm4ForCausalLM`: separate q/k/v/o (bias on
  q/k/v only), four-RMSNorm sandwich (input / post_self_attn / post_attention /
  post_mlp), partial RoPE, fused gate_up. Built with the `native-port` workflow:
  the local gemma-4-12b-coder model drafted the Swift, then it was integrated and
  gated against an mlx-lm parity oracle.
- Routing: new `glm4` ArchitectureRule ordered BEFORE the legacy `glm` rule
  (`arch.contains("glm")` would otherwise capture it → garbage). `ModelFamily.glm4`
  + capabilities + `loadGlm4` + exhaustive-switch updates (ModelAdapter,
  ModelProfiles).
- Aliases: `glm-4-9b-0414`, `glm-z1-9b`, `glm-4-32b-0414` → mlx-community 4-bit.
- Gates: `tools/verify_glm4_parity.py` + `Glm4ParityTests` (logit parity vs mlx-lm,
  argmax-exact / cosine > 0.9999); `testGLM4Detection`. Real-checkpoint smoke
  (glm-4-9b-0414) loads in ~0.9s and generates coherent text.
- Bug fixed in the process: `createCachedCausalMask` defaulted to fp16; nvfp4
  dequants activations to bf16 → SDPA "mask must promote to bf16" crash. Glm4 now
  builds the mask in the activation dtype (`h.dtype`).

## Honest benchmark (GLM-4-9B-0414, 4-bit, single box, 200 tok)

3-way runtime comparison (`tools/bench_runtimes.py`):

| Runtime | decode tok/s | cold-start (s) |
|---|---|---|
| GGUF (llama.cpp / Ollama) | 36.0 | 2.2 |
| MLX-Python (mlx-lm) | 47.8 | 5.9 |
| Native MLX-Swift (Krill) | 40.6 | **0.9** |

Honest reading: Krill **leads cold-start** (no-Python native load) and beats
llama.cpp on decode, but **trails mlx-lm on single-stream decode (~13%)**. The
real, defensible leadership is capability (the whole GLM-4 family runs natively;
Ollama-MLX and mlx-lm AWQ cannot) + cold-start + concurrency — NOT single-stream
decode.

## Open follow-ups (NOT quick wins on a 24GB box)

1. **Decode ~13% gap vs mlx-lm.** Ruled OUT (measured): the 256MB MLX pool cap
   (`MLXMemoryConfig.swift:31`; uncapped = no change), per-token detok in the
   timer (`KRILL_BENCH_NO_DETOK=1` compute-only = full), and a measurement-window
   artifact (mlx-lm 0.31.3 doesn't compile generate). The gap is real MLX forward
   throughput on identical weights → needs an MLX Metal-trace / per-op profile of
   the Glm4 decode forward vs mlx-lm. May be partly bandwidth-bound (see
   docs/CEILINGS_AND_REATTEMPTS.md). `KRILL_BENCH_NO_DETOK` is a kept diagnostic.

2. **Quant quality — RAM-blocked here.** Goal: beat the mlx-community affine-4bit
   on fidelity-to-bf16 (`QuantFidelityTests`: top-1 + KL vs bf16). Results on this
   box: mixed-nvfp4 (o_proj@8) is WORSE than affine-4bit (90.6% vs 92.5% top-1);
   AWQ is `glm4`-NYI in mlx-lm; GPTQ and DWQ **OOM** (18.8GB bf16 + calibration >
   25.7GB RAM). Re-attempt GPTQ/DWQ on a >32GB machine to produce a genuine
   quality-leader 4-bit, then publish.

## Reusable artifacts from this work

- `tools/bench_runtimes.py` — 3-way (GGUF / MLX-Python / Native MLX-Swift) bench,
  family-agnostic via flags.
- `Tests/KLMCoreTests/QuantFidelityTests.swift` — logit-fidelity-vs-bf16 harness
  (honors per-module quant overrides; the only consistent way to compare a mixed
  Krill quant against a baseline).
- `.claude/skills/native-port/` — the general port skill (`families/glm4.md`).
