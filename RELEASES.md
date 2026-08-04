# Krill Releases

A short, human-readable note on **what each release ships** — one blurb per
version. This is the quick "what's new" view; the granular, categorized history
lives in [`CHANGELOG.md`](CHANGELOG.md), and install/usage lives in the
[`README`](README.md). New releases get a brief entry here first.

---

## v0.18.0 — 2026-08-05
**Krill opens as an agent now.** Bare `krill` lands in the full-screen agent
TUI — tools on, read-only `plan` permissions until you raise them — with a new
toolset (`now`, `todo`, and a native `repo_map` that hands even small models an
accurate map of your codebase), the Krill.md loop (`/init` writes it, every
session reads it), and ambient context so the model knows the date, directory,
and machine without burning a turn. `web_search` now degrades to keyless
DuckDuckGo instead of failing when a backend is missing. The TUI got a full
polish pass: live status above the input box, collapsible tool output
(`ctrl+o`), live tok/s in the footer, aligned columns, branded ember loading,
and a session receipt on exit. Under the hood, the server hardening line
shipped: bearer auth for remote serving, transactional model pulls with
rollback, and digest-verified installs and self-updates.

## v0.17.0 — 2026-07-28
**Sampling was broken for every model at any temperature above zero.** The
sampler handed `MLXRandom.categorical` probabilities where it expects logits, so
it softmaxed an already-softmaxed distribution — flattening it toward uniform and
giving tokens the filters had *rejected* the same weight as the model's top
choice. Greedy decoding never touches that path, which is why `--temp 0` always
looked right and the bug shipped. If you ever set a temperature and got fluent
nonsense, this was why.

Two more things that only show up when you stream one token at a time: a
trailing `Strip` decoder ate the leading space of every token on SentencePiece
models (`ThecapitalofFranceisParis.`), and byte-fallback newlines were dropped
entirely, flattening markdown lists onto one line. Mistral was affected too.

Also adds native support for **Nanbeige 4.2 3B** — a *looped* transformer that
runs its 22 blocks twice over the same weights, giving 44 effective layers from
3B parameters' worth of memory. It holds 44 KV caches rather than 22, applies its
final norm at the end of every loop, and decouples `head_dim` from `hidden_size`.
llama.cpp cannot load the architecture at all and mlx-lm has no port, so Krill's
runtime is gated against the authors' own PyTorch code — and beats it 3.9x on
decode at a fifth of the memory. Ships as nvfp4 (2.26 GB from 8.3 GB):
`krill pull nanbeige-4.2-3b`. Apache 2.0, built on `Nanbeige/Nanbeige4.2-3B` —
credit to the Nanbeige team at BOSS Zhipin.

One usability note that came with it: a reasoning model can spend an entire
512-token budget inside a hidden `<think>` block and return *nothing*. Single-shot
`krill run` now gives thinking models the same headroom interactive chat already
had.

## v0.16.3 — 2026-07-26
A correctness release, found by installing Krill from scratch and following our
own instructions. **`gemma-4-e2b` — the model the install caveats tell you to
pull first — crashed on its first generation.** 4-bit checkpoints ship the
per-layer projection already quantized; the loader was skipping it, so a dense
layer held a packed tensor and the first matmul trapped. Fixed, and the golden
path (`brew install` → `krill pull gemma-4-e2b` → `krill run gemma-4-e2b`) works
end to end again.

Also: **tool names are now constrained while the model samples them.** A model
trained on another harness's vocabulary used to ask for `Read` where Krill
offers `read_file` and die on its first tool call. A trigger-activated grammar
now makes an unknown tool name unrepresentable rather than something to detect
and repair — at no measurable decode cost. See
[`docs/TOOL_NAME_RESOLUTION.md`](docs/TOOL_NAME_RESOLUTION.md).

Plus a quieter CLI: diagnostics go to stderr instead of corrupting `krill list |
…`, the one-line installer no longer asks for a password it does not need, and
the pull progress bar reaches 100%.

## v0.16.2 — 2026-07-13
Adds native support for **NVIDIA LocateAnything-3B**, a visual-grounding VLM that
locates anything in an image as bounding boxes (`<box><x1><y1><x2><y2></box>`,
coords 0–1000). It's a new native Swift+MLX runtime: the **MoonViT** (Kimi-VL)
native-resolution vision tower + a 2-layer connector + a **Qwen2.5-3B** text
decoder, with the vision path logit-parity-verified against the NVIDIA reference.
Ships as a mixed-precision **nvfp4** build (~3.1 GB, grounding-parity with bf16,
~50 tok/s) at `srv-sngh/LocateAnything-3B-mlx-nvfp4` under the NVIDIA License
(non-commercial). `krill pull locateanything-3b`, then
`krill run locateanything-3b "Locate the red car." --image street.jpg`.
Built on `nvidia/LocateAnything-3B` — credit to the original authors.

## v0.16.1 — 2026-07-01
Adds the **`qwythos-9b-nvfp4`** model — the `empero-ai/Qwythos-9B-Claude-Mythos-5-1M`
fine-tune in Krill's mixed-precision **nvfp4** format. It's a Qwen3.5-class hybrid
(GatedDeltaNet linear-attention + full attention), the same architecture as
Ornith-9B, so it runs on Krill's existing **native `.qwen35` text decoder** (vision
deferred to mlx_vlm) — no new runtime. The build is g16 nvfp4 with `down_proj`/`o_proj`
protected at 8-bit affine and the vision tower preserved (~6.4 GB), published at
`srv-sngh/Qwythos-9B-Claude-Mythos-5-1M-mlx-nvfp4`. `krill pull qwythos-9b-nvfp4`.
Built on `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` (Apache-2.0) — credit to the original authors.

## v0.16.0 — 2026-06-30
Adds a **native Swift + MLX runtime for Unlimited-OCR (DeepSeek-OCR)** —
`krill run unlimited-ocr --image <page> "document parsing."` parses documents and
images to grounded text **natively on Apple Silicon**: no Python, no
`trust_remote_code`. It's a DeepSeek-MoE language backbone + the native
**DeepEncoder** vision tower (SAM-ViT-B + CLIP-L + projector), with the vision
features spliced at the `<image>` block before the LM. Every stage is
parity-validated against the HF reference, and it reads real multi-line invoices
(titles, line-item tables, totals) correctly.

Ships as a **2.2 GB mixed-precision Krill blob** (`srv-sngh/Unlimited-OCR-mixed-nvfp4`,
from the 6.67 GB bf16 source): MoE experts at **nvfp4**, attention / FFN / embed /
lm_head / vision tower at 8-bit affine. nvfp4 expert support is an *additive*
mode on the shared switched-expert runtime, so every other MoE family is
unchanged. Serves the parity-validated **base view** (full pages, including wide
layouts); gundam tiling for very dense scans is a tracked follow-up. Built on
`baidu/Unlimited-OCR` (MIT) — credit to the original authors.

## v0.15.0 — 2026-06-28
**Web search works out of the box.** `web_search` (and `DeepResearch` /
`POST /research`) now ships a keyless **DuckDuckGo** backend as the default
(`search_backend = auto`) — a fresh install can search the web with no setup.
For reliable, rate-limit-free results, opt into a **BYOK** backend: **Brave** or
**Tavily** (both free tiers), or self-hosted **SearXNG**. API keys are redacted
in `/config`. Also surfaces DuckDuckGo rate-limiting instead of returning silent
empty results, and fixes Ornith-9B (qwen3_5) looping on multi-turn/agentic turns
by degrading an unrenderable chat template to ChatML rather than Llama-3.

## v0.14.0 — 2026-06-27
Adds a **native Swift + MLX runtime for Ornith-1.0-9B** (`krill run ornith-9b`) —
a Qwen3.5-class hybrid (GatedDeltaNet linear-attention/SSM layers interleaved
with full softmax-attention), ported from scratch and verified to match mlx_vlm
token-for-token. Text is served natively; the vision tower runs via mlx_vlm for
now. Quant published at `srv-sngh/Ornith-1.0-9B-4bit` (int4; nvfp4 to follow in
v0.14.1). Credit to the original creators, `deepreinforce-ai/Ornith-1.0-9B`.

Also ships **`krill update`** — a self-update command that checks the latest
release, semver-compares the running `KrillVersion`, and re-runs the installer
(Homebrew installs are redirected to `brew upgrade`).

## v0.13.0 — 2026-06-25
Adds the **`gemma-4-12b-agentic`** model — the Gemma-4-12B *agentic* fine-tune in
Krill's mixed-NVFP4 format, runnable with `krill pull gemma-4-12b-agentic`. It's
uniquely Krill-loadable (gemma4_unified + mixed-NVFP4; not GGUF/transformers).
Benchmarks and card on Hugging Face (`srv-sngh/…-agentic-…-v2-nvfp4`).

## v0.12.0 — 2026-06-25
Completed the **KrillLM → Krill** rename across every surface (tap is now
`srvsngh99/krill`, artifacts `krill-<version>`, `~/.krillm` → `~/.krill`
migration kept). Added a one-line `curl | sh` installer and a unified
chat + agent TUI (`/agent` toggles tools, `Shift+Tab` cycles permission posture).

## v0.10.0 — 2026-06-19
Native Swift+MLX **GLM-4-0414 / GLM-Z1** runtime.

## v0.9.0 — 2026-06-17
Earlier 0.9.x line. See `CHANGELOG.md` for details.

## v0.8.0 — 2026-06-17
Earlier 0.8.x line. See `CHANGELOG.md` for details.
