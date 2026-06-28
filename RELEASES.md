# Krill Releases

A short, human-readable note on **what each release ships** — one blurb per
version. This is the quick "what's new" view; the granular, categorized history
lives in [`CHANGELOG.md`](CHANGELOG.md), and install/usage lives in the
[`README`](README.md). New releases get a brief entry here first.

---

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
