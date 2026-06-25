# Krill Releases

A short, human-readable note on **what each release ships** — one blurb per
version. This is the quick "what's new" view; the granular, categorized history
lives in [`CHANGELOG.md`](CHANGELOG.md), and install/usage lives in the
[`README`](README.md). New releases get a brief entry here first.

---

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
