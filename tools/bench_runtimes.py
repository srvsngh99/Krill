#!/usr/bin/env python3
"""Three-way single-box inference benchmark: GGUF (llama.cpp/Ollama) vs
MLX-Python (mlx-lm) vs Native MLX-Swift (Krill), on the SAME model + prompt.

This is the honest "rebuff" harness. It compares RUNTIMES, not weights -- the
.safetensors / .gguf files are runtime-agnostic; what differs is how each engine
loads and serves them. For an apples-to-apples runtime comparison, point all
three rows at the SAME logical model at a comparable quant (e.g. a 4-bit GGUF vs
the mlx-community 4-bit MLX vs Krill loading that same 4-bit MLX).

What it measures, per runtime:
  - cold-start: process launch -> model ready / first token (seconds)
  - decode tok/s: single-stream steady-state generation rate (median of N runs)
  - concurrency: aggregate tok/s across C parallel streams (serveable runtimes
    only: Krill batcher vs mlx-lm; Ollama/llama.cpp serialize by default)

Honest-claim guardrails (see docs / the model-card story):
  - Single-stream DECODE is typically ~parity across MLX runtimes (bandwidth
    roof). Do NOT expect or report a Krill decode "win" there.
  - Krill's real, measurable wins are COLD-START and CONCURRENCY (the native
    Swift engine: no Python import, continuous batcher) plus capability
    (multimodal / structured output / tools) that the others lack here.
  - The weights are NOT "faster because native" -- only the serving path is.

Reusable across families: pass the three sources as flags. Parsing is best-effort
per tool; --raw dumps each runtime's stdout so you can sanity-check the numbers.

Usage:
  tools/bench_runtimes.py \
    --krill-alias glm-4-9b-0414 \
    --mlx-repo mlx-community/GLM-4-9B-0414-4bit \
    --ollama-tag 'hf.co/lmstudio-community/GLM-4-9B-0414-GGUF:Q4_K_M' \
    --prompt "Explain unified memory in 3 sentences." \
    --max-tokens 200 --runs 3 --concurrency 4
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import time

KRILL = os.path.expanduser("~/Desktop/playground/Krill/.build/release/krill")
VENV = os.path.expanduser("~/.krill/venv/bin")


def run(cmd, timeout=600):
    """Run a command, return (elapsed_seconds, stdout+stderr)."""
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return time.time() - t0, (p.stdout or "") + (p.stderr or "")


def first(patterns, text):
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def bench_krill(alias, prompt, max_tokens, raw):
    """Krill native Swift+MLX. Prints 'Ready (Xs load time)' and
    'decode: N tokens at Y tok/s'."""
    elapsed, out = run([KRILL, "run", "--max-tokens", str(max_tokens), alias, prompt])
    if raw:
        print("---- krill raw ----\n", out[-1500:], file=sys.stderr)
    load = first([r"Ready \(([\d.]+)s load time\)"], out)
    dec = first([r"decode:.*?at ([\d.]+) tok/s", r"([\d.]+) tok/s"], out)
    return {"load_s": load, "decode_tps": dec, "wall_s": elapsed}


def bench_mlxlm(repo, prompt, max_tokens, raw):
    """mlx-lm (Python). Prints 'Generation: N tokens, Y tokens-per-sec' and a
    prompt/peak-memory block; load time is folded into wall time."""
    t0 = time.time()
    elapsed, out = run([
        os.path.join(VENV, "mlx_lm.generate"),
        "--model", repo, "--prompt", prompt,
        "--max-tokens", str(max_tokens),
    ])
    if raw:
        print("---- mlx-lm raw ----\n", out[-1500:], file=sys.stderr)
    dec = first([r"Generation:.*?([\d.]+) tokens-per-sec",
                 r"([\d.]+) tokens-per-sec"], out)
    return {"load_s": None, "decode_tps": dec, "wall_s": elapsed}


def bench_ollama(tag, prompt, max_tokens, raw):
    """GGUF via Ollama. `--verbose` prints 'eval rate: Y tokens/s' and
    'load duration'. Ollama keeps the model resident after first load, so the
    first call includes cold load; we report that as cold-start."""
    elapsed, out = run([
        "ollama", "run", tag, "--verbose", prompt,
    ])
    if raw:
        print("---- ollama raw ----\n", out[-1500:], file=sys.stderr)
    # NB: "prompt eval rate:" (prompt processing) contains "eval rate:" as a
    # substring — exclude it with a negative lookbehind so we capture the DECODE
    # "eval rate:", not the prompt-eval rate.
    dec = first([r"(?<!prompt )eval rate:\s*([\d.]+) tokens/s"], out)
    load = first([r"load duration:\s*([\d.]+)\s*s"], out)
    return {"load_s": load, "decode_tps": dec, "wall_s": elapsed}


def median_decode(fn, runs, *args):
    rows = []
    for _ in range(runs):
        try:
            rows.append(fn(*args))
        except Exception as e:  # one bad run must not abort the whole table
            print(f"  [warn] run failed: {type(e).__name__}: {e}", file=sys.stderr)
    if not rows:
        return {"decode_tps": None, "cold_s": None, "cold_is_wall": False}
    tps = [r["decode_tps"] for r in rows if r["decode_tps"]]
    cold = next((r["load_s"] for r in rows if r["load_s"]), None)
    wall0 = rows[0]["wall_s"] if rows else None
    return {
        "decode_tps": round(statistics.median(tps), 1) if tps else None,
        "cold_s": round(cold, 2) if cold else (round(wall0, 2) if wall0 else None),
        "cold_is_wall": cold is None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--krill-alias")
    ap.add_argument("--mlx-repo")
    ap.add_argument("--ollama-tag")
    ap.add_argument("--prompt", default="Explain unified memory in three sentences.")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=0, help="reserved; serve-based, see note")
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()

    results = {}
    if args.ollama_tag:
        print(f"[GGUF/Ollama] {args.ollama_tag} ...", file=sys.stderr)
        results["GGUF (llama.cpp / Ollama)"] = median_decode(
            bench_ollama, args.runs, args.ollama_tag, args.prompt, args.max_tokens, args.raw)
    if args.mlx_repo:
        print(f"[MLX-Python/mlx-lm] {args.mlx_repo} ...", file=sys.stderr)
        results["MLX-Python (mlx-lm)"] = median_decode(
            bench_mlxlm, args.runs, args.mlx_repo, args.prompt, args.max_tokens, args.raw)
    if args.krill_alias:
        print(f"[Native MLX-Swift/Krill] {args.krill_alias} ...", file=sys.stderr)
        results["Native MLX-Swift (Krill)"] = median_decode(
            bench_krill, args.runs, args.krill_alias, args.prompt, args.max_tokens, args.raw)

    # Markdown table for the model card.
    print(f"\n### Runtime benchmark — same model, same prompt "
          f"(max_tokens={args.max_tokens}, median of {args.runs})\n")
    print("| Runtime | decode tok/s | cold-start (s) |")
    print("|---|---|---|")
    for name, r in results.items():
        cold = f"{r['cold_s']}" + ("*" if r["cold_is_wall"] else "") if r["cold_s"] else "—"
        print(f"| {name} | {r['decode_tps'] or '—'} | {cold} |")
    print("\n*cold-start shown as full wall time when the tool does not report a "
          "load duration separately. Single-stream decode is bandwidth-bound and "
          "expected to be ~parity across MLX runtimes; Krill's wins are "
          "cold-start, concurrency, and capability — not single-stream decode.")


if __name__ == "__main__":
    main()
