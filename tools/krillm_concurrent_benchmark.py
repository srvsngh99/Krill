#!/usr/bin/env python3
"""Concurrent-throughput benchmark: KrillLM vs Ollama under N simultaneous streams.

Single-stream decode on Apple silicon is memory-bandwidth bound, so neither
engine can out-decode the other by much on one request. The lever is
CONCURRENCY: one weight read can serve many decode rows. KrillLM's continuous
batcher amortizes weights across rows; Ollama serializes/limits concurrent
generates. This harness drives N simultaneous `/api/generate` streams against
each engine, sweeps N, and reports AGGREGATE decode tok/s — the axis where the
batcher wins.

Both engines must already be running and speak the Ollama `/api/generate`
protocol (KrillLM serves it natively). Server-mode only — concurrent throughput
is meaningless for a CLI subprocess.

Examples:
    # KrillLM server with n-gram + batching enabled vs Ollama daemon
    KRILL_NUM_PARALLEL=16 KRILL_NGRAM_SPEC=1 krillm serve &   # (started separately)
    python3 tools/krillm_concurrent_benchmark.py \
        --krillm-url http://127.0.0.1:11434 --krill-model llama-3.2-3b \
        --ollama-host http://127.0.0.1:11435 --ollama-model llama3.2:3b \
        --concurrency-sweep 1,2,4,8,16 --max-tokens 128

    # Baseline arm labeling (run twice against servers launched with
    # KRILL_NUM_PARALLEL=1 then =16) to find the batched-vs-serial crossover:
    python3 tools/krillm_concurrent_benchmark.py ... --server-arm serial
    python3 tools/krillm_concurrent_benchmark.py ... --server-arm batched
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

DEFAULT_PROMPTS = [
    "Write a short paragraph explaining how photosynthesis works.",
    "List five tips for writing clean, maintainable code.",
    "Summarize the plot of a classic adventure novel in a few sentences.",
    "Explain the difference between TCP and UDP to a beginner.",
    "Describe how a bicycle stays upright while moving.",
    "Give step-by-step instructions for making a simple omelette.",
    "What are the main causes of inflation? Explain briefly.",
    "Write a friendly email inviting a colleague to a lunch meeting.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--krillm-url", help="KrillLM server base URL (e.g. http://127.0.0.1:11434).")
    p.add_argument("--krill-model", default="llama-3.2-3b", help="KrillLM model name.")
    p.add_argument("--ollama-host", help="Ollama API base URL (e.g. http://127.0.0.1:11435).")
    p.add_argument("--ollama-model", default="llama3.2:3b", help="Ollama model name.")
    p.add_argument("--concurrency-sweep", default="1,2,4,8,16",
                   help="Comma-separated concurrency levels to sweep.")
    p.add_argument("--max-tokens", type=int, default=128, help="Tokens generated per request.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--prompt-file", help="One prompt per line; cycled across streams. "
                   "Defaults to a built-in set of distinct prompts.")
    p.add_argument("--server-arm", choices=["serial", "batched", "unspecified"],
                   default="unspecified",
                   help="Label only — describes how the KrillLM server was launched "
                        "(KRILL_NUM_PARALLEL=1 vs >=16), so a serial-vs-batched A/B "
                        "lands in one comparable report.")
    p.add_argument("--output", default=".build/benchmarks/concurrent-throughput.json")
    return p.parse_args()


def prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if lines:
            return lines
    return DEFAULT_PROMPTS


def one_request(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Stream one /api/generate request; return per-request timing + token counts."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first_token_s: Optional[float] = None
    text_parts: list[str] = []
    final: Optional[dict[str, Any]] = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                if not raw.strip():
                    continue
                ev = json.loads(raw)
                t = ev.get("response") or ""
                if t and first_token_s is None:
                    first_token_s = time.perf_counter() - start
                text_parts.append(t)
                if ev.get("done"):
                    final = ev
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc)}
    wall = time.perf_counter() - start
    if final is None:
        return {"ok": False, "error": "stream ended without final stats"}
    eval_count = int(final.get("eval_count") or 0)
    eval_dur_s = int(final.get("eval_duration") or 0) / 1e9
    text = "".join(text_parts)
    return {
        "ok": True,
        "wall_s": wall,
        "ttft_ms": first_token_s * 1000 if first_token_s is not None else None,
        "generated_tokens": eval_count,
        "decode_tps": eval_count / eval_dur_s if eval_dur_s > 0 else None,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def payload(model: str, prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": model, "prompt": prompt, "stream": True,
        "options": {"temperature": args.temperature, "top_p": args.top_p,
                    "num_predict": args.max_tokens},
    }


def run_concurrency(base_url: str, model: str, n: int, ps: list[str],
                    args: argparse.Namespace) -> dict[str, Any]:
    """Fire N requests at once (distinct prompts, cycled); aggregate the results."""
    url = base_url.rstrip("/") + "/api/generate"
    payloads = [payload(model, ps[i % len(ps)], args) for i in range(n)]
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda pl: one_request(url, pl, args.timeout), payloads))
    wall = time.perf_counter() - wall_start

    ok = [r for r in results if r.get("ok")]
    failed = len(results) - len(ok)
    total_tokens = sum(r["generated_tokens"] for r in ok)
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    per_req_tps = [r["decode_tps"] for r in ok if r["decode_tps"]]

    def pct(xs: list[float], q: float) -> Optional[float]:
        if not xs:
            return None
        xs = sorted(xs)
        k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
        return xs[k]

    return {
        "concurrency": n,
        "failed": failed,
        "aggregate_decode_tps": total_tokens / wall if wall > 0 else None,
        "wall_s": wall,
        "total_generated_tokens": total_tokens,
        "per_request_decode_tps_p50": pct(per_req_tps, 0.5),
        "per_request_decode_tps_p99": pct(per_req_tps, 0.99),
        "ttft_ms_p50": pct(ttfts, 0.5),
        "ttft_ms_p99": pct(ttfts, 0.99),
    }


def environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "KRILL_NUM_PARALLEL": os.environ.get("KRILL_NUM_PARALLEL"),
        "KRILL_NGRAM_SPEC": os.environ.get("KRILL_NGRAM_SPEC"),
        "KRILL_SPEC_CONCURRENCY_MAX": os.environ.get("KRILL_SPEC_CONCURRENCY_MAX"),
    }


def main() -> int:
    args = parse_args()
    if not args.krillm_url and not args.ollama_host:
        print("error: pass --krillm-url and/or --ollama-host (server-mode only).", file=sys.stderr)
        return 77
    sweep = [int(x) for x in args.concurrency_sweep.split(",") if x.strip()]
    ps = prompts(args)

    report: dict[str, Any] = {"environment": environment(), "server_arm": args.server_arm,
                              "max_tokens": args.max_tokens, "engines": {}}

    engines: list[tuple[str, str, str]] = []
    if args.krillm_url:
        engines.append(("krillm", args.krillm_url, args.krill_model))
    if args.ollama_host:
        engines.append(("ollama", args.ollama_host, args.ollama_model))

    for name, url, model in engines:
        rows = []
        for n in sweep:
            try:
                rows.append(run_concurrency(url, model, n, ps, args))
            except Exception as exc:  # noqa: BLE001 — report, don't abort the sweep
                rows.append({"concurrency": n, "error": str(exc)})
        report["engines"][name] = rows

    # Print a comparison table.
    hdr = f"{'N':>3} | {'krillm agg tok/s':>17} | {'ollama agg tok/s':>17} | {'ratio':>6} | {'krillm p99 TTFT':>16} | {'ollama p99 TTFT':>16}"
    print(f"\nConcurrent throughput (arm={args.server_arm}, max_tokens={args.max_tokens})")
    print(hdr)
    print("-" * len(hdr))
    k_rows = {r["concurrency"]: r for r in report["engines"].get("krillm", []) if "concurrency" in r}
    o_rows = {r["concurrency"]: r for r in report["engines"].get("ollama", []) if "concurrency" in r}
    for n in sweep:
        k = k_rows.get(n, {})
        o = o_rows.get(n, {})
        ktps = k.get("aggregate_decode_tps")
        otps = o.get("aggregate_decode_tps")
        ratio = (ktps / otps) if (ktps and otps) else None
        print(f"{n:>3} | {fmt(ktps):>17} | {fmt(otps):>17} | {fmt(ratio,2):>6} | "
              f"{fmt(k.get('ttft_ms_p99')):>16} | {fmt(o.get('ttft_ms_p99')):>16}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {args.output}")
    return 0


def fmt(x: Optional[float], nd: int = 1) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


if __name__ == "__main__":
    raise SystemExit(main())
