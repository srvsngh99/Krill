#!/usr/bin/env python3
"""Concurrent-throughput benchmark: Krill vs Ollama under N simultaneous streams.

Single-stream decode on Apple silicon is memory-bandwidth bound, so neither
engine can out-decode the other by much on one request. The lever is
CONCURRENCY: one weight read can serve many decode rows. Krill's continuous
batcher amortizes weights across rows; Ollama serializes/limits concurrent
generates. This harness drives N simultaneous `/api/generate` streams against
each engine, sweeps N, and reports AGGREGATE decode tok/s — the axis where the
batcher wins.

Both engines must already be running and speak the Ollama `/api/generate`
protocol (Krill serves it natively). Server-mode only — concurrent throughput
is meaningless for a CLI subprocess.

Examples:
    # Krill server with n-gram + batching enabled vs Ollama daemon
    KRILL_NUM_PARALLEL=16 KRILL_NGRAM_SPEC=1 krill serve &   # (started separately)
    python3 tools/krill_concurrent_benchmark.py \
        --krill-url http://127.0.0.1:57455 --krill-model llama-3.2-3b \
        --ollama-host http://127.0.0.1:11434 --ollama-model llama3.2:3b \
        --concurrency-sweep 1,2,4,8,16 --max-tokens 128

    # Baseline arm labeling (run twice against servers launched with
    # KRILL_NUM_PARALLEL=1 then =16) to find the batched-vs-serial crossover:
    python3 tools/krill_concurrent_benchmark.py ... --server-arm serial
    python3 tools/krill_concurrent_benchmark.py ... --server-arm batched
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

# At least as many DISTINCT prompts as the largest swept concurrency, so a high-N
# run does not reuse a prompt — duplicate prompts hit Krill's prefix cache,
# which shrinks wall time and inflates the wall-based aggregate tok/s (a
# measurement artifact, not real decode scaling). The harness also warns if the
# sweep exceeds the prompt-set size.
DEFAULT_PROMPTS = [
    "Write a short paragraph explaining how photosynthesis works.",
    "List five tips for writing clean, maintainable code.",
    "Summarize the plot of a classic adventure novel in a few sentences.",
    "Explain the difference between TCP and UDP to a beginner.",
    "Describe how a bicycle stays upright while moving.",
    "Give step-by-step instructions for making a simple omelette.",
    "What are the main causes of inflation? Explain briefly.",
    "Write a friendly email inviting a colleague to a lunch meeting.",
    "Explain what a hash table is and when you would use one.",
    "Describe the water cycle from evaporation to precipitation.",
    "What is the role of mitochondria in a cell? Keep it short.",
    "Give three reasons regular exercise improves mental health.",
    "Explain recursion to someone who has never programmed.",
    "Summarize how vaccines train the immune system.",
    "Describe the difference between weather and climate.",
    "Write a haiku about the ocean at dawn, then explain it.",
    "Explain how a compiler differs from an interpreter.",
    "List four ways to reduce household energy consumption.",
    "Describe what happens during a solar eclipse.",
    "Explain the concept of supply and demand with an example.",
    "What makes sourdough bread rise? Explain the chemistry.",
    "Describe how GPS determines your location.",
    "Explain why the sky is blue in simple terms.",
    "Give a brief overview of how the internet routes packets.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--krill-url", help="Krill server base URL (e.g. http://127.0.0.1:57455).")
    p.add_argument("--krill-model", default="llama-3.2-3b", help="Krill model name.")
    p.add_argument("--ollama-host", help="Ollama API base URL (e.g. http://127.0.0.1:11434).")
    p.add_argument("--ollama-model", default="llama3.2:3b", help="Ollama model name.")
    p.add_argument("--concurrency-sweep", default="1,2,4,8,16",
                   help="Comma-separated concurrency levels to sweep.")
    p.add_argument("--max-tokens", type=int, default=128, help="Tokens generated per request.")
    p.add_argument("--runs", type=int, default=3, help="Measured runs per concurrency level (averaged — batch-formation timing is noisy, so >1 is recommended).")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per concurrency level (discarded).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--prompt-file", help="One prompt per line; cycled across streams. "
                   "Defaults to a built-in set of distinct prompts.")
    p.add_argument("--server-arm", choices=["serial", "batched", "unspecified"],
                   default="unspecified",
                   help="Label only — describes how the Krill server was launched "
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
        # How many successful streams yielded a first-token time. A blank cell in
        # the TTFT column with ok streams but 0 samples here means the stream
        # shape was not recognized (no per-chunk `response`), NOT that the arm was
        # skipped; the summary calls this out so the gap is never silent.
        "ttft_samples": len(ttfts),
    }


_AVG_KEYS = [
    "aggregate_decode_tps", "wall_s", "total_generated_tokens",
    "per_request_decode_tps_p50", "per_request_decode_tps_p99",
    "ttft_ms_p50", "ttft_ms_p99",
]


def averaged_concurrency(base_url: str, model: str, n: int, ps: list[str],
                         args: argparse.Namespace) -> dict[str, Any]:
    """Warmup (discarded) + `runs` measured passes, averaged — batch-formation
    timing varies run to run, so a single pass at moderate N is noisy."""
    for _ in range(max(0, args.warmup)):
        run_concurrency(base_url, model, n, ps, args)
    rows = [run_concurrency(base_url, model, n, ps, args) for _ in range(max(1, args.runs))]
    out: dict[str, Any] = {"concurrency": n, "runs": len(rows),
                           "failed": sum(r.get("failed", 0) for r in rows),
                           "ttft_samples": sum(r.get("ttft_samples", 0) for r in rows)}
    for k in _AVG_KEYS:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = statistics.mean(vals) if vals else None
    return out


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
    if not args.krill_url and not args.ollama_host:
        print("error: pass --krill-url and/or --ollama-host (server-mode only).", file=sys.stderr)
        return 77
    sweep = [int(x) for x in args.concurrency_sweep.split(",") if x.strip()]
    ps = prompts(args)
    if sweep and max(sweep) > len(ps):
        print(f"warning: max concurrency {max(sweep)} exceeds {len(ps)} distinct prompts; "
              "reused prompts hit the prefix cache and inflate wall-based throughput. "
              "Pass a larger --prompt-file for an honest high-N measurement.", file=sys.stderr)

    report: dict[str, Any] = {"environment": environment(), "server_arm": args.server_arm,
                              "max_tokens": args.max_tokens, "engines": {}}

    engines: list[tuple[str, str, str]] = []
    if args.krill_url:
        engines.append(("krill", args.krill_url, args.krill_model))
    if args.ollama_host:
        engines.append(("ollama", args.ollama_host, args.ollama_model))

    for name, url, model in engines:
        rows = []
        for n in sweep:
            try:
                rows.append(averaged_concurrency(url, model, n, ps, args))
            except Exception as exc:  # noqa: BLE001 — report, don't abort the sweep
                rows.append({"concurrency": n, "error": str(exc)})
        report["engines"][name] = rows

    # Print a comparison table.
    hdr = f"{'N':>3} | {'krill agg tok/s':>17} | {'ollama agg tok/s':>17} | {'ratio':>6} | {'krill p99 TTFT':>16} | {'ollama p99 TTFT':>16}"
    print(f"\nConcurrent throughput (arm={args.server_arm}, max_tokens={args.max_tokens})")
    print(hdr)
    print("-" * len(hdr))
    k_rows = {r["concurrency"]: r for r in report["engines"].get("krill", []) if "concurrency" in r}
    o_rows = {r["concurrency"]: r for r in report["engines"].get("ollama", []) if "concurrency" in r}
    saw_failure = False
    for n in sweep:
        k = k_rows.get(n, {})
        o = o_rows.get(n, {})
        ktps = k.get("aggregate_decode_tps")
        otps = o.get("aggregate_decode_tps")
        ratio = (ktps / otps) if (ktps and otps) else None
        # Aggregate tok/s = (successful streams' tokens) / wall. A failed stream
        # contributes no tokens but its slot still freed the GPU, so a partial
        # failure makes the number an optimistic, lower-effective-N reading —
        # mark it so it isn't trusted as a clean N-stream result.
        kfail = k.get("failed") or 0
        ofail = o.get("failed") or 0
        saw_failure = saw_failure or bool(kfail or ofail)
        kcell = fmt(ktps) + ("*" if kfail else "")
        ocell = fmt(otps) + ("*" if ofail else "")
        print(f"{n:>3} | {kcell:>17} | {ocell:>17} | {fmt(ratio,2):>6} | "
              f"{fmt(k.get('ttft_ms_p99')):>16} | {fmt(o.get('ttft_ms_p99')):>16}")
    if saw_failure:
        print("\n* one or more streams failed at this N; the aggregate tok/s "
              "sums only the surviving streams over the full wall, so it is an "
              "optimistic (lower-effective-N) reading — not a clean N-stream "
              "measurement. See per-level `failed` in the JSON.")

    # Disambiguate a blank TTFT cell: distinguish "engine arm not run" from "arm
    # ran but first-token timing was not captured". Both Krill and Ollama
    # `/api/generate` stream NDJSON with a per-chunk `response`, so a streamed arm
    # should always populate TTFT; a 0-sample arm signals a parse gap (e.g. a
    # non-streaming or differently-shaped response), the silent failure that made
    # earlier runs show a blank cell with no explanation.
    for name, rows in report["engines"].items():
        starved = [
            r["concurrency"] for r in rows
            if r.get("aggregate_decode_tps") and not r.get("ttft_samples")
        ]
        if starved:
            print(f"\nnote: {name} produced streams but captured no first-token "
                  f"timing at N={','.join(str(s) for s in starved)} (its TTFT "
                  f"cells are blank for this reason, not because the arm was "
                  f"skipped). Check that the arm streams NDJSON `/api/generate` "
                  f"with a per-chunk `response` field.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {args.output}")
    return 0


def fmt(x: Optional[float], nd: int = 1) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


if __name__ == "__main__":
    raise SystemExit(main())
