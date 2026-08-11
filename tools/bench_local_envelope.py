#!/usr/bin/env python3
"""Measure what a local model can actually do on a memory-constrained Mac.

Answers the question "how big a model, at how much context, before this machine
stops being useful?" — by sweeping (model x context) and recording throughput
AND the memory/paging cost of getting it.

The point is the SECOND half. Any benchmark can print tok/s; the number that
decides whether a model is usable on 24 GB is whether the machine was paging
while it produced that number. A model that decodes at 1 tok/s with 17 GB of
swap is not "slow", it is out of envelope, and no amount of tuning fixes it.

Measurement goes through `krill bench`, NOT by scraping `krill run`. An earlier
version of this script did the latter and was silently wrong in two ways:

  * `krill run` prints its stats line only when generation is CUT OFF by
    `--max-tokens`. A model that stops at EOS prints nothing, so the sweep
    measured verbose models and left concise ones blank.
  * its "context" was a filler prompt sized by a words-per-token guess.
    `krill bench --prompt-len` is exact, in tokens.

`krill bench` also forces the generation length, averages over runs, and does a
warmup pass, so a cold Metal pipeline no longer lands in the first number.

Guardrails (this sweep can and will push a small machine to its limit):
  * hard abort if free disk falls below --disk-floor GB, because swap lives on
    disk and filling it destabilises the whole OS;
  * per-run timeout, so a thrashing model cannot hang the sweep;
  * runs are sequential, never concurrent. Do not use the machine while this
    runs — one concurrent chat measured 6.6 tok/s for a model this sweep
    measured at 41.9.

Usage:
    python3 tools/bench_local_envelope.py \\
        --models qwen3-0.6b llama-3.2-3b qwen3-8b \\
        --contexts 512 4096 16384 \\
        --out docs/bench/local-envelope.md
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

# The `Results (avg of N runs)` block `krill bench` prints last. Parsed
# individually rather than as one block regex so a single missing line
# degrades that field instead of dropping the whole measurement.
RE_PREFILL = re.compile(r"^Prefill:\s+([\d.]+)\s*tok/s", re.M)
RE_DECODE = re.compile(r"^Decode:\s+([\d.]+)\s*tok/s", re.M)
RE_TTFT = re.compile(r"^TTFT:\s+([\d.]+)\s*ms", re.M)
RE_PEAKMEM = re.compile(r"^Peak memory:\s+([\d.]+)\s*MB", re.M)
RE_LOAD = re.compile(r"^Load time:\s+([\d.]+)\s*s", re.M)
RE_FAMILY = re.compile(r"family:\s*([A-Za-z0-9_]+)")


def free_disk_gb():
    st = os.statvfs(os.path.expanduser("~"))
    return st.f_bavail * st.f_frsize / 2**30


def swap_used_mb():
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True).stdout
        m = re.search(r"used\s*=\s*([\d.]+)M", out)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def model_size_gb(model):
    d = os.path.expanduser(f"~/.krill/models/blobs/{model}")
    if not os.path.isdir(d):
        return None
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 2**30


def run_one(krill, model, prompt_len, gen_len, runs, warmup, timeout):
    swap0 = swap_used_mb()
    t0 = time.time()
    cmd = [krill, "bench", model,
           "--prompt-len", str(prompt_len), "--gen-len", str(gen_len),
           "--runs", str(runs), "--warmup", str(warmup)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = p.stdout + p.stderr
        timed_out = False
    except subprocess.TimeoutExpired as e:
        # `subprocess.run(text=True)` does NOT decode the streams it attaches to
        # TimeoutExpired — they come back as bytes even in text mode, and either
        # may be None. Normalise both before touching them.
        def _txt(v):
            if v is None:
                return ""
            return v.decode("utf-8", "replace") if isinstance(v, bytes) else v
        out = _txt(e.stdout) + _txt(e.stderr)
        timed_out = True
    wall = time.time() - t0

    row = {"wall_s": round(wall, 1), "timed_out": timed_out,
           "swap_delta_mb": round(swap_used_mb() - swap0, 1),
           "free_disk_gb": round(free_disk_gb(), 1)}

    def grab(rx, key, cast=float):
        m = rx.search(out)
        if m:
            row[key] = cast(m.group(1))

    grab(RE_PREFILL, "prefill_tps")
    grab(RE_DECODE, "decode_tps")
    grab(RE_TTFT, "ttft_ms")
    grab(RE_PEAKMEM, "peak_mem_mb")
    grab(RE_LOAD, "load_s")
    m = RE_FAMILY.search(out)
    if m:
        row["family"] = m.group(1)
    if "decode_tps" not in row:
        # No measurement. Keep the tail so the row stays diagnosable instead of
        # silently becoming a dash in the table.
        row["raw_tail"] = out[-400:].replace("\n", " ")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--contexts", nargs="+", type=int, default=[512, 4096, 16384],
                    help="prompt lengths in TOKENS (exact, not estimated)")
    ap.add_argument("--gen-len", type=int, default=64)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800, help="per-config seconds")
    ap.add_argument("--disk-floor", type=float, default=4.0,
                    help="abort below this many GB free")
    ap.add_argument("--krill", default=".build/arm64-apple-macosx/release/krill")
    ap.add_argument("--out", default="docs/bench/local-envelope.md")
    args = ap.parse_args()

    krill = args.krill if os.path.exists(args.krill) else shutil.which("krill")
    if not krill:
        sys.exit("krill binary not found; build it or pass --krill")

    results = []
    aborted = False
    for model in args.models:
        size = model_size_gb(model)
        for ctx in args.contexts:
            free = free_disk_gb()
            if free < args.disk_floor:
                print(f"!! ABORT: free disk {free:.1f}GB below floor "
                      f"{args.disk_floor}GB", flush=True)
                aborted = True
                break
            print(f"-> {model} @ {ctx} tok ...", flush=True, end=" ")
            row = run_one(krill, model, ctx, args.gen_len,
                          args.runs, args.warmup, args.timeout)
            row.update(model=model, prompt_len=ctx, gen_len=args.gen_len,
                       weights_gb=None if size is None else round(size, 1))
            results.append(row)
            write_report(results, args.out, args)
            print(f"decode={row.get('decode_tps','—')} tok/s "
                  f"peak={row.get('peak_mem_mb','—')}MB "
                  f"swap+{row['swap_delta_mb']:.0f}MB "
                  f"{'TIMEOUT' if row['timed_out'] else ''}", flush=True)
        if aborted:
            break

    write_report(results, args.out, args, aborted=aborted)
    print(f"\nwrote {args.out}")


def verdict_for(r):
    if r["timed_out"]:
        return "timeout"
    if "decode_tps" not in r:
        # Distinct from "slow": we have no measurement at all. Labelling this
        # out-of-envelope would defame a perfectly fast model.
        return "no data"
    if r["decode_tps"] < 5 or r["swap_delta_mb"] > 2000:
        return "**out of envelope**"
    if r["decode_tps"] < 20:
        return "usable"
    return "comfortable"


def write_report(results, out, args, aborted=False):
    """Rewrite both outputs from scratch. Called after EVERY run, so a crash or
    an abort mid-sweep still leaves every completed measurement on disk — the
    first version of this script only wrote at the end and lost a whole sweep to
    one exception."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out.replace(".md", ".json"), "w") as fh:
        json.dump(results, fh, indent=2)

    with open(out, "w") as fh:
        fh.write("# Local inference envelope\n\n")
        fh.write("Measured with `tools/bench_local_envelope.py`, which drives "
                 "`krill bench` (forced generation length, warmup pass, averaged "
                 "runs) on this machine.\n\n")
        fh.write(f"Settings: generate {args.gen_len} tokens, {args.runs} run(s) "
                 f"averaged after {args.warmup} warmup. Prompt lengths are exact "
                 "token counts.\n\n")
        fh.write("`swap Δ` is the change in system swap across the config — it is "
                 "the column that separates 'slow' from 'out of envelope'. A "
                 "negative delta just means macOS reclaimed swap another process "
                 "had taken.\n\n")
        fh.write("Read `swap Δ` with the sweep ORDER in mind: it is not a stable "
                 "property of a model. Whichever large model loads first pays the "
                 "cold-page cost and shows a big positive delta, while a later one "
                 "can show a negative delta by reusing pages the previous model "
                 "freed. `peak MB` is the order-independent memory number — prefer "
                 "it when comparing models.\n\n")
        fh.write("| model | weights | prompt tok | load s | prefill tok/s "
                 "| decode tok/s | TTFT ms | peak MB | swap Δ MB | verdict |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in results:
            fh.write(f"| {r['model']} | {r.get('weights_gb','—')} GB "
                     f"| {r['prompt_len']} | {r.get('load_s','—')} "
                     f"| {r.get('prefill_tps','—')} | {r.get('decode_tps','—')} "
                     f"| {r.get('ttft_ms','—')} | {r.get('peak_mem_mb','—')} "
                     f"| {r['swap_delta_mb']:.0f} | {verdict_for(r)} |\n")
        if aborted:
            fh.write("\n**Sweep aborted early** — free disk fell below the floor. "
                     "Rows below the abort point were never measured; their "
                     "absence is not a result.\n")


if __name__ == "__main__":
    main()
