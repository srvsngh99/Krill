#!/usr/bin/env python3
"""Reproducible KrillLM vs Ollama benchmark harness.

The harness intentionally records inputs and environment rather than publishing
fixed claims. It exits 77 when local prerequisites are missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple


SKIP_EXIT_CODE = 77
DEFAULT_PROMPT = "Explain quantum computing in simple terms."
STATS_RE = re.compile(
    r"prompt:\s+(?P<prompt_tokens>\d+)\s+tokens,\s+"
    r"prefill:\s+(?P<prefill_tps>[0-9.]+)\s+tok/s,\s+"
    r"decode:\s+(?P<generated_tokens>\d+)\s+tokens\s+at\s+"
    r"(?P<decode_tps>[0-9.]+)\s+tok/s,\s+"
    r"TTFT:\s+(?P<ttft_ms>[0-9.]+)ms,\s+"
    r"total:\s+(?P<total_s>[0-9.]+)s"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible KrillLM vs Ollama local benchmark."
    )
    parser.add_argument("--krillm-bin", help="Path to krillm binary. Defaults to .build/release/krillm or PATH.")
    parser.add_argument("--ollama-bin", default="ollama", help="Path to ollama binary.")
    parser.add_argument("--krill-model", default="llama-3.2-1b", help="KrillLM registry model name or model directory.")
    parser.add_argument("--ollama-model", default="llama3.2:1b", help="Ollama model name.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text used for both engines.")
    parser.add_argument("--prompt-file", help="Read prompt text from this file.")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum generated tokens per run.")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs per engine.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs per engine.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed passed to both engines.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature passed to both engines.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p passed to both engines.")
    parser.add_argument("--krillm-url", help="KrillLM server URL (e.g. http://127.0.0.1:11435). When set, benchmarks against a running KrillLM server instead of CLI subprocess.")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434", help="Ollama API host.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-run timeout in seconds.")
    parser.add_argument("--krillm-draft-model", default=None, help="Enable speculative decoding by loading this draft model (alias, path, or 'auto'). KrillLM only.")
    parser.add_argument(
        "--output",
        default=".build/benchmarks/krillm-vs-ollama.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["cold", "warm", "cache_hit", "auto"],
        default="auto",
        help=(
            "Cache labelling mode for results. 'auto' infers per-run labels "
            "from --warmup and prefix-cache heuristics (cold=first run with "
            "warmup==0, warm=runs after warmup, cache_hit=server prefix-cache "
            "active on repeated prompts). When set to an explicit value, "
            "every result is force-tagged with that label and "
            "cache_mode_source is recorded as 'explicit'."
        ),
    )
    return parser.parse_args()


def run_cmd(command: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def find_krillm_binary(value: Optional[str]) -> Optional[str]:
    if value:
        return value if os.access(value, os.X_OK) else None
    repo_binary = Path(".build/release/krillm")
    if repo_binary.exists() and os.access(repo_binary, os.X_OK):
        return str(repo_binary)
    return shutil.which("krillm")


def prompt_text(args: argparse.Namespace) -> str:
    if not args.prompt_file:
        return args.prompt
    return Path(args.prompt_file).read_text(encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_krillm_list(output: str) -> set[str]:
    models: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("NAME") or stripped.startswith("No models"):
            continue
        models.add(stripped.split()[0])
    return models


def krillm_model_ok(krillm_bin: str, model: str) -> Tuple[bool, Optional[str], dict[str, Any]]:
    if Path(model).expanduser().exists():
        return True, None, {"source": "path", "path": str(Path(model).expanduser())}

    completed = run_cmd([krillm_bin, "list"], timeout=30.0)
    detail = {
        "command": [krillm_bin, "list"],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        return False, f"could not list KrillLM models; run `{krillm_bin} list` for details", detail
    if model not in parse_krillm_list(completed.stdout):
        return False, f"KrillLM model `{model}` is not installed; run `{krillm_bin} pull {model}`", detail
    return True, None, detail


def ollama_model_ok(ollama_bin: str, model: str, timeout: float) -> Tuple[bool, bool, Optional[str], dict[str, Any]]:
    if not shutil.which(ollama_bin) and not os.access(ollama_bin, os.X_OK):
        return False, True, f"`{ollama_bin}` not found; install Ollama and ensure the binary is on PATH", {}

    command = [ollama_bin, "show", model]
    completed = run_cmd(command, timeout=timeout)
    detail = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode == 0:
        return True, False, None, detail

    combined = f"{completed.stdout}\n{completed.stderr}".lower()
    if "connection refused" in combined or "could not connect" in combined or "dial tcp" in combined:
        return False, True, "Ollama is installed but the daemon is unavailable; start it with `ollama serve`", detail
    if "not found" in combined or "pull model" in combined:
        return False, True, f"Ollama model `{model}` is not installed; run `{ollama_bin} pull {model}`", detail
    return False, False, f"`{' '.join(command)}` failed; inspect report diagnostics", detail


def environment(krillm_bin: Optional[str], ollama_bin: str) -> dict[str, Any]:
    def output_or_none(command: list[str]) -> Optional[str]:
        try:
            completed = run_cmd(command, timeout=15.0)
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        text = (completed.stdout or completed.stderr).strip()
        return text or None

    chip = output_or_none(["sysctl", "-n", "machdep.cpu.brand_string"]) if platform.system() == "Darwin" else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "chip": chip,
        "python": sys.version.split()[0],
        "swift": output_or_none(["swift", "--version"]),
        "krillm_version": output_or_none([krillm_bin, "version"]) if krillm_bin else None,
        "ollama_version": output_or_none([ollama_bin, "--version"]),
        "git_commit": output_or_none(["git", "rev-parse", "HEAD"]),
        "git_status": output_or_none(["git", "status", "--short"]),
    }


def krillm_command(args: argparse.Namespace, krillm_bin: str, prompt: str) -> list[str]:
    cmd = [
        krillm_bin,
        "run",
        args.krill_model,
        prompt,
        "--temp",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-tokens",
        str(args.max_tokens),
        "--seed",
        str(args.seed),
    ]
    if args.krillm_draft_model:
        cmd += ["--draft-model", args.krillm_draft_model]
    return cmd


SPEC_RE = re.compile(
    r"spec:\s+rounds=(?P<rounds>\d+),\s+accepted=(?P<accepted>\d+),"
    r"\s+final_k=(?P<final_k>\d+),\s+acceptance=(?P<acceptance>[0-9.]+)"
)


def parse_krillm_result(stdout: str, wall_s: float, command: list[str]) -> dict[str, Any]:
    match = STATS_RE.search(stdout)
    if not match:
        raise RuntimeError("KrillLM output did not contain a parseable stats line")
    generated = stdout.split("\n---\n", 1)[0]
    # Strip operational lines that precede the generated text. `spec:` is
    # belt-and-braces: today it lives in the stats section after `---`
    # and never reaches `generated`, but stripping it here means a future
    # CLI reformat (or the line landing before `---`) cannot cause
    # output_sha256 to diverge silently between spec-on and spec-off
    # parity comparisons.
    generated_lines = [
        line for line in generated.splitlines()
        if line and not line.startswith("Loading model")
        and not line.startswith("Ready (")
        and not line.startswith("Speculative decoding enabled")
        and not line.startswith("spec:")
    ]
    generated_text = "\n".join(generated_lines)
    result: dict[str, Any] = {
        "command": command,
        "prompt_tokens": int(match.group("prompt_tokens")),
        "generated_tokens": int(match.group("generated_tokens")),
        "prefill_tokens_per_second": float(match.group("prefill_tps")),
        "decode_tokens_per_second": float(match.group("decode_tps")),
        "ttft_ms": float(match.group("ttft_ms")),
        "total_s": float(match.group("total_s")),
        "wall_time_s": wall_s,
        "output_sha256": sha256_text(generated_text),
        "output_preview": generated_text[:200],
    }
    spec_match = SPEC_RE.search(stdout)
    if spec_match:
        result["speculative"] = {
            "rounds": int(spec_match.group("rounds")),
            "accepted_tokens": int(spec_match.group("accepted")),
            "final_k": int(spec_match.group("final_k")),
            "acceptance_rate": float(spec_match.group("acceptance")),
        }
    return result


def run_krillm(args: argparse.Namespace, krillm_bin: str, prompt: str, measured: bool) -> Optional[dict[str, Any]]:
    command = krillm_command(args, krillm_bin, prompt)
    start = time.perf_counter()
    completed = run_cmd(command, timeout=args.timeout)
    wall_s = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"KrillLM command failed with exit {completed.returncode}: {completed.stderr or completed.stdout}"
        )
    if not measured:
        return None
    return parse_krillm_result(completed.stdout, wall_s, command)


def ollama_payload(args: argparse.Namespace, prompt: str, stream: bool) -> dict[str, Any]:
    return {
        "model": args.ollama_model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "num_predict": args.max_tokens,
        },
    }


def run_ollama(args: argparse.Namespace, prompt: str, measured: bool) -> Optional[dict[str, Any]]:
    payload = ollama_payload(args, prompt, stream=measured)
    data = json.dumps(payload).encode("utf-8")
    url = args.ollama_host.rstrip("/") + "/api/generate"
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            if not measured:
                response.read()
                return None
            first_token_s: Optional[float] = None
            chunks: list[str] = []
            final: Optional[dict[str, Any]] = None
            for raw_line in response:
                if not raw_line.strip():
                    continue
                event = json.loads(raw_line)
                text = event.get("response") or ""
                if text and first_token_s is None:
                    first_token_s = time.perf_counter() - start
                chunks.append(text)
                if event.get("done"):
                    final = event
            wall_s = time.perf_counter() - start
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama API request failed: {exc}") from exc

    if final is None:
        raise RuntimeError("Ollama stream ended without final stats")

    prompt_eval_count = int(final.get("prompt_eval_count") or 0)
    eval_count = int(final.get("eval_count") or 0)
    prompt_eval_duration_s = int(final.get("prompt_eval_duration") or 0) / 1_000_000_000
    eval_duration_s = int(final.get("eval_duration") or 0) / 1_000_000_000
    generated_text = "".join(chunks)
    return {
        "api": url,
        "payload": payload,
        "prompt_tokens": prompt_eval_count,
        "generated_tokens": eval_count,
        "prefill_tokens_per_second": prompt_eval_count / prompt_eval_duration_s if prompt_eval_duration_s > 0 else None,
        "decode_tokens_per_second": eval_count / eval_duration_s if eval_duration_s > 0 else None,
        "ttft_ms_wall": first_token_s * 1000 if first_token_s is not None else None,
        "prompt_eval_ms": prompt_eval_duration_s * 1000,
        "total_s": int(final.get("total_duration") or 0) / 1_000_000_000,
        "wall_time_s": wall_s,
        "load_ms": int(final.get("load_duration") or 0) / 1_000_000,
        "output_sha256": sha256_text(generated_text),
        "output_preview": generated_text[:200],
    }


def krillm_server_payload(args: argparse.Namespace, prompt: str, stream: bool) -> dict[str, Any]:
    return {
        "model": args.krill_model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "num_predict": args.max_tokens,
        },
    }


def run_krillm_server(args: argparse.Namespace, prompt: str, measured: bool) -> Optional[dict[str, Any]]:
    """Run a benchmark request against a persistent KrillLM server."""
    payload = krillm_server_payload(args, prompt, stream=measured)
    data = json.dumps(payload).encode("utf-8")
    url = args.krillm_url.rstrip("/") + "/api/generate"
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            if not measured:
                response.read()
                return None
            first_token_s: Optional[float] = None
            chunks: list[str] = []
            final: Optional[dict[str, Any]] = None
            for raw_line in response:
                if not raw_line.strip():
                    continue
                event = json.loads(raw_line)
                text = event.get("response") or ""
                if text and first_token_s is None:
                    first_token_s = time.perf_counter() - start
                chunks.append(text)
                if event.get("done"):
                    final = event
            wall_s = time.perf_counter() - start
    except urllib.error.URLError as exc:
        raise RuntimeError(f"KrillLM server request failed: {exc}") from exc

    if final is None:
        raise RuntimeError("KrillLM server stream ended without final stats")

    prompt_eval_count = int(final.get("prompt_eval_count") or 0)
    eval_count = int(final.get("eval_count") or 0)
    prompt_eval_duration_s = int(final.get("prompt_eval_duration") or 0) / 1_000_000_000
    eval_duration_s = int(final.get("eval_duration") or 0) / 1_000_000_000
    generated_text = "".join(chunks)
    result: dict[str, Any] = {
        "api": url,
        "payload": payload,
        "prompt_tokens": prompt_eval_count,
        "generated_tokens": eval_count,
        "prefill_tokens_per_second": prompt_eval_count / prompt_eval_duration_s if prompt_eval_duration_s > 0 else None,
        "decode_tokens_per_second": eval_count / eval_duration_s if eval_duration_s > 0 else None,
        "ttft_ms_wall": first_token_s * 1000 if first_token_s is not None else None,
        "prompt_eval_ms": prompt_eval_duration_s * 1000,
        "total_s": int(final.get("total_duration") or 0) / 1_000_000_000,
        "wall_time_s": wall_s,
        "output_sha256": sha256_text(generated_text),
        "output_preview": generated_text[:200],
        "mode": "server",
    }
    # Include server-side TTFT if available
    ttft_ns = final.get("ttft_ns")
    if ttft_ns is not None:
        result["ttft_ms_server"] = int(ttft_ns) / 1_000_000
    return result


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "prompt_tokens",
        "generated_tokens",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "ttft_ms",
        "ttft_ms_wall",
        "prompt_eval_ms",
        "total_s",
        "wall_time_s",
        "load_ms",
    ]
    summary: dict[str, Any] = {"runs": len(runs)}
    for key in keys:
        values = [run[key] for run in runs if run.get(key) is not None]
        if values:
            summary[f"{key}_median"] = statistics.median(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)
    return summary


def infer_cache_mode(index: int, warmup: int, use_server: bool) -> str:
    """Infer per-run cache_mode label.

    cold      = first measured run when no warmup was performed
    warm      = run executed after at least one warmup or after the first run
    cache_hit = same prompt repeated against a server with prefix cache active
    """
    if use_server and (warmup > 0 or index > 0):
        return "cache_hit"
    if warmup == 0 and index == 0:
        return "cold"
    return "warm"


def apply_cache_mode(
    runs: list[Optional[dict[str, Any]]],
    cli_mode: str,
    warmup: int,
    use_server: bool,
) -> tuple[str, str]:
    """Tag each run with cache_mode and return (group_label, source)."""
    source = "explicit" if cli_mode != "auto" else "auto"
    labels: list[str] = []
    for i, run in enumerate(runs):
        if run is None:
            continue
        if cli_mode == "auto":
            label = infer_cache_mode(i, warmup, use_server)
        else:
            label = cli_mode
        run["cache_mode"] = label
        labels.append(label)
    if not labels:
        group = cli_mode if cli_mode != "auto" else "warm"
    elif all(l == labels[0] for l in labels):
        group = labels[0]
    else:
        group = "mixed"
    return group, source


def check_input_parity(
    krillm_runs: list[dict[str, Any]],
    ollama_runs: list[dict[str, Any]],
    prompt: str,
) -> dict[str, Any]:
    """Compare prompt token (or character) counts between engines."""
    def median_tokens(runs: list[dict[str, Any]]) -> Optional[float]:
        values = [r["prompt_tokens"] for r in runs if r and r.get("prompt_tokens")]
        if not values:
            return None
        return statistics.median(values)

    krill_tokens = median_tokens(krillm_runs)
    ollama_tokens = median_tokens(ollama_runs)
    if krill_tokens and ollama_tokens:
        basis = "prompt_tokens"
        a, b = krill_tokens, ollama_tokens
    else:
        basis = "prompt_chars"
        a = b = float(len(prompt))
    if max(a, b) == 0:
        return {"status": "ok", "basis": basis, "krillm": a, "ollama": b, "delta_ratio": 0.0}
    delta_ratio = abs(a - b) / max(a, b)
    status = "ok" if delta_ratio <= 0.10 else "mismatch"
    details = {
        "status": status,
        "basis": basis,
        "krillm": a,
        "ollama": b,
        "delta_ratio": round(delta_ratio, 4),
    }
    if status == "mismatch":
        details["details"] = (
            f"prompt size differs by {delta_ratio*100:.1f}% between engines "
            f"({basis}: krillm={a}, ollama={b}); results may not be comparable"
        )
    return details


def write_report(path: str, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        print("ERROR: --runs must be >= 1", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("ERROR: --warmup must be >= 0", file=sys.stderr)
        return 2
    if args.max_tokens < 1:
        print("ERROR: --max-tokens must be >= 1", file=sys.stderr)
        return 2

    prompt = prompt_text(args)
    use_server = bool(args.krillm_url)
    krillm_bin = None if use_server else find_krillm_binary(args.krillm_bin)
    report: dict[str, Any] = {
        "status": "pending",
        "benchmark": {
            "krill_model": args.krill_model,
            "ollama_model": args.ollama_model,
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "max_tokens": args.max_tokens,
            "runs": args.runs,
            "warmup": args.warmup,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "cache_mode_requested": args.cache_mode,
        },
        "environment": environment(krillm_bin, args.ollama_bin),
        "preflight": {},
        "results": {},
    }

    if use_server:
        report["benchmark"]["krillm_url"] = args.krillm_url
        report["benchmark"]["mode"] = "server"

    skips: list[str] = []
    failures: list[str] = []

    if use_server:
        # Server mode: verify KrillLM server is reachable
        health_url = args.krillm_url.rstrip("/") + "/healthz"
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                health = json.loads(resp.read())
            report["preflight"]["krillm"] = {
                "ok": True,
                "mode": "server",
                "url": args.krillm_url,
                "health": health,
            }
        except Exception as exc:
            skips.append(f"KrillLM server not reachable at {args.krillm_url}: {exc}")
            report["preflight"]["krillm"] = {
                "ok": False,
                "mode": "server",
                "url": args.krillm_url,
                "error": str(exc),
            }
    else:
        if krillm_bin is None:
            skips.append("KrillLM binary not found; run `make release` or pass `--krillm-bin`")
        else:
            ok, reason, detail = krillm_model_ok(krillm_bin, args.krill_model)
            report["preflight"]["krillm"] = {"ok": ok, "binary": krillm_bin, "detail": detail}
            if not ok and reason:
                skips.append(reason)

    ok, is_skip, reason, detail = ollama_model_ok(args.ollama_bin, args.ollama_model, args.timeout)
    report["preflight"]["ollama"] = {"ok": ok, "binary": args.ollama_bin, "detail": detail}
    if not ok and reason:
        if is_skip:
            skips.append(reason)
        else:
            failures.append(reason)

    if failures or skips:
        report["status"] = "failed" if failures else "skipped"
        report["failures"] = failures
        report["skips"] = skips
        write_report(args.output, report)
        for message in failures:
            print(f"FAIL: {message}", file=sys.stderr)
        for message in skips:
            print(f"SKIP: {message}", file=sys.stderr)
        print(f"Report: {args.output}")
        return 1 if failures else SKIP_EXIT_CODE

    try:
        if use_server:
            # Server mode: warm server is already running, just do warmup requests
            for _ in range(args.warmup):
                run_krillm_server(args, prompt, measured=False)
                run_ollama(args, prompt, measured=False)

            krillm_runs = [run_krillm_server(args, prompt, measured=True) for _ in range(args.runs)]
        else:
            assert krillm_bin is not None
            for _ in range(args.warmup):
                run_krillm(args, krillm_bin, prompt, measured=False)
                run_ollama(args, prompt, measured=False)

            krillm_runs = [run_krillm(args, krillm_bin, prompt, measured=True) for _ in range(args.runs)]

        ollama_runs = [run_ollama(args, prompt, measured=True) for _ in range(args.runs)]
    except Exception as exc:
        report["status"] = "failed"
        report["failures"] = [str(exc)]
        write_report(args.output, report)
        print(f"FAIL: {exc}", file=sys.stderr)
        print(f"Report: {args.output}")
        return 1

    report["status"] = "ok"

    krill_group, krill_source = apply_cache_mode(krillm_runs, args.cache_mode, args.warmup, use_server)
    ollama_group, ollama_source = apply_cache_mode(ollama_runs, args.cache_mode, args.warmup, False)

    report["results"] = {
        "krillm": {
            "runs": krillm_runs,
            "summary": summarize([run for run in krillm_runs if run]),
            "cache_mode": krill_group,
            "cache_mode_source": krill_source,
        },
        "ollama": {
            "runs": ollama_runs,
            "summary": summarize([run for run in ollama_runs if run]),
            "cache_mode": ollama_group,
            "cache_mode_source": ollama_source,
        },
    }

    if args.cache_mode != "auto":
        top_cache_mode = args.cache_mode
        top_source = "explicit"
    elif krill_group == ollama_group:
        top_cache_mode = krill_group
        top_source = "auto"
    else:
        top_cache_mode = "mixed"
        top_source = "auto"
    report["cache_mode"] = top_cache_mode
    report["cache_mode_source"] = top_source

    parity = check_input_parity(
        [r for r in krillm_runs if r],
        [r for r in ollama_runs if r],
        prompt,
    )
    report["input_parity"] = parity
    if parity.get("status") == "mismatch":
        print(f"WARN: input parity mismatch: {parity.get('details')}", file=sys.stderr)

    if use_server:
        report["server_media"] = {
            "status": "supported",
            "image": "native",
            "audio": "bridge",
        }

    # Detect prefix cache influence: if KrillLM prefill speed varies >3x between
    # runs, later runs likely hit the prefix cache.
    if use_server:
        valid_krillm = [r for r in krillm_runs if r and r.get("prefill_tokens_per_second")]
        if len(valid_krillm) >= 2:
            prefills = [r["prefill_tokens_per_second"] for r in valid_krillm]
            if max(prefills) > 3 * min(prefills):
                report["cache_warning"] = (
                    "KrillLM prefill speed varies >3x across runs, indicating prefix cache hits "
                    "on repeated prompts. Later runs may show near-zero prefill cost. "
                    "Use distinct prompts or --warmup 0 to measure cold prefill."
                )
        report["benchmark"]["prefix_cache_active"] = True
        report["benchmark"]["note"] = (
            "Server mode with repeated prompts: warmup requests populate the prefix cache, "
            "so measured runs may reflect cached prefill (near-zero TTFT). This is the "
            "expected production behavior for repeated system prompts."
        )

    write_report(args.output, report)
    print(f"Benchmark report: {args.output}")
    print(json.dumps(report["results"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
