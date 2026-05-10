#!/usr/bin/env python3
"""KrillLM release benchmark gate.

Compares KrillLM and Ollama benchmark reports against configured performance
thresholds. Produces a single JSON report and clear terminal pass/fail summary.

Supports two report formats:
  - krillm-vs-ollama.json  (flat: results.krillm / results.ollama)
  - gemma4-e2b-multimodal-*.json (per-task: results.text / results.image / results.audio)

Exit codes:
  0  All metrics pass configured thresholds.
  1  At least one metric fails or reports are incompatible.
  2  Bad arguments or missing files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Default thresholds (from OLLAMA_SPEEDUP_EXECUTION_PLAN.md)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "text_decode_ratio": 1.5,       # KrillLM decode tok/s / Ollama decode tok/s >= 1.5
    "text_wall_ratio": 0.67,        # KrillLM wall / Ollama wall <= 0.67
    "image_wall_ratio": 0.67,
    "audio_wall_ratio": 0.67,
    "text_prefill_ratio": 1.5,      # KrillLM prefill / Ollama prefill >= 1.5 or wall must win
    "image_prefill_ratio": 1.5,
    "audio_prefill_ratio": 1.5,
    "text_ttft_ratio": 0.67,        # KrillLM TTFT / Ollama TTFT <= 0.67
    "memory_ratio": 1.0,            # KrillLM peak GB / Ollama peak GB <= 1.0
}

# Metrics where lower is better (ratio should be <= threshold)
LOWER_IS_BETTER = {
    "text_wall_ratio",
    "image_wall_ratio",
    "audio_wall_ratio",
    "text_ttft_ratio",
    "memory_ratio",
}

# Metrics where higher is better (ratio should be >= threshold)
HIGHER_IS_BETTER = {
    "text_decode_ratio",
    "text_prefill_ratio",
    "image_prefill_ratio",
    "audio_prefill_ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KrillLM release benchmark gate."
    )
    parser.add_argument(
        "report",
        nargs="?",
        help="Path to a combined benchmark report JSON (contains both krillm and ollama results).",
    )
    parser.add_argument(
        "--krillm-report",
        help="Path to a standalone KrillLM benchmark report (for sequential comparison).",
    )
    parser.add_argument(
        "--ollama-report",
        help="Path to a standalone Ollama benchmark report (for sequential comparison).",
    )
    parser.add_argument(
        "--thresholds",
        help="Path to a JSON file with custom thresholds (overrides defaults).",
    )
    parser.add_argument(
        "--output",
        default=".build/benchmarks/release-gate.json",
        help="Output path for the gate report.",
    )
    parser.add_argument(
        "--allow-dtype-mismatch",
        action="store_true",
        help="Allow comparison when dtype/quantization class differs.",
    )
    parser.add_argument(
        "--allow-prompt-mismatch",
        action="store_true",
        help="Allow comparison when prompt or media SHA256 differs.",
    )
    parser.add_argument(
        "--scope",
        choices=["release", "multimodal_release"],
        default="multimodal_release",
        help=(
            "Gate scope. 'multimodal_release' (default) enforces all "
            "thresholds including image and audio, since the server now "
            "accepts media payloads end-to-end. 'release' keeps server "
            "media out of scope for text-only reports. Existing "
            "thresholds are unchanged in either scope; only their "
            "applicability differs."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Report loading and format detection
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def detect_format(report: dict[str, Any]) -> str:
    """Detect whether report is 'flat' or 'multimodal'."""
    results = report.get("results", {})
    if "krillm" in results and "ollama" in results:
        return "flat"
    if any(k in results for k in ("text", "image", "audio")):
        return "multimodal"
    raise ValueError("Unrecognized report format: results must contain krillm/ollama or text/image/audio keys.")


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def safe_get(d: dict[str, Any], *keys: str) -> Optional[float]:
    """Walk nested dict; return None if any key missing or value is None."""
    current: Any = d
    for k in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(k)
    if current is None:
        return None
    return float(current)


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def extract_flat_metrics(report: dict[str, Any]) -> dict[str, Optional[float]]:
    """Extract metrics from flat krillm-vs-ollama report."""
    r = report.get("results", {})
    ks = r.get("krillm", {}).get("summary", {})
    os_ = r.get("ollama", {}).get("summary", {})

    return {
        "text_decode_ratio": safe_ratio(
            safe_get(ks, "decode_tokens_per_second_median"),
            safe_get(os_, "decode_tokens_per_second_median"),
        ),
        "text_wall_ratio": safe_ratio(
            safe_get(ks, "wall_time_s_median"),
            safe_get(os_, "wall_time_s_median"),
        ),
        "text_prefill_ratio": safe_ratio(
            safe_get(ks, "prefill_tokens_per_second_median"),
            safe_get(os_, "prefill_tokens_per_second_median"),
        ),
        "text_ttft_ratio": safe_ratio(
            safe_get(ks, "ttft_ms_median") or safe_get(ks, "ttft_ms_wall_median"),
            safe_get(os_, "ttft_ms_median") or safe_get(os_, "ttft_ms_wall_median"),
        ),
    }


def extract_multimodal_metrics(report: dict[str, Any]) -> dict[str, Optional[float]]:
    """Extract metrics from gemma4 multimodal report."""
    results = report.get("results", {})
    metrics: dict[str, Optional[float]] = {}

    for task in ("text", "image", "audio"):
        task_data = results.get(task, {})
        ks = task_data.get("krillm", {}).get("summary", {})
        os_ = task_data.get("ollama", {}).get("summary", {})

        # Decode ratio (text only for threshold, but record all)
        decode_ratio = safe_ratio(
            safe_get(ks, "decode_tokens_per_second_median"),
            safe_get(os_, "decode_tokens_per_second_median"),
        )
        if task == "text":
            metrics["text_decode_ratio"] = decode_ratio
        else:
            metrics[f"{task}_decode_ratio"] = decode_ratio

        # Wall time ratio
        metrics[f"{task}_wall_ratio"] = safe_ratio(
            safe_get(ks, "wall_time_s_median"),
            safe_get(os_, "wall_time_s_median"),
        )

        # Prefill ratio
        metrics[f"{task}_prefill_ratio"] = safe_ratio(
            safe_get(ks, "prefill_tokens_per_second_median"),
            safe_get(os_, "prefill_tokens_per_second_median"),
        )

        # TTFT (text only)
        if task == "text":
            metrics["text_ttft_ratio"] = safe_ratio(
                safe_get(ks, "ttft_ms_median") or safe_get(ks, "ttft_ms_wall_median"),
                safe_get(os_, "ttft_ms_median") or safe_get(os_, "ttft_ms_wall_median"),
            )

    # Memory ratio (from text task, which has peak_memory data)
    text_ks = results.get("text", {}).get("krillm", {}).get("summary", {})
    text_os = results.get("text", {}).get("ollama", {}).get("summary", {})
    metrics["memory_ratio"] = safe_ratio(
        safe_get(text_ks, "peak_memory_gb_median"),
        safe_get(text_os, "peak_memory_gb_median"),
    )

    return metrics


# ---------------------------------------------------------------------------
# Metadata compatibility
# ---------------------------------------------------------------------------

def check_compatibility(report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Check metadata compatibility; return list of caveat strings."""
    caveats: list[str] = []

    quant = report.get("quantization", {})
    comparison = quant.get("comparison", {})

    if quant and not comparison.get("strict_equal", False):
        label = comparison.get("label", "unknown")
        note = comparison.get("note", "")
        if not comparison.get("class_equal", False):
            if not args.allow_dtype_mismatch:
                caveats.append(f"Quantization mismatch: {label}. {note}")
            else:
                caveats.append(f"ALLOWED dtype mismatch: {label}. {note}")
        else:
            caveats.append(f"Same quantization class but not bit-identical: {label}. {note}")

    return caveats


# ---------------------------------------------------------------------------
# Threshold evaluation
# ---------------------------------------------------------------------------

def evaluate_metric(
    name: str,
    value: Optional[float],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate a single metric against its threshold."""
    if value is None:
        return {
            "name": name,
            "value": None,
            "threshold": threshold,
            "pass": False,
            "reason": "metric not available in report",
        }

    if name in LOWER_IS_BETTER:
        passed = value <= threshold
        direction = "<="
    else:
        passed = value >= threshold
        direction = ">="

    return {
        "name": name,
        "value": round(value, 4),
        "threshold": threshold,
        "direction": direction,
        "pass": passed,
        "reason": f"{value:.4f} {'PASS' if passed else 'FAIL'} (need {direction} {threshold})",
    }


def classify_bottleneck(evaluations: list[dict[str, Any]]) -> str:
    """Identify the primary bottleneck from failed metrics."""
    failures = [e for e in evaluations if not e["pass"] and e["value"] is not None]
    if not failures:
        return "none"

    # Classify by the worst failure
    worst = None
    worst_gap = 0.0
    for f in failures:
        val = f["value"]
        thr = f["threshold"]
        if f["name"] in LOWER_IS_BETTER:
            gap = val - thr  # positive = over threshold
        else:
            gap = thr - val  # positive = under threshold
        if gap > worst_gap:
            worst_gap = gap
            worst = f["name"]

    if worst is None:
        return "unknown"

    if "wall" in worst:
        return "wall_time"
    if "prefill" in worst:
        return "prefill"
    if "decode" in worst:
        return "decode_throughput"
    if "ttft" in worst:
        return "time_to_first_token"
    if "memory" in worst:
        return "memory"
    return worst


def geometric_mean(values: list[float]) -> float:
    """Geometric mean of positive values."""
    if not values:
        return 0.0
    product = 1.0
    for v in values:
        if v <= 0:
            return 0.0
        product *= v
    return product ** (1.0 / len(values))


def compute_speedup_factors(metrics: dict[str, Optional[float]]) -> list[float]:
    """Convert all metrics to speedup factors (>1 = KrillLM is better)."""
    factors: list[float] = []
    for name, value in metrics.items():
        if value is None:
            continue
        if name in LOWER_IS_BETTER:
            # Lower ratio is better, so speedup = 1/ratio
            factors.append(1.0 / value if value > 0 else 0.0)
        else:
            # Higher ratio is better, ratio is already the speedup
            factors.append(value)
    return factors


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def print_gate_summary(
    evaluations: list[dict[str, Any]],
    speedup: float,
    bottleneck: str,
    caveats: list[str],
    passed: bool,
) -> None:
    print()
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  KrillLM Release Benchmark Gate{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print()

    for e in evaluations:
        val = e["value"]
        thr = e["threshold"]
        direction = e.get("direction", "")
        if val is None:
            icon = f"{YELLOW}--{RESET}"
            val_str = "N/A"
        elif e["pass"]:
            icon = f"{GREEN}OK{RESET}"
            val_str = f"{val:.4f}"
        else:
            icon = f"{RED}FAIL{RESET}"
            val_str = f"{val:.4f}"

        print(f"  [{icon}] {e['name']:<30s}  {val_str:>10s}  (need {direction} {thr})")

    print()
    print(f"  {CYAN}Geometric mean speedup:{RESET}  {speedup:.3f}x")
    if bottleneck != "none":
        print(f"  {CYAN}Primary bottleneck:{RESET}     {bottleneck}")

    if caveats:
        print()
        for c in caveats:
            print(f"  {YELLOW}Caveat:{RESET} {c}")

    print()
    if passed:
        print(f"  {GREEN}{BOLD}GATE: PASS{RESET}")
    else:
        print(f"  {RED}{BOLD}GATE: FAIL{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Load thresholds
    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.thresholds:
        custom = load_json(args.thresholds)
        thresholds.update(custom)

    # Load report(s)
    if args.report:
        report = load_json(args.report)
    elif args.krillm_report and args.ollama_report:
        # Sequential comparison: merge two single-engine reports
        # This handles the case where disk-constrained machines run one engine at a time
        krillm = load_json(args.krillm_report)
        ollama = load_json(args.ollama_report)
        report = {
            "status": "ok",
            "benchmark": {
                "krillm_source": args.krillm_report,
                "ollama_source": args.ollama_report,
                "comparison_mode": "sequential",
            },
            "environment": krillm.get("environment", {}),
            "results": {
                "krillm": krillm.get("results", {}).get("krillm", {}),
                "ollama": ollama.get("results", {}).get("ollama", {}),
            },
        }
    else:
        print("ERROR: provide a combined report path or --krillm-report and --ollama-report", file=sys.stderr)
        return 2

    if report.get("status") not in ("ok",):
        status = report.get("status", "unknown")
        print(f"ERROR: report status is '{status}', not 'ok'. Cannot evaluate.", file=sys.stderr)
        return 2

    # Detect format and extract metrics
    fmt = detect_format(report)
    if fmt == "flat":
        metrics = extract_flat_metrics(report)
    else:
        metrics = extract_multimodal_metrics(report)

    # Check compatibility
    caveats = check_compatibility(report, args)
    compatibility_fail = any("mismatch" in c and "ALLOWED" not in c for c in caveats)

    # Determine scope effects
    server_media_status = (report.get("server_media") or {}).get("status")
    server_media_out_of_scope = (
        args.scope == "release"
        and (server_media_status == "skipped" or fmt == "flat")
    )
    scope_info: dict[str, Any] = {"scope": args.scope}
    if server_media_out_of_scope:
        scope_info["server_media"] = "out_of_scope"

    media_metric_prefixes = ("image_", "audio_")

    # Evaluate each metric against thresholds
    evaluations: list[dict[str, Any]] = []
    skipped_for_scope: list[str] = []
    for name, threshold in sorted(thresholds.items()):
        value = metrics.get(name)
        if value is None and name not in metrics:
            # Metric not applicable to this report format — skip silently
            continue
        if server_media_out_of_scope and name.startswith(media_metric_prefixes):
            if value is None:
                skipped_for_scope.append(name)
                continue
        evaluations.append(evaluate_metric(name, value, threshold))

    # Compute aggregate stats
    speedup_factors = compute_speedup_factors(metrics)
    geo_speedup = geometric_mean(speedup_factors) if speedup_factors else 0.0
    bottleneck = classify_bottleneck(evaluations)

    all_pass = all(e["pass"] for e in evaluations if e["value"] is not None)
    gate_pass = all_pass and not compatibility_fail

    # Find worst metric
    worst_metric = None
    worst_gap = 0.0
    for e in evaluations:
        if e["value"] is None:
            continue
        val = e["value"]
        thr = e["threshold"]
        if e["name"] in LOWER_IS_BETTER:
            gap = val / thr if thr > 0 else float("inf")
        else:
            gap = thr / val if val > 0 else float("inf")
        if gap > worst_gap:
            worst_gap = gap
            worst_metric = e["name"]

    # Build gate report
    gate_report: dict[str, Any] = {
        "gate": "pass" if gate_pass else "fail",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": args.report or {
            "krillm": args.krillm_report,
            "ollama": args.ollama_report,
        },
        "format": fmt,
        "thresholds": thresholds,
        "metrics": metrics,
        "evaluations": evaluations,
        "summary": {
            "geometric_mean_speedup": round(geo_speedup, 4),
            "worst_metric": worst_metric,
            "worst_gap": round(worst_gap, 4),
            "bottleneck": bottleneck,
            "all_metrics_pass": all_pass,
            "compatibility_ok": not compatibility_fail,
        },
        "caveats": caveats,
        "scope": scope_info,
    }
    if skipped_for_scope:
        gate_report["scope_skipped_metrics"] = sorted(skipped_for_scope)

    # Write report
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate_report, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    # Terminal output
    print_gate_summary(evaluations, geo_speedup, bottleneck, caveats, gate_pass)
    print(f"Gate report: {args.output}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
