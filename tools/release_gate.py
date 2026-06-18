#!/usr/bin/env python3
"""Krill release benchmark gate.

Compares Krill and Ollama benchmark reports against configured performance
thresholds. Produces a single JSON report and clear terminal pass/fail summary.

Supports two report formats:
  - krill-vs-ollama.json  (flat: results.krill / results.ollama)
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
    "text_decode_ratio": 1.5,       # Krill decode tok/s / Ollama decode tok/s >= 1.5
    "text_wall_ratio": 0.67,        # Krill wall / Ollama wall <= 0.67
    "image_wall_ratio": 0.67,
    "audio_wall_ratio": 0.67,
    "text_prefill_ratio": 1.5,      # Krill prefill / Ollama prefill >= 1.5 or wall must win
    "image_prefill_ratio": 1.5,
    "audio_prefill_ratio": 1.5,
    "text_ttft_ratio": 0.67,        # Krill TTFT / Ollama TTFT <= 0.67
    "memory_ratio": 1.0,            # Krill peak GB / Ollama peak GB <= 1.0
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
    "text_decode_ratio_floor",
    "text_prefill_ratio",
    "image_prefill_ratio",
    "audio_prefill_ratio",
}

# Hard non-regression floors applied to a metric that a profile demoted to
# "advisory" but which must still never regress. When `text_decode_ratio`
# is advisory, a synthetic `<metric>_floor` HARD evaluation is appended so
# the gate still hard-fails if Krill ever decodes SLOWER than Ollama
# (ratio < 1.0). The advisory >= 1.5 target is still evaluated and printed;
# it just no longer blocks. The floor applies in EVERY profile that demotes
# the metric to advisory - keyed on the metric kind, not the profile name.
# See OLLAMA_SPEEDUP_EXECUTION_PLAN.md §4 for the objective re-promotion
# contract.
ADVISORY_HARD_FLOORS = {
    "text_decode_ratio": 1.0,
}

# Owner-acceptance provenance for the `text_decode_ratio` advisory demotion,
# per profile. Each demotion is a recorded owner decision; the gate caveat
# cites the one that applies so the relaxation is never silent.
DECODE_ADVISORY_PROVENANCE = {
    "release_candidate": "owner-accepted 2026-05-16; docs/RELEASE_GATE_DECODE_PROPOSAL.md",
    "strict": "owner-accepted 2026-05-22; docs/RELEASE_GATE_STRICT_DECODE_PROPOSAL.md",
}

# Owner-acceptance provenance for advisory demotions that, unlike decode, carry
# NO `<metric>_floor` because the metric is structurally < 1.0 by design — a
# non-regression floor would be meaningless. Keyed on (profile, metric). The
# gate still emits a caveat so the relaxation is never silent. Today the only
# entry is `image_prefill_ratio` under `strict`: the vision-encoder cache lifts
# SigLIP2 forward + projector cost out of the measured prefill window, so the
# prefill-TPS bucket understates a user win that the HARD `image_wall_ratio`
# already captures. See docs/RELEASE_GATE_IMAGE_PREFILL_PROPOSAL.md.
ADVISORY_DEMOTION_PROVENANCE = {
    ("strict", "image_prefill_ratio"):
        "owner-accepted 2026-05-22; docs/RELEASE_GATE_IMAGE_PREFILL_PROPOSAL.md",
}

# ---------------------------------------------------------------------------
# Gate profiles
# ---------------------------------------------------------------------------
#
# Each profile maps every gated metric to one of three kinds:
#
#   "hard"          — failure breaks the gate (exit 1).
#   "advisory"      — evaluated and reported; failure does NOT break the gate.
#   "out_of_scope"  — skipped entirely. Listed in `scope_skipped_metrics` with
#                     the profile-level reason so the omission is auditable.
#
# `strict` (the default) hard-gates every threshold EXCEPT two metrics proven
# to be mismeasured or unreachable, each demoted to advisory by a recorded
# owner decision:
#   - `text_decode_ratio`: advisory with a hard >= 1.0 non-regression floor -
#     the same treatment `release_candidate` uses, extended to `strict` on
#     2026-05-22 (owner-accepted; docs/RELEASE_GATE_STRICT_DECODE_PROPOSAL.md)
#     because the >= 1.5x decode-ratio target is structurally unreachable on
#     M-series with the draft models available in mlx-community: the
#     spec-decode throughput ratio asymptotes near 1.10x even at infinite K
#     and 100% acceptance (see docs/SPECULATIVE_DECODING.md).
#   - `image_prefill_ratio`: advisory with NO floor (owner-accepted 2026-05-22;
#     docs/RELEASE_GATE_IMAGE_PREFILL_PROPOSAL.md). The vision-encoder cache
#     lifts SigLIP2 forward + projector cost out of the measured prefill
#     window, so this prefill-TPS bucket understates real performance. The
#     HARD `image_wall_ratio` carries the user-visible image guarantee; a
#     >= 1.0 floor here would be meaningless since the metric is structurally
#     < 1.0 by design. Re-promotes to hard once the benchmark separates
#     vision-encoder time from language-model prefill time.
# Every other metric stays hard under strict, so it remains the uncompromised
# reference for everything except the two metrics proven mismeasured.
#
# `release_candidate` is the profile defined in
# OLLAMA_SPEEDUP_EXECUTION_PLAN.md §4 ("Release Gate Semantics"). It hard-gates
# the metrics that map directly to user-visible latency or memory and treats
# the prefill TPS bucket as advisory because:
#   - text_prefill_ratio: text_wall and text_ttft are the user-visible signals
#     and are already hard-gated; prefill TPS is noisy on short prompts.
#   - image_prefill_ratio: the vision encoder cache lifts work out of the
#     measured prefill window, so this bucket understates the user win that
#     image_wall already captures. Advisory under BOTH profiles since
#     2026-05-22 (owner-accepted; docs/RELEASE_GATE_IMAGE_PREFILL_PROPOSAL.md).
#     Re-promotes to hard once the benchmark separates vision-encoder time
#     from language-model prefill time.
#   - text_decode_ratio: advisory at the >= 1.5x target, but with a HARD
#     non-regression floor (>= 1.0x; see ADVISORY_HARD_FLOORS). Decode of a
#     dense model is per-token weight-read-bandwidth bound; on tiny 4-bit
#     Gemma 4 e2b, llama.cpp's mature Metal kernels are at parity, and the
#     user-visible "1.5x faster" claim is carried by text_wall/text_ttft
#     (both hard and passing). The floor still guarantees Krill is never
#     slower than Ollama at decode. Re-promote text_decode_ratio to hard
#     >= 1.5x when EITHER (a) Gemma 4 speculative decoding lands and sustains
#     >= 1.5x with greedy parity, OR (b) the matrix adds a long-output decode
#     task where decode dominates wall time. Owner-accepted 2026-05-16; see
#     docs/RELEASE_GATE_DECODE_PROPOSAL.md.
# WS6 landed native Swift+MLX audio (default-on), numerically validated vs
# the mlx-vlm oracle and benchmarked faster than Ollama on the M4 target.
# `audio_wall_ratio` is HARD in both profiles (stable end-to-end metric;
# consistently passes <= 0.67). `audio_prefill_ratio` follows the same
# pattern as text/image prefill — advisory under release_candidate, hard
# under strict — because prefill TPS on a short clip is measured over a
# ~10-30ms window and is dominated by cold-start + Ollama-side jitter
# (observed 1.29-2.42x run-to-run with identical setup), not a real
# Krill signal. See docs/BENCHMARKING.md "audio prefill measurement".
GATE_PROFILES: dict[str, dict[str, str]] = {
    "strict": {
        # text_decode_ratio: advisory at >= 1.5x, with the synthetic HARD
        # `text_decode_ratio_floor` (>= 1.0x) from ADVISORY_HARD_FLOORS. The
        # >= 1.5x decode ratio is structurally unreachable on M-series
        # (owner-accepted 2026-05-22; docs/RELEASE_GATE_STRICT_DECODE_PROPOSAL.md).
        # image_prefill_ratio: advisory at >= 1.5x with NO floor — the
        # vision-encoder cache makes the prefill-TPS bucket structurally < 1.0
        # (owner-accepted 2026-05-22; docs/RELEASE_GATE_IMAGE_PREFILL_PROPOSAL.md);
        # the HARD image_wall_ratio carries the user-visible guarantee. Every
        # other strict metric stays hard.
        "text_decode_ratio":     "advisory",
        "text_wall_ratio":       "hard",
        "text_ttft_ratio":       "hard",
        "text_prefill_ratio":    "hard",
        "image_wall_ratio":      "hard",
        "image_prefill_ratio":   "advisory",
        "audio_wall_ratio":      "hard",
        "audio_prefill_ratio":   "hard",
        "memory_ratio":          "hard",
    },
    "release_candidate": {
        # advisory at >= 1.5x, but ADVISORY_HARD_FLOORS adds a synthetic
        # HARD `text_decode_ratio_floor` (>= 1.0x) so a decode regression
        # vs Ollama still breaks the gate. See the comment block above.
        "text_decode_ratio":     "advisory",
        "text_wall_ratio":       "hard",
        "text_ttft_ratio":       "hard",
        "text_prefill_ratio":    "advisory",
        "image_wall_ratio":      "hard",
        "image_prefill_ratio":   "advisory",
        # WS6: audio_wall is the stable end-to-end audio metric (hard).
        # audio_prefill is advisory here (like text/image prefill): its
        # short-window measurement is too noisy to hard-gate. Hard under
        # strict (the uncompromised reference).
        "audio_wall_ratio":      "hard",
        "audio_prefill_ratio":   "advisory",
        # memory_ratio is hard now that `gemma4_multimodal_benchmark.py` samples
        # peak RSS for both engines. It is still automatically downgraded to
        # advisory for any comparison whose quantization classes differ (e.g.
        # Krill bf16 vs Ollama Q4_K_M) — a cross-quantization memory
        # comparison cannot fairly gate a release. See `resolve_metric_kinds`.
        "memory_ratio":          "hard",
    },
}

# Human-readable reason shown next to each out_of_scope skip in reports.
# Empty since WS6: audio_* are now hard in both profiles (native Swift
# audio default-on, validated and faster than Ollama). Kept for any future
# scoped metric.
SCOPE_REASONS: dict[str, dict[str, str]] = {}

VALID_METRIC_KINDS = {"hard", "advisory", "out_of_scope"}


def resolve_metric_kinds(profile: str, report: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Return (metric -> kind) for `profile`, applying report-dependent
    adjustments, plus a list of human-readable notes describing them.

    Currently the only adjustment: `memory_ratio` is downgraded from hard to
    advisory under non-strict profiles when the two engines report different
    quantization classes (Krill bf16 vs Ollama Q4_K_M, say). A peak-memory
    comparison across quantization classes is dominated by the weight-format
    difference, not the runtime, so it cannot fairly hard-gate a release. It
    re-promotes to hard automatically once a quantization-class-equal report is
    supplied. This `memory_ratio` downgrade is not applied under `strict`
    (the `text_decode_ratio` advisory demotion is a separate, profile-level
    decision encoded directly in `GATE_PROFILES`).
    """
    kinds = dict(GATE_PROFILES[profile])
    notes: list[str] = []
    if profile != "strict" and kinds.get("memory_ratio") == "hard":
        comparison = (report.get("quantization") or {}).get("comparison") or {}
        if not comparison.get("class_equal", False):
            kinds["memory_ratio"] = "advisory"
            notes.append(
                "memory_ratio downgraded to advisory: Krill and Ollama report "
                "different quantization classes, so the peak-memory comparison is "
                "not apples-to-apples. It re-promotes to hard automatically once a "
                "quantization-class-equal benchmark is run."
            )
    return kinds, notes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Krill release benchmark gate."
    )
    parser.add_argument(
        "report",
        nargs="?",
        help="Path to a combined benchmark report JSON (contains both krill and ollama results).",
    )
    parser.add_argument(
        "--krill-report",
        help="Path to a standalone Krill benchmark report (for sequential comparison).",
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
    parser.add_argument(
        "--profile",
        choices=sorted(GATE_PROFILES.keys()),
        default="strict",
        help=(
            "Gate profile. 'strict' (default) hard-gates every metric and "
            "preserves the original behavior. 'release_candidate' applies "
            "the semantics in OLLAMA_SPEEDUP_EXECUTION_PLAN.md §4: "
            "prefill TPS metrics become advisory and audio metrics are "
            "out_of_scope until native Swift audio ships. See "
            "docs/BENCHMARKING.md for the rationale."
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
    if "krill" in results and "ollama" in results:
        return "flat"
    if any(k in results for k in ("text", "image", "audio")):
        return "multimodal"
    raise ValueError("Unrecognized report format: results must contain krill/ollama or text/image/audio keys.")


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
    """Extract metrics from flat krill-vs-ollama report."""
    r = report.get("results", {})
    ks = r.get("krill", {}).get("summary", {})
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
        ks = task_data.get("krill", {}).get("summary", {})
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
    text_ks = results.get("text", {}).get("krill", {}).get("summary", {})
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
    kind: str = "hard",
) -> dict[str, Any]:
    """Evaluate a single metric against its threshold.

    `kind` is one of `hard`, `advisory`, `out_of_scope`. Only `hard` failures
    propagate to the gate verdict; advisory failures are reported alongside
    passing ones so reviewers can still see them.
    """
    if kind not in VALID_METRIC_KINDS:
        raise ValueError(f"invalid metric kind {kind!r} for {name}")

    if value is None:
        return {
            "name": name,
            "kind": kind,
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
        "kind": kind,
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


# Metrics excluded from the geometric-mean "speedup" headline. memory_ratio is
# a footprint ratio, not a speed ratio, and it is dominated by the weight format
# (Krill bf16 vs Ollama Q4_K_M) rather than the runtime — folding it into a
# perf headline would understate the speed result for reasons unrelated to speed.
SPEEDUP_EXCLUDED_METRICS = {"memory_ratio"}


def compute_speedup_factors(metrics: dict[str, Optional[float]]) -> list[float]:
    """Convert speed metrics to speedup factors (>1 = Krill is better)."""
    factors: list[float] = []
    for name, value in metrics.items():
        if value is None or name in SPEEDUP_EXCLUDED_METRICS:
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
    profile: str,
    scope_skipped: list[dict[str, Any]],
    kv_cache_dtype: str = "fp16",
) -> None:
    print()
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Krill Release Benchmark Gate{RESET}  ({CYAN}profile: {profile}, kv: {kv_cache_dtype}{RESET})")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print()

    for e in evaluations:
        val = e["value"]
        thr = e["threshold"]
        direction = e.get("direction", "")
        kind = e.get("kind", "hard")
        kind_tag = "" if kind == "hard" else f" {YELLOW}[advisory]{RESET}"

        if val is None:
            icon = f"{YELLOW}--{RESET}"
            val_str = "N/A"
        elif e["pass"]:
            icon = f"{GREEN}OK{RESET}"
            val_str = f"{val:.4f}"
        elif kind == "advisory":
            # Advisory misses are warnings, not gate failures.
            icon = f"{YELLOW}WARN{RESET}"
            val_str = f"{val:.4f}"
        else:
            icon = f"{RED}FAIL{RESET}"
            val_str = f"{val:.4f}"

        print(f"  [{icon}] {e['name']:<30s}  {val_str:>10s}  (need {direction} {thr}){kind_tag}")

    if scope_skipped:
        print()
        for entry in scope_skipped:
            print(f"  {CYAN}SKIP{RESET} {entry['metric']:<28s}  {entry['reason']}")

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
    elif args.krill_report and args.ollama_report:
        # Sequential comparison: merge two single-engine reports
        # This handles the case where disk-constrained machines run one engine at a time
        krill = load_json(args.krill_report)
        ollama = load_json(args.ollama_report)
        report = {
            "status": "ok",
            "benchmark": {
                "krill_source": args.krill_report,
                "ollama_source": args.ollama_report,
                "comparison_mode": "sequential",
            },
            "environment": krill.get("environment", {}),
            "results": {
                "krill": krill.get("results", {}).get("krill", {}),
                "ollama": ollama.get("results", {}).get("ollama", {}),
            },
        }
    else:
        print("ERROR: provide a combined report path or --krill-report and --ollama-report", file=sys.stderr)
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
    scope_info: dict[str, Any] = {"scope": args.scope, "profile": args.profile}
    if server_media_out_of_scope:
        scope_info["server_media"] = "out_of_scope"

    media_metric_prefixes = ("image_", "audio_")

    profile_kinds, profile_kind_notes = resolve_metric_kinds(args.profile, report)
    profile_reasons = SCOPE_REASONS.get(args.profile, {})
    for note in profile_kind_notes:
        caveats.append(note)
    if profile_kinds.get("memory_ratio") != GATE_PROFILES[args.profile].get("memory_ratio"):
        scope_info["memory_ratio"] = (
            f"{GATE_PROFILES[args.profile].get('memory_ratio')} -> "
            f"{profile_kinds.get('memory_ratio')}"
        )

    # Evaluate each metric against thresholds
    evaluations: list[dict[str, Any]] = []
    skipped_for_scope: list[dict[str, Any]] = []
    for name, threshold in sorted(thresholds.items()):
        # Default kind is hard for metrics not listed in the profile (forwards-
        # compatible: a new metric added to DEFAULT_THRESHOLDS is hard-gated
        # until a profile explicitly downgrades it).
        kind = profile_kinds.get(name, "hard")

        value = metrics.get(name)
        if value is None and name not in metrics:
            continue

        if kind == "out_of_scope":
            skipped_for_scope.append({
                "metric": name,
                "reason": profile_reasons.get(name, f"out of scope under profile '{args.profile}'"),
            })
            continue

        if server_media_out_of_scope and name.startswith(media_metric_prefixes):
            if value is None:
                skipped_for_scope.append({
                    "metric": name,
                    "reason": "server media payload reported as out_of_scope for this report.",
                })
                continue

        evaluations.append(evaluate_metric(name, value, threshold, kind=kind))

        # Non-regression floor: when a profile demoted this metric to
        # advisory but it carries a hard floor, append a synthetic HARD
        # `<metric>_floor` evaluation so a regression past the floor still
        # breaks the gate. A MISSING value also hard-fails the floor — a
        # release cannot rest on unmeasured decode. Fully visible in
        # evaluations / summary / report. Keyed on the metric kind, not the
        # profile name: every profile that demotes the metric (today both
        # `release_candidate` and `strict`) gets the same floor.
        if (
            kind == "advisory"
            and name in ADVISORY_HARD_FLOORS
        ):
            floor = ADVISORY_HARD_FLOORS[name]
            floor_name = f"{name}_floor"
            evaluations.append(
                evaluate_metric(floor_name, value, floor, kind="hard")
            )
            scope_info[name] = (
                f"{GATE_PROFILES[args.profile].get(name)} (>= {threshold}x "
                f"advisory) + HARD non-regression floor {floor_name} "
                f">= {floor}x"
            )
            provenance = DECODE_ADVISORY_PROVENANCE.get(
                args.profile, "owner-accepted")
            caveats.append(
                f"{name} demoted to advisory under '{args.profile}' (>= "
                f"{threshold}x target) but hard-gated by {floor_name} >= "
                f"{floor}x: Krill must never decode slower than Ollama. "
                f"{provenance}; re-promotes to hard >= {threshold}x per the "
                f"contract in OLLAMA_SPEEDUP_EXECUTION_PLAN.md §4."
            )

        # Floor-less advisory demotion: the metric is structurally < 1.0 by
        # design (mismeasured, not slow), so no `<metric>_floor` is added —
        # the hard guarantee is carried by a sibling metric. Still emit a
        # caveat + scope note citing the owner decision so the relaxation is
        # never silent. Today: `image_prefill_ratio` under `strict`.
        elif (
            kind == "advisory"
            and (args.profile, name) in ADVISORY_DEMOTION_PROVENANCE
        ):
            provenance = ADVISORY_DEMOTION_PROVENANCE[(args.profile, name)]
            scope_info[name] = (
                f"advisory (>= {threshold}x target, no floor): the "
                f"vision-encoder cache lifts encoder+projector cost out of "
                f"the prefill window; the HARD image_wall_ratio carries the "
                f"user-visible guarantee."
            )
            caveats.append(
                f"{name} demoted to advisory under '{args.profile}' (>= "
                f"{threshold}x target, no floor). The vision-encoder cache "
                f"moves SigLIP2 forward + projector cost out of the measured "
                f"prefill window, so this prefill-TPS bucket understates real "
                f"performance; the hard image_wall_ratio is the user-visible "
                f"guarantee. {provenance}; re-promotes to hard >= {threshold}x "
                f"once the benchmark separates vision-encoder time from "
                f"language-model prefill time."
            )

    # Compute aggregate stats
    speedup_factors = compute_speedup_factors(metrics)
    geo_speedup = geometric_mean(speedup_factors) if speedup_factors else 0.0
    bottleneck = classify_bottleneck(evaluations)

    # Only hard failures break the gate. Advisory metrics are reported but
    # never gate-blocking under any profile. Missing hard metrics count as
    # failures: we cannot claim the gate passes for a metric we did not
    # measure. Missing advisory metrics are tolerated (they print N/A and
    # do not affect the verdict either way).
    hard_evals = [e for e in evaluations if e.get("kind", "hard") == "hard"]
    hard_pass = all(e["pass"] for e in hard_evals)
    all_pass = all(e["pass"] for e in evaluations if e["value"] is not None)
    gate_pass = hard_pass and not compatibility_fail

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

    # Surface KV cache dtype so reviewers can spot int8 vs fp16 runs without
    # opening the source benchmark report. The benchmark harness writes this
    # under benchmark.kv_cache_dtype; older reports without the field are
    # treated as fp16 (the default).
    bench = report.get("benchmark") or {}
    kv_cache_dtype = bench.get("kv_cache_dtype") or "fp16"

    # Build gate report
    gate_report: dict[str, Any] = {
        "gate": "pass" if gate_pass else "fail",
        "profile": args.profile,
        "kv_cache_dtype": kv_cache_dtype,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": args.report or {
            "krill": args.krill_report,
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
            "hard_metrics_pass": hard_pass,
            "all_metrics_pass": all_pass,
            "compatibility_ok": not compatibility_fail,
        },
        "caveats": caveats,
        "scope": scope_info,
    }
    if skipped_for_scope:
        gate_report["scope_skipped_metrics"] = sorted(
            skipped_for_scope, key=lambda x: x["metric"]
        )

    # Write report
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate_report, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    # Terminal output
    print_gate_summary(
        evaluations, geo_speedup, bottleneck, caveats, gate_pass,
        profile=args.profile,
        scope_skipped=gate_report.get("scope_skipped_metrics", []),
        kv_cache_dtype=kv_cache_dtype,
    )
    print(f"Gate report: {args.output}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
