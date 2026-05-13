"""Unit tests for tools/release_gate.py profile semantics.

Run with: `python3 -m unittest tools.test_release_gate` from repo root.

These tests exercise the metric-kind dispatch added for Workstream 4 of
OLLAMA_SPEEDUP_EXECUTION_PLAN.md. They use a small synthetic multimodal
report so they do not require a real benchmark run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
GATE_SCRIPT = REPO_ROOT / "tools" / "release_gate.py"


def make_report(
    text_decode: tuple[float, float],
    text_prefill: tuple[float, float],
    text_wall: tuple[float, float],
    text_ttft: tuple[float, float],
    image_wall: tuple[float, float],
    image_prefill: tuple[float, float],
    audio_wall: tuple[float, float],
    audio_prefill: tuple[float, float],
    *,
    kv_cache_dtype: str = "fp16",
    text_memory: tuple[float, float] | None = None,
    quant_class_equal: bool | None = None,
) -> dict:
    """Build a minimal multimodal report. Each pair is (krillm, ollama).

    `text_memory`, if given, adds `peak_memory_gb_median` to the text task so
    the gate can evaluate `memory_ratio`. `quant_class_equal`, if not None,
    adds a `quantization.comparison` block so the gate's
    quantization-class-aware downgrade of `memory_ratio` can be exercised.
    """

    def task(decode: tuple[float, float] | None, prefill: tuple[float, float],
             wall: tuple[float, float], ttft: tuple[float, float] | None = None,
             memory: tuple[float, float] | None = None) -> dict:
        ks: dict[str, float] = {
            "prefill_tokens_per_second_median": prefill[0],
            "wall_time_s_median": wall[0],
        }
        os_: dict[str, float] = {
            "prefill_tokens_per_second_median": prefill[1],
            "wall_time_s_median": wall[1],
        }
        if decode is not None:
            ks["decode_tokens_per_second_median"] = decode[0]
            os_["decode_tokens_per_second_median"] = decode[1]
        if ttft is not None:
            ks["ttft_ms_median"] = ttft[0]
            os_["ttft_ms_median"] = ttft[1]
        if memory is not None:
            ks["peak_memory_gb_median"] = memory[0]
            os_["peak_memory_gb_median"] = memory[1]
        return {"krillm": {"summary": ks}, "ollama": {"summary": os_}}

    report: dict = {
        "status": "ok",
        "benchmark": {"kv_cache_dtype": kv_cache_dtype},
        "results": {
            "text":  task(text_decode, text_prefill, text_wall, ttft=text_ttft, memory=text_memory),
            "image": task(None, image_prefill, image_wall),
            "audio": task(None, audio_prefill, audio_wall),
        },
    }
    if quant_class_equal is not None:
        # strict_equal mirrors class_equal so check_compatibility stays quiet
        # (a class-equal-but-not-bit-identical caveat is harmless either way).
        report["quantization"] = {
            "comparison": {"class_equal": quant_class_equal, "strict_equal": quant_class_equal}
        }
    return report


def run_gate(report: dict, *extra_args: str) -> tuple[int, dict]:
    """Run release_gate.py against an inline report; return (exit, gate_json)."""
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        report_path.write_text(json.dumps(report))
        out_path = Path(tmp) / "gate.json"
        proc = subprocess.run(
            [
                sys.executable, str(GATE_SCRIPT), str(report_path),
                "--output", str(out_path), *extra_args,
            ],
            check=False, capture_output=True, text=True,
        )
        gate = json.loads(out_path.read_text()) if out_path.exists() else {}
    return proc.returncode, gate


def baseline_report(**overrides) -> dict:
    """A report whose wall/ttft/decode metrics pass, prefill metrics
    near-miss (advisory under release_candidate), and audio is intentionally
    awful (mirrors current v4-mm.json) to exercise the out_of_scope path."""
    defaults = dict(
        text_decode=(150.0, 100.0),    # ratio 1.50 (hard pass)
        text_prefill=(145.0, 100.0),   # ratio 1.45 (advisory warn)
        text_wall=(0.50, 1.00),        # ratio 0.50 (hard pass)
        text_ttft=(50.0, 500.0),       # ratio 0.10 (hard pass)
        image_wall=(0.50, 1.00),       # ratio 0.50 (hard pass)
        image_prefill=(105.0, 100.0),  # ratio 1.05 (advisory warn)
        audio_wall=(4.00, 1.00),       # ratio 4.00 (out_of_scope)
        audio_prefill=(50.0, 100.0),   # ratio 0.50 (out_of_scope)
    )
    defaults.update(overrides)
    return make_report(**defaults)


class ReleaseCandidateProfileTests(unittest.TestCase):
    """Profile 'release_candidate' downgrades prefill TPS to advisory and
    scopes audio out, so a report with failing prefill TPS but passing wall
    metrics must still gate-pass."""

    def _baseline_report(self, **overrides) -> dict:
        return baseline_report(**overrides)

    def test_strict_profile_fails_when_prefill_below_target(self):
        code, gate = run_gate(self._baseline_report())
        self.assertEqual(code, 1, "strict profile should fail on prefill misses")
        self.assertEqual(gate["gate"], "fail")
        self.assertEqual(gate["profile"], "strict")

    def test_release_candidate_passes_when_only_advisory_misses(self):
        code, gate = run_gate(self._baseline_report(), "--profile", "release_candidate")
        self.assertEqual(code, 0, "release_candidate should pass on hard metrics alone")
        self.assertEqual(gate["gate"], "pass")
        self.assertEqual(gate["profile"], "release_candidate")

    def test_release_candidate_marks_audio_out_of_scope(self):
        _, gate = run_gate(self._baseline_report(), "--profile", "release_candidate")
        skipped = {entry["metric"] for entry in gate.get("scope_skipped_metrics", [])}
        self.assertEqual(skipped, {"audio_wall_ratio", "audio_prefill_ratio"})

    def test_release_candidate_fails_on_hard_metric(self):
        # Break text_wall (a hard-gated metric); the gate must fail even
        # though prefill misses are advisory.
        code, gate = run_gate(
            self._baseline_report(text_wall=(2.00, 1.00)),
            "--profile", "release_candidate",
        )
        self.assertEqual(code, 1, "hard-gated regression must still break gate")
        self.assertEqual(gate["gate"], "fail")

    def test_release_candidate_records_kv_cache_dtype(self):
        _, gate = run_gate(
            self._baseline_report(kv_cache_dtype="int8"),
            "--profile", "release_candidate",
        )
        self.assertEqual(gate.get("kv_cache_dtype"), "int8")

    def test_advisory_evaluations_are_tagged(self):
        _, gate = run_gate(self._baseline_report(), "--profile", "release_candidate")
        kinds = {e["name"]: e["kind"] for e in gate["evaluations"]}
        self.assertEqual(kinds.get("text_prefill_ratio"), "advisory")
        self.assertEqual(kinds.get("image_prefill_ratio"), "advisory")
        self.assertEqual(kinds.get("text_wall_ratio"), "hard")
        self.assertEqual(kinds.get("image_wall_ratio"), "hard")

    def test_missing_hard_metric_fails_the_gate(self):
        # Strip text_decode (a hard-gated metric) from the report by handing
        # the gate a results section with no decode_tokens_per_second_median.
        # The gate must treat the absence as a failure, not a silent pass.
        report = self._baseline_report()
        for engine in ("krillm", "ollama"):
            report["results"]["text"][engine]["summary"].pop("decode_tokens_per_second_median", None)

        code, gate = run_gate(report, "--profile", "release_candidate", "--allow-dtype-mismatch")
        self.assertEqual(code, 1, "missing hard metric must fail the gate")
        decode = next(e for e in gate["evaluations"] if e["name"] == "text_decode_ratio")
        self.assertIsNone(decode["value"])
        self.assertEqual(decode["kind"], "hard")
        self.assertFalse(decode["pass"])
        self.assertFalse(gate["summary"]["hard_metrics_pass"])


class MemoryMetricTests(unittest.TestCase):
    """`memory_ratio` is hard under release_candidate, but auto-downgraded to
    advisory when the engines' quantization classes differ (bf16 vs Q4_K_M).
    It is also kept out of the geometric-mean speedup headline."""

    def _eval(self, gate: dict, name: str) -> dict:
        return next(e for e in gate["evaluations"] if e["name"] == name)

    def test_memory_ratio_recorded_when_present(self):
        report = baseline_report(text_memory=(5.2, 1.6))  # ratio 3.25
        _, gate = run_gate(report, "--profile", "release_candidate")
        mem = self._eval(gate, "memory_ratio")
        self.assertAlmostEqual(mem["value"], 3.25, places=2)

    def test_memory_advisory_when_quant_classes_differ(self):
        report = baseline_report(text_memory=(5.2, 1.6), quant_class_equal=False)
        code, gate = run_gate(report, "--profile", "release_candidate", "--allow-dtype-mismatch")
        self.assertEqual(code, 0, "advisory memory miss must not break the gate")
        self.assertEqual(gate["gate"], "pass")
        mem = self._eval(gate, "memory_ratio")
        self.assertEqual(mem["kind"], "advisory")
        self.assertFalse(mem["pass"])
        self.assertIn("memory_ratio", gate["scope"])
        self.assertTrue(any("downgraded to advisory" in c for c in gate["caveats"]))

    def test_memory_hard_fails_when_classes_match_and_over_budget(self):
        report = baseline_report(text_memory=(2.0, 1.6), quant_class_equal=True)  # ratio 1.25
        code, gate = run_gate(report, "--profile", "release_candidate")
        self.assertEqual(code, 1, "hard memory miss must break the gate")
        self.assertEqual(gate["gate"], "fail")
        mem = self._eval(gate, "memory_ratio")
        self.assertEqual(mem["kind"], "hard")
        self.assertFalse(mem["pass"])
        self.assertFalse(gate["summary"]["hard_metrics_pass"])

    def test_memory_hard_passes_when_classes_match_and_under_budget(self):
        report = baseline_report(text_memory=(1.4, 1.6), quant_class_equal=True)  # ratio 0.875
        code, gate = run_gate(report, "--profile", "release_candidate")
        self.assertEqual(code, 0)
        mem = self._eval(gate, "memory_ratio")
        self.assertEqual(mem["kind"], "hard")
        self.assertTrue(mem["pass"])

    def test_strict_keeps_memory_hard_when_classes_differ(self):
        report = baseline_report(text_memory=(5.2, 1.6), quant_class_equal=False)
        _, gate = run_gate(report, "--allow-dtype-mismatch")  # default profile = strict
        mem = self._eval(gate, "memory_ratio")
        self.assertEqual(mem["kind"], "hard")
        self.assertNotIn("memory_ratio", gate["scope"])

    def test_memory_excluded_from_speedup_headline(self):
        without_mem = baseline_report()
        with_bad_mem = baseline_report(text_memory=(10.0, 1.0))  # awful ratio
        _, g0 = run_gate(without_mem, "--profile", "release_candidate")
        _, g1 = run_gate(with_bad_mem, "--profile", "release_candidate")
        self.assertEqual(
            g0["summary"]["geometric_mean_speedup"],
            g1["summary"]["geometric_mean_speedup"],
            "memory_ratio must not move the geometric-mean speedup headline",
        )


if __name__ == "__main__":
    unittest.main()
