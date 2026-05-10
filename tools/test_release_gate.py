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
) -> dict:
    """Build a minimal multimodal report. Each pair is (krillm, ollama)."""

    def task(decode: tuple[float, float] | None, prefill: tuple[float, float],
             wall: tuple[float, float], ttft: tuple[float, float] | None = None) -> dict:
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
        return {"krillm": {"summary": ks}, "ollama": {"summary": os_}}

    return {
        "status": "ok",
        "benchmark": {"kv_cache_dtype": kv_cache_dtype},
        "results": {
            "text":  task(text_decode, text_prefill, text_wall, ttft=text_ttft),
            "image": task(None, image_prefill, image_wall),
            "audio": task(None, audio_prefill, audio_wall),
        },
    }


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


class ReleaseCandidateProfileTests(unittest.TestCase):
    """Profile 'release_candidate' downgrades prefill TPS to advisory and
    scopes audio out, so a report with failing prefill TPS but passing wall
    metrics must still gate-pass."""

    def _baseline_report(self, **overrides) -> dict:
        # Numbers picked so wall/ttft pass and prefill nearly-passes; audio
        # is intentionally awful (mirrors current v4-mm.json) to exercise
        # the out_of_scope path.
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


if __name__ == "__main__":
    unittest.main()
