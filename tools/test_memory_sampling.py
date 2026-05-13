"""Unit tests for the peak-memory sampling helpers in
tools/gemma4_multimodal_benchmark.py.

Run with `python3 -m unittest tools.test_memory_sampling` from the repo root.
These tests do not require mlx-vlm, Pillow, or a running Ollama/KrillLM
server — they exercise the sampler against short-lived python subprocesses.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "krillm_bm", str(Path(__file__).resolve().parent / "gemma4_multimodal_benchmark.py")
)
bm = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(bm)


class PidParsingTests(unittest.TestCase):
    def test_parse_pid_list_handles_commas_and_spaces(self):
        self.assertEqual(bm._parse_pid_list("1,2 3,,4"), [1, 2, 3, 4])
        self.assertEqual(bm._parse_pid_list(""), [])
        self.assertEqual(bm._parse_pid_list(None), [])
        self.assertEqual(bm._parse_pid_list("not-a-pid"), [])

    def test_pgrep_excludes_self(self):
        # Whatever pgrep matches, it must never include this process's PID.
        for pattern in ("python", "ollama", "krillm"):
            for full in (False, True):
                self.assertNotIn(os.getpid(), bm._pgrep(pattern, full=full))

    def test_pgrep_handles_no_match_gracefully(self):
        # An obviously-nonexistent process name must return an empty list,
        # not raise (pgrep exits 1 on no match — we treat that as success).
        self.assertEqual(bm._pgrep("definitely-not-a-real-process-xyz", full=False), [])

    def test_resolve_overrides_skip_pgrep(self):
        # When the operator passes explicit PIDs, we honor them verbatim
        # rather than running pgrep.
        self.assertEqual(bm.resolve_ollama_pids("11,22, 33"), [11, 22, 33])
        self.assertEqual(bm.resolve_krillm_server_pids("7"), [7])


class ProcessTreeRssTests(unittest.TestCase):
    def test_empty_roots_returns_zero(self):
        self.assertEqual(bm._process_tree_rss_kb(set()), 0)

    def test_dead_pid_returns_zero(self):
        # Pick a PID extremely unlikely to be live. 0 (kernel) is not in `ps`
        # output for normal users; a nonexistent high pid likewise.
        self.assertEqual(bm._process_tree_rss_kb({999999}), 0)

    def test_self_pid_has_nonzero_rss(self):
        # Our own process is alive and using memory; the RSS must be > 0.
        kb = bm._process_tree_rss_kb({os.getpid()})
        self.assertGreater(kb, 0)


class RSSSamplerTests(unittest.TestCase):
    def test_sampler_with_no_pids_returns_none(self):
        sampler = bm.RSSSampler([], interval_s=0.02)
        with sampler:
            time.sleep(0.05)
        self.assertIsNone(sampler.peak_gb)

    def test_sampler_captures_subprocess_peak(self):
        # Allocate ~80 MB in a child process and hold it briefly. The sampler
        # must observe a peak RSS comfortably above zero.
        proc = subprocess.Popen([
            sys.executable, "-c",
            "x = bytearray(80 * 1024 * 1024); import time; time.sleep(0.5)",
        ])
        try:
            sampler = bm.RSSSampler([proc.pid], interval_s=0.02)
            with sampler:
                proc.wait(timeout=5.0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5.0)
        self.assertIsNotNone(sampler.peak_gb)
        # 80 MB in decimal GB ≈ 0.080; allow generous slack for interpreter
        # overhead and rounding.
        self.assertGreater(sampler.peak_gb, 0.04)
        self.assertEqual(sampler.basis, "rss_process_tree")

    def test_sampler_thread_stops_on_exit(self):
        # The daemon thread must be joined when the context manager exits.
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"])
        try:
            sampler = bm.RSSSampler([proc.pid], interval_s=0.02)
            with sampler:
                proc.wait(timeout=2.0)
            self.assertIsNone(sampler._thread, "sampler thread must be cleared on __exit__")
        finally:
            if proc.poll() is None:
                proc.kill()


class MemoryProbeTests(unittest.TestCase):
    def _args(self, **overrides) -> argparse.Namespace:
        defaults = dict(
            sample_memory="auto",
            memory_sample_interval_ms=50.0,
            ollama_pids=None,
            krillm_server_pid=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_disabled_when_off(self):
        probe = bm._MemoryProbe()
        probe.configure(self._args(sample_memory="off"))
        self.assertFalse(probe.enabled)
        sampler = probe.sampler_for("ollama")
        self.assertIsNone(sampler.peak_gb)
        # Empty pids when disabled, regardless of override
        probe2 = bm._MemoryProbe()
        probe2.configure(self._args(sample_memory="off", ollama_pids="123"))
        self.assertEqual(probe2.sampler_for("ollama")._roots, set())

    def test_enabled_uses_overrides(self):
        probe = bm._MemoryProbe()
        probe.configure(self._args(ollama_pids="123 456", krillm_server_pid="789"))
        self.assertTrue(probe.enabled)
        # Sampler for ollama uses the override directly (no pgrep needed).
        ollama_sampler = probe.sampler_for("ollama")
        self.assertEqual(ollama_sampler._roots, {123, 456})
        krillm_sampler = probe.sampler_for("krillm")
        self.assertEqual(krillm_sampler._roots, {789})

    def test_sampler_for_pids_records_observed(self):
        probe = bm._MemoryProbe()
        probe.configure(self._args())
        sampler = probe.sampler_for_pids([os.getpid()])
        self.assertEqual(sampler._roots, {os.getpid()})
        self.assertIn(os.getpid(), probe.observed["krillm"])

    def test_report_block_shape(self):
        probe = bm._MemoryProbe()
        probe.configure(self._args(memory_sample_interval_ms=80.0, ollama_pids="42"))
        probe.sampler_for("ollama")
        block = probe.report_block()
        self.assertEqual(block["requested"], "auto")
        self.assertTrue(block["enabled"])
        self.assertEqual(block["interval_ms"], 80)
        self.assertIn("rss", block["basis"])
        self.assertIn("mlx_metal", block["basis"])
        self.assertEqual(block["ollama_pids_sampled"], [42])
        self.assertEqual(block["krillm_pids_sampled"], [])


if __name__ == "__main__":
    unittest.main()
