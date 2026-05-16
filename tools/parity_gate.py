#!/usr/bin/env python3
"""KrillLM <-> Ollama macOS parity gate.

Sibling to ``release_gate.py``. Where the release gate tracks *speed*
parity, this tracks *feature / configuration* parity: it boots
``krillm serve`` and fires real Ollama-client-shaped requests at it,
asserting response-shape parity per the gap matrix in
``docs/OLLAMA_MAC_PARITY_PLAN.md``.

Profiles (mirror the speedup gate):
  - ``mac_parity``    : every ``H`` (hard) row must pass. ``A`` rows may
                        be skipped with a logged advisory. ``OOS`` excluded.
  - ``strict_parity`` : every ``H`` and ``A`` row must pass; no skips.

A check may report SKIP when its precondition (e.g. an installed model)
is absent; under ``mac_parity`` a SKIP on an ``H`` row is a FAIL, on an
``A`` row it is an allowed advisory.

Exit codes:
  0  Selected profile satisfied.
  1  Profile not satisfied (one or more required rows failed).
  2  Bad arguments / could not start server.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parent.parent


@dataclass
class CheckResult:
    pid: str          # parity ID from the gap matrix (e.g. "T0-3")
    gate: str         # "H" or "A"
    name: str
    status: str       # "PASS" | "FAIL" | "SKIP"
    detail: str = ""


@dataclass
class Gate:
    base_url: str
    results: list[CheckResult] = field(default_factory=list)

    # -- HTTP helpers -----------------------------------------------------
    def _req(self, method: str, path: str, body: Optional[dict] = None,
             timeout: float = 10.0) -> tuple[int, bytes]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    # -- check registration ----------------------------------------------
    def check(self, pid: str, gate: str, name: str,
              fn: Callable[[], tuple[str, str]]) -> None:
        try:
            status, detail = fn()
        except Exception as e:  # noqa: BLE001 - gate must never crash
            status, detail = "FAIL", f"exception: {e}"
        self.results.append(CheckResult(pid, gate, name, status, detail))
        mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
        print(f"  [{mark}] {pid:<5} ({gate}) {name}"
              + (f" - {detail}" if detail else ""))

    # -- individual parity checks ----------------------------------------
    def run_all(self) -> None:
        print("Parity checks:")

        def api_version() -> tuple[str, str]:
            code, raw = self._req("GET", "/api/version")
            if code != 200:
                return "FAIL", f"status {code}"
            j = json.loads(raw)
            if "version" not in j:
                return "FAIL", "missing 'version'"
            return "PASS", f"version={j['version']} krillm={j.get('krillm_version')}"

        self.check("T0-3", "H", "GET /api/version", api_version)

        def api_tags() -> tuple[str, str]:
            code, raw = self._req("GET", "/api/tags")
            if code != 200:
                return "FAIL", f"status {code}"
            j = json.loads(raw)
            if "models" not in j or not isinstance(j["models"], list):
                return "FAIL", "missing 'models' list"
            return "PASS", f"{len(j['models'])} model(s)"

        self.check("T0-3", "H", "GET /api/tags", api_tags)

        def api_ps() -> tuple[str, str]:
            code, raw = self._req("GET", "/api/ps")
            if code != 200:
                return "FAIL", f"status {code}"
            j = json.loads(raw)
            if "models" not in j or not isinstance(j["models"], list):
                return "FAIL", "missing 'models' list"
            return "PASS", f"{len(j['models'])} loaded"

        self.check("T0-3", "H", "GET /api/ps", api_ps)

        # /api/show needs an installed model; SKIP cleanly if none.
        def api_show() -> tuple[str, str]:
            _, raw = self._req("GET", "/api/tags")
            models = json.loads(raw).get("models", [])
            if not models:
                return "SKIP", "no installed model"
            name = models[0]["name"]
            code, sraw = self._req("POST", "/api/show", {"model": name})
            if code != 200:
                return "FAIL", f"status {code}"
            j = json.loads(sraw)
            need = {"modelfile", "template", "details", "capabilities"}
            missing = need - j.keys()
            if missing:
                return "FAIL", f"missing {sorted(missing)}"
            return "PASS", f"shape ok for {name}"

        self.check("T0-3", "H", "POST /api/show", api_show)

        def api_show_404() -> tuple[str, str]:
            code, _ = self._req("POST", "/api/show", {"model": "no-such-model"})
            return ("PASS", "404 as expected") if code == 404 \
                else ("FAIL", f"status {code} (expected 404)")

        self.check("T0-3", "H", "POST /api/show unknown -> 404", api_show_404)

        def api_delete_404() -> tuple[str, str]:
            code, _ = self._req("DELETE", "/api/delete", {"model": "no-such-model"})
            return ("PASS", "404 as expected") if code == 404 \
                else ("FAIL", f"status {code} (expected 404)")

        self.check("T2-7", "H", "DELETE /api/delete unknown -> 404", api_delete_404)

        def api_copy_400() -> tuple[str, str]:
            code, _ = self._req("POST", "/api/copy", {"source": ""})
            return ("PASS", "400 as expected") if code == 400 \
                else ("FAIL", f"status {code} (expected 400)")

        self.check("T2-7", "H", "POST /api/copy bad-req -> 400", api_copy_400)

        def api_blobs_head() -> tuple[str, str]:
            code, _ = self._req("HEAD", "/api/blobs/sha256:0", None)
            return ("PASS", "404 as expected") if code == 404 \
                else ("FAIL", f"status {code} (expected 404)")

        self.check("T2-7", "H", "HEAD /api/blobs/:digest -> 404", api_blobs_head)

        # Not yet implemented (later phases) -- recorded so the profile
        # verdict honestly reflects that mac_parity is not yet green.
        def embeddings() -> tuple[str, str]:
            code, _ = self._req("POST", "/api/embed",
                                 {"model": "x", "input": "hi"})
            if code == 404:
                return "FAIL", "not implemented (WS-B, Phase 1 pending)"
            return "PASS", f"status {code}"

        self.check("T0-2", "H", "POST /api/embed", embeddings)

        def tools() -> tuple[str, str]:
            code, raw = self._req(
                "POST", "/v1/chat/completions",
                {"model": "x",
                 "messages": [{"role": "user", "content": "hi"}],
                 "tools": []})
            # Today tools[] is rejected at parse time (400). Parity needs
            # it accepted -> WS-D D1 (Phase 2).
            if code == 400 and b"not supported" in raw:
                return "FAIL", "tools rejected (WS-D D1, Phase 2 pending)"
            return "PASS", f"status {code}"

        self.check("T0-4", "H", "tools/function calling", tools)

    # -- verdict ----------------------------------------------------------
    def verdict(self, profile: str) -> bool:
        print(f"\nProfile: {profile}")
        ok = True
        for r in self.results:
            required = (
                r.gate == "H"
                or (profile == "strict_parity" and r.gate == "A")
            )
            if not required:
                continue
            if r.status == "PASS":
                continue
            if r.status == "SKIP" and profile == "strict_parity":
                ok = False
            elif r.status == "SKIP":
                # mac_parity: SKIP on H is a fail, on A is advisory
                if r.gate == "H":
                    ok = False
            else:  # FAIL
                ok = False

        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        print(f"  {passed} pass / {failed} fail / {skipped} skip")
        print(f"\n{'GREEN' if ok else 'NOT GREEN'}: {profile} "
              f"{'satisfied' if ok else 'not yet satisfied'}")
        if not ok:
            print("  Outstanding (expected during phased delivery):")
            for r in self.results:
                req = r.gate == "H" or profile == "strict_parity"
                if req and r.status != "PASS":
                    print(f"    - {r.pid} {r.name}: {r.status} {r.detail}")
        return ok


def find_binary() -> Optional[Path]:
    candidates = [REPO / ".build" / cfg / "krillm"
                  for cfg in ("release", "debug")]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    # Prefer the most recently built binary so a stale release build
    # doesn't shadow a fresh `swift build` (debug).
    return max(existing, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser(description="KrillLM macOS parity gate")
    ap.add_argument("--profile", choices=["mac_parity", "strict_parity"],
                    default="mac_parity")
    ap.add_argument("--port", type=int, default=11534,
                    help="ephemeral port for the gate's krillm serve")
    ap.add_argument("--base-url", default=None,
                    help="test an already-running server instead of spawning one")
    args = ap.parse_args()

    if args.base_url:
        gate = Gate(args.base_url.rstrip("/"))
        gate.run_all()
        return 0 if gate.verdict(args.profile) else 1

    binary = find_binary()
    if not binary:
        print("error: krillm binary not found; run `swift build` first",
              file=sys.stderr)
        return 2

    print(f"Starting {binary} serve --port {args.port} --compat both ...")
    proc = subprocess.Popen(
        [str(binary), "serve", "--port", str(args.port), "--compat", "both"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        base = f"http://127.0.0.1:{args.port}"
        # Wait for readiness via /healthz.
        for _ in range(50):
            try:
                with urllib.request.urlopen(base + "/healthz", timeout=1):
                    break
            except Exception:  # noqa: BLE001
                if proc.poll() is not None:
                    out = proc.stdout.read() if proc.stdout else ""
                    print(f"error: server exited early:\n{out}",
                          file=sys.stderr)
                    return 2
                time.sleep(0.2)
        else:
            print("error: server did not become ready", file=sys.stderr)
            return 2

        gate = Gate(base)
        gate.run_all()
        return 0 if gate.verdict(args.profile) else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
