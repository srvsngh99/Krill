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
            # If a dedicated embedding model is installed, do a real embed
            # and assert a numeric vector. Otherwise verify the endpoint is
            # *implemented* (route exists, validates model/input) rather
            # than the default "Not found:" 404.
            _, traw = self._req("GET", "/api/tags")
            models = json.loads(traw).get("models", [])
            embed_model = next(
                (m["name"] for m in models
                 if (m.get("details") or {}).get("family") == "bert"), None)
            if embed_model:
                code, raw = self._req(
                    "POST", "/api/embed",
                    {"model": embed_model, "input": "hello world"})
                if code != 200:
                    return "FAIL", f"status {code}: {raw[:120]!r}"
                vecs = json.loads(raw).get("embeddings")
                if (not vecs or not isinstance(vecs[0], list)
                        or not isinstance(vecs[0][0], (int, float))):
                    return "FAIL", "malformed embeddings array"
                return "PASS", f"{embed_model}: dim={len(vecs[0])}"
            code, raw = self._req("POST", "/api/embed",
                                   {"model": "__not_installed__", "input": "hi"})
            if code == 404 and b"Not found:" in raw:
                return "FAIL", "endpoint not implemented"
            if code in (400, 404):
                return "PASS", ("implemented (no embed model installed; "
                                "pull e.g. bge-small-en for a live check)")
            return "FAIL", f"unexpected status {code}"

        self.check("T0-2", "H", "POST /api/embed", embeddings)

        def tools() -> tuple[str, str]:
            code, raw = self._req(
                "POST", "/v1/chat/completions",
                {"model": "x",
                 "messages": [{"role": "user", "content": "weather in NYC?"}],
                 "tools": [{
                     "type": "function",
                     "function": {
                         "name": "get_weather",
                         "description": "Get weather for a city",
                         "parameters": {
                             "type": "object",
                             "properties": {"city": {"type": "string"}},
                             "required": ["city"],
                         },
                     },
                 }]})
            # tools[] must be accepted (no parse-time 400 rejection). With no
            # model loaded the request reaches the model gate (503); with a
            # model it returns a 200 chat completion (optionally tool_calls).
            if code == 400 and b"not supported" in raw:
                return "FAIL", "tools rejected at parse time"
            if code == 503:
                return "PASS", "accepted (reaches model gate; no model loaded)"
            if code == 200:
                try:
                    j = json.loads(raw)
                    msg = j["choices"][0]["message"]
                    if "tool_calls" in msg or "content" in msg:
                        return "PASS", "accepted; well-formed chat completion"
                except Exception:  # noqa: BLE001
                    pass
                return "FAIL", "200 but malformed completion"
            return "FAIL", f"unexpected status {code}: {raw[:120]!r}"

        self.check("T0-4", "H", "tools/function calling", tools)

        def cors() -> tuple[str, str]:
            # OPTIONS preflight from a localhost origin must be granted.
            req = urllib.request.Request(
                self.base_url + "/api/chat", method="OPTIONS",
                headers={"Origin": "http://localhost",
                         "Access-Control-Request-Method": "POST"})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    ao = r.headers.get("Access-Control-Allow-Origin")
            except urllib.error.HTTPError as e:
                ao = e.headers.get("Access-Control-Allow-Origin")
            except Exception as e:  # noqa: BLE001
                return "FAIL", f"preflight error: {e}"
            return ("PASS", f"ACAO={ao}") if ao else \
                ("FAIL", "no Access-Control-Allow-Origin on preflight")

        self.check("T3-1", "H", "CORS preflight (OPTIONS)", cors)

        def sampler_params() -> tuple[str, str]:
            # min_p / presence_penalty / frequency_penalty must be accepted
            # (no parse-time 400). 503 (no model) or 200 are both fine.
            code, raw = self._req(
                "POST", "/v1/chat/completions",
                {"model": "x",
                 "messages": [{"role": "user", "content": "hi"}],
                 "min_p": 0.05, "presence_penalty": 0.2,
                 "frequency_penalty": 0.2})
            if code == 400:
                return "FAIL", f"rejected sampler params: {raw[:120]!r}"
            if code in (200, 503):
                return "PASS", f"accepted (status {code})"
            return "FAIL", f"unexpected status {code}"

        self.check("T2-10", "H", "extended sampler params", sampler_params)

        def structured_output() -> tuple[str, str]:
            # Ollama format:"json" and OpenAI response_format must be
            # accepted (no parse-time 400). 503/200 are both fine.
            c1, r1 = self._req(
                "POST", "/api/chat",
                {"model": "x", "stream": False,
                 "messages": [{"role": "user", "content": "give json"}],
                 "format": "json"})
            c2, r2 = self._req(
                "POST", "/v1/chat/completions",
                {"model": "x",
                 "messages": [{"role": "user", "content": "give json"}],
                 "response_format": {"type": "json_object"}})
            for c, r, who in ((c1, r1, "ollama format"),
                              (c2, r2, "openai response_format")):
                if c == 400:
                    return "FAIL", f"{who} rejected: {r[:100]!r}"
                if c not in (200, 503):
                    return "FAIL", f"{who} unexpected {c}"
            return "PASS", "format + response_format accepted"

        self.check("T1-1", "H", "structured output (JSON/schema)",
                   structured_output)

        def modelfile_create() -> tuple[str, str]:
            _, traw = self._req("GET", "/api/tags")
            models = [m["name"] for m in json.loads(traw).get("models", [])]
            base = next((m for m in models
                         if (json.loads(self._req("POST", "/api/show",
                             {"model": m})[1]).get("details") or {}).get("family")
                         not in ("bert", None)), None)
            if not base:
                return "SKIP", "no chat base model installed"
            new = "_paritygate_custom"
            self._req("DELETE", "/api/delete", {"model": new})
            mf = f"FROM {base}\nPARAMETER temperature 0.1\nSYSTEM You are Krill."
            c, r = self._req("POST", "/api/create",
                             {"model": new, "modelfile": mf, "stream": False})
            if c != 200:
                return "FAIL", f"create status {c}: {r[:120]!r}"
            c2, r2 = self._req("POST", "/api/show", {"model": new})
            ok = c2 == 200 and json.loads(r2).get("system") == "You are Krill."
            self._req("DELETE", "/api/delete", {"model": new})
            return ("PASS", "create->show overrides round-trip") if ok \
                else ("FAIL", f"show did not reflect overrides: {r2[:120]!r}")

        self.check("T1-2", "H", "Modelfile create + show", modelfile_create)

        def keep_alive() -> tuple[str, str]:
            # keep_alive must be accepted (not 400) and `stop`
            # (/v1/models/unload) must succeed.
            c1, r1 = self._req(
                "POST", "/api/chat",
                {"model": "x", "stream": False, "keep_alive": "10m",
                 "messages": [{"role": "user", "content": "hi"}]})
            if c1 == 400:
                return "FAIL", f"keep_alive rejected: {r1[:100]!r}"
            c2, _ = self._req("POST", "/v1/models/unload", {})
            if c2 != 200:
                return "FAIL", f"stop/unload status {c2}"
            c3, r3 = self._req("GET", "/api/ps")
            if c3 != 200 or "models" not in json.loads(r3):
                return "FAIL", "/api/ps malformed"
            return "PASS", "keep_alive accepted; stop + ps ok"

        self.check("T1-4", "H", "keep_alive + stop + ps", keep_alive)

        def anthropic_messages() -> tuple[str, str]:
            # POST /v1/messages must be implemented (not the default 404)
            # and accept the Anthropic request shape.
            code, raw = self._req(
                "POST", "/v1/messages",
                {"model": "claude-x", "max_tokens": 64,
                 "system": "be brief",
                 "messages": [{"role": "user", "content": "hi"}],
                 "tools": [{"name": "t", "description": "d",
                            "input_schema": {"type": "object"}}]})
            if code == 404 and b"Not found:" in raw:
                return "FAIL", "endpoint not implemented"
            if code in (200, 503):
                return "PASS", f"accepted (status {code})"
            return "FAIL", f"unexpected status {code}: {raw[:120]!r}"

        self.check("T2-9", "A", "Anthropic /v1/messages", anthropic_messages)

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
