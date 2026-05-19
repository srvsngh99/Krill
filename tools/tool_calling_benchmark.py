#!/usr/bin/env python3
"""Controlled KrillLM vs Ollama tool-calling benchmark.

Companion to ``krillm_vs_ollama_benchmark.py``. Drives BOTH engines over
the Ollama-compatible ``/api/chat`` endpoint (KrillLM has parity here) on
the *same* Gemma 4 E2B weights and scores tool-call correctness, not
speed. The metric is the KrillLM-vs-Ollama parity ratio; the gate is
"KrillLM is no worse than Ollama on valid+exact tool calls".

Skip discipline mirrors the sibling harness: exit 77 when a local
prerequisite (a reachable engine, the model) is missing, so CI can
treat "couldn't run" distinctly from "ran and failed". The report is an
input+environment record, not a published claim.

See docs/NATIVE_TOOL_CALLING_PLAN.md §5.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SKIP_EXIT_CODE = 77

# --- Tool definitions (JSON-schema, OpenAI/Ollama shape) --------------------

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First addend."},
                "b": {"type": "integer", "description": "Second addend."},
            },
            "required": ["a", "b"],
        },
    },
}

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

MULTIPLY_TOOL = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two integers and return the product.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
}

# --- Suite ------------------------------------------------------------------
# Each task: id, description, tools, the opening user message, and an
# `expect` block the scorer checks against the parsed tool_calls.

SUITE: list[dict[str, Any]] = [
    {
        "id": "T1_single_tool",
        "desc": "One explicit tool; must call it once with exact args.",
        "tools": [ADD_TOOL],
        "user": "Use the add tool to compute 12 plus 30.",
        "expect": {"name": "add", "args": {"a": 12, "b": 30}, "must_call": True},
    },
    {
        "id": "T2_tool_selection",
        "desc": "Three tools; must pick exactly the right one.",
        "tools": [ADD_TOOL, GET_WEATHER_TOOL, MULTIPLY_TOOL],
        "user": "What is 7 multiplied by 6? Use a tool.",
        "expect": {"name": "multiply", "args": {"a": 7, "b": 6}, "must_call": True},
    },
    {
        "id": "T3_no_tool",
        "desc": "Answerable without tools; a tool call is a failure.",
        "tools": [ADD_TOOL, GET_WEATHER_TOOL],
        "user": "In one word, what color is a clear daytime sky?",
        "expect": {"must_call": False},
    },
    {
        "id": "T4_multistep_agentic",
        "desc": "Call add, inject the result, then expect a final answer.",
        "tools": [ADD_TOOL],
        "user": "Compute 100 + 23 using the tool, then state the result in a sentence.",
        "expect": {"name": "add", "args": {"a": 100, "b": 23}, "must_call": True},
        "followup": {"tool_result": "123", "expect_final_contains": "123"},
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--krillm-url", default="http://127.0.0.1:11435",
                   help="Running KrillLM server base URL.")
    p.add_argument("--ollama-host", default="http://127.0.0.1:11434",
                   help="Ollama API host.")
    p.add_argument("--krill-model", default="gemma-4-e2b")
    p.add_argument("--ollama-model", default="gemma4:e2b")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--output", default=".build/benchmarks/tool-calling.json")
    return p.parse_args()


def chat(base: str, model: str, messages: list[dict[str, Any]],
         tools: Optional[list[dict[str, Any]]], temperature: float,
         timeout: float) -> dict[str, Any]:
    """One non-streaming /api/chat turn. Returns the `message` object plus
    wall latency. Raises on transport failure (callers map to skip)."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "seed": 0},
    }
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode("utf-8")
    url = base.rstrip("/") + "/api/chat"
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    wall = time.perf_counter() - start
    msg = body.get("message", {}) or {}
    return {
        "message": msg,
        "tool_calls": msg.get("tool_calls") or [],
        "content": msg.get("content") or "",
        "latency_s": round(wall, 3),
    }


def normalize_args(raw: Any) -> dict[str, Any]:
    """Ollama returns arguments as an object; OpenAI-style as a JSON string.
    Coerce to a dict; unparseable -> {} so the scorer marks it malformed."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def score(task: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
    """Grade one engine's response to one task."""
    expect = task["expect"]
    calls = turn["tool_calls"]
    must_call = expect.get("must_call", False)

    if not must_call:
        # T3: success == no tool call emitted at all.
        ok = len(calls) == 0
        return {"valid_tool_call": ok, "args_exact": ok, "well_formed": ok,
                "n_calls": len(calls), "detail": "no-tool task"}

    if not calls:
        return {"valid_tool_call": False, "args_exact": False,
                "well_formed": False, "n_calls": 0, "detail": "no call emitted"}

    fn = calls[0].get("function", calls[0])  # tolerate flat or nested shape
    name = fn.get("name", "")
    args = normalize_args(fn.get("arguments"))
    exp_args = expect.get("args", {})

    name_ok = name == expect["name"]
    # well_formed: args is a flat dict whose values are scalars (the known
    # failure mode is double-nesting: {"a": {"a": 12, "b": 30}}).
    well_formed = bool(args) and all(
        not isinstance(v, (dict, list)) for v in args.values())
    args_exact = all(
        str(args.get(k)) == str(v) for k, v in exp_args.items()) and well_formed
    valid = name_ok and well_formed and len(calls) == 1

    return {"valid_tool_call": valid, "args_exact": args_exact,
            "well_formed": well_formed, "n_calls": len(calls),
            "got_name": name, "got_args": args}


def run_engine(label: str, base: str, model: str, temperature: float,
                timeout: float) -> dict[str, Any]:
    results: dict[str, Any] = {"engine": label, "base": base,
                               "model": model, "tasks": {}}
    for task in SUITE:
        messages = [{"role": "user", "content": task["user"]}]
        turn = chat(base, model, messages, task["tools"], temperature, timeout)
        sc = score(task, turn)
        entry: dict[str, Any] = {
            "score": sc,
            "latency_s": turn["latency_s"],
            "content_preview": turn["content"][:160],
            "raw_tool_calls": turn["tool_calls"],
        }
        # T4 follow-up: feed the tool result back and require a final answer.
        if sc["valid_tool_call"] and "followup" in task:
            fu = task["followup"]
            messages.append(turn["message"])
            messages.append({"role": "tool", "content": fu["tool_result"]})
            try:
                f = chat(base, model, messages, task["tools"],
                         temperature, timeout)
                final_ok = (fu["expect_final_contains"] in f["content"]
                            and not f["tool_calls"])
                entry["followup"] = {"final_ok": final_ok,
                                     "content_preview": f["content"][:160]}
            except urllib.error.URLError as exc:
                entry["followup"] = {"final_ok": False, "error": str(exc)}
        results["tasks"][task["id"]] = entry
    return results


def reachable(base: str, timeout: float) -> bool:
    try:
        urllib.request.urlopen(
            base.rstrip("/") + "/api/version", timeout=min(timeout, 10))
        return True
    except urllib.error.HTTPError:
        return True  # endpoint exists, just no /api/version - good enough
    except (urllib.error.URLError, OSError):
        return False


def aggregate(res: dict[str, Any]) -> dict[str, int]:
    v = sum(t["score"]["valid_tool_call"] for t in res["tasks"].values())
    a = sum(t["score"]["args_exact"] for t in res["tasks"].values())
    f = sum(t.get("followup", {}).get("final_ok", True) is True
            and "followup" in t for t in res["tasks"].values())
    return {"valid": v, "args_exact": a, "followups_ok": f,
            "total": len(res["tasks"])}


def main() -> int:
    args = parse_args()

    for label, base in (("ollama", args.ollama_host),
                        ("krillm", args.krillm_url)):
        if not reachable(base, args.timeout):
            print(f"SKIP: {label} not reachable at {base} "
                  f"(start it, then re-run)", file=sys.stderr)
            return SKIP_EXIT_CODE

    report: dict[str, Any] = {
        "schema": "tool_calling_benchmark/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "config": {
            "krill_model": args.krill_model,
            "ollama_model": args.ollama_model,
            "temperature": args.temperature,
        },
    }

    try:
        ollama = run_engine("ollama", args.ollama_host, args.ollama_model,
                            args.temperature, args.timeout)
        krillm = run_engine("krillm", args.krillm_url, args.krill_model,
                            args.temperature, args.timeout)
    except urllib.error.URLError as exc:
        print(f"SKIP: engine request failed mid-run: {exc}", file=sys.stderr)
        return SKIP_EXIT_CODE

    report["ollama"] = ollama
    report["krillm"] = krillm
    o_agg, k_agg = aggregate(ollama), aggregate(krillm)
    report["summary"] = {"ollama": o_agg, "krillm": k_agg}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\nTool-calling parity ({krillm['model']} vs {ollama['model']}, "
          f"temp {args.temperature})")
    print(f"  {'task':<22} {'ollama':>16} {'krillm':>16}")
    for tid in (t["id"] for t in SUITE):
        os_, ks = ollama["tasks"][tid]["score"], krillm["tasks"][tid]["score"]
        def cell(s: dict[str, Any]) -> str:
            return ("valid" if s["valid_tool_call"] else "FAIL") + (
                "+exact" if s["args_exact"] else "")
        print(f"  {tid:<22} {cell(os_):>16} {cell(ks):>16}")
    print(f"  {'TOTAL valid':<22} {o_agg['valid']:>13}/{o_agg['total']} "
          f"{k_agg['valid']:>13}/{k_agg['total']}")
    print(f"  {'TOTAL args_exact':<22} {o_agg['args_exact']:>13}/{o_agg['total']} "
          f"{k_agg['args_exact']:>13}/{k_agg['total']}")
    print(f"\nReport: {out}")

    gate = (k_agg["valid"] >= o_agg["valid"]
            and k_agg["args_exact"] >= o_agg["args_exact"])
    if gate:
        print("GATE: PASS (KrillLM >= Ollama)")
        return 0
    print("GATE: FAIL (KrillLM behind Ollama - expected pre-fix; "
          "see docs/NATIVE_TOOL_CALLING_PLAN.md)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
