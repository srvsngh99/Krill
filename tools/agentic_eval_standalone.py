#!/usr/bin/env python3
"""Standalone agentic tool-use eval for a Krill-served model. No Ollama, no
comparison - single-engine capability numbers.

Runs a real agent loop over /v1/chat/completions: the model is given tool
schemas, and when it emits tool_calls the harness EXECUTES them locally, feeds
the results back, and continues until a final answer or step budget. This tests
the things agents actually need: valid tool-call JSON, correct tool selection,
correct argument extraction, multi-step sequencing with result dependencies, and
knowing when NOT to call a tool.

Per scenario it scores:
  valid_call      - emitted schema-valid tool_call(s) when one was expected
  correct_tool    - called the right tool(s)
  correct_args    - arguments matched expectation
  task_complete   - final assistant answer contains the expected result

Usage:
  python3 tools/agentic_eval_standalone.py --model gemma-4-12b-coder-nvfp4
"""
import argparse
import json
import time
import urllib.request

# ---- locally-executable tools (deterministic, known outputs) ----------------
def _add(a, b): return a + b
def _multiply(a, b): return a * b
def _get_weather(city): return {"city": city, "temp_c": 21, "conditions": "clear"}
def _word_count(text): return len(text.split())

EXEC = {"add": _add, "multiply": _multiply, "get_weather": _get_weather,
        "word_count": _word_count}

TOOLS = [
    {"type": "function", "function": {
        "name": "add", "description": "Add two numbers and return the sum.",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"]}}},
    {"type": "function", "function": {
        "name": "multiply", "description": "Multiply two numbers and return the product.",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"]}}},
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "word_count", "description": "Count the words in a piece of text.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}}, "required": ["text"]}}},
]

# ---- scenarios. expect.calls = ordered list of (name, args) the model should
#      make (set [] for the no-tool case); expect.answer = substring(s) the final
#      assistant message must contain. ----------------------------------------
SCEN = [
    {"id": "single_tool", "user": "Use a tool to compute 12 plus 30.",
     "calls": [("add", {"a": 12, "b": 30})], "answer": ["42"]},
    {"id": "tool_selection", "user": "What is 7 multiplied by 6? Use a tool.",
     "calls": [("multiply", {"a": 7, "b": 6})], "answer": ["42"]},
    {"id": "no_tool", "user": "Say hello and tell me you are ready. Do not call any tool.",
     "calls": [], "answer": []},
    {"id": "args_from_nl", "user": "I'm flying to Paris tomorrow - what's the weather there?",
     "calls": [("get_weather", {"city": "Paris"})], "answer": ["21", "clear"]},
    {"id": "two_step_nested", "user": "Compute (3 + 4) * 5 using the tools, then give the number.",
     "calls": [("add", {"a": 3, "b": 4}), ("multiply", {"a": 7, "b": 5})], "answer": ["35"]},
    {"id": "chained_dependency",
     "user": "Add 100 and 23 with the tool, then multiply that result by 2 with the tool. Give the final number.",
     "calls": [("add", {"a": 100, "b": 23}), ("multiply", {"a": 123, "b": 2})], "answer": ["246"]},
    {"id": "word_count_tool",
     "user": "Use a tool to count the words in: the quick brown fox jumps",
     "calls": [("word_count", {"text": "the quick brown fox jumps"})], "answer": ["5"]},
    {"id": "weather_then_reason",
     "user": "Check the weather in Tokyo with the tool, then tell me if I need a coat (need one below 10C).",
     "calls": [("get_weather", {"city": "Tokyo"})], "answer": ["no"]},
]


def post(url, payload, timeout=180):
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_args_obj(raw):
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return {}


def args_match(got, want):
    for k, v in want.items():
        if k not in got:
            return False
        g = got[k]
        if isinstance(v, (int, float)) and isinstance(g, (int, float, str)):
            try:
                if abs(float(g) - float(v)) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        elif str(g).strip().lower() != str(v).strip().lower():
            return False
    return True


def run_scenario(url, model, scen, max_steps=6, temp=0.0):
    messages = [{"role": "user", "content": scen["user"]}]
    made = []  # (name, args) actually executed
    valid_call = True
    final = ""
    for _ in range(max_steps):
        resp = post(url, {"model": model, "messages": messages, "tools": TOOLS,
                          "temperature": temp, "max_tokens": 512})
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            final = msg.get("content") or ""
            messages.append({"role": "assistant", "content": final})
            break
        messages.append(msg)
        for c in calls:
            fn = (c.get("function") or {})
            name = fn.get("name")
            a = parse_args_obj(fn.get("arguments"))
            if name not in EXEC:
                valid_call = False
                result = {"error": f"unknown tool {name}"}
            else:
                made.append((name, a))
                try:
                    result = EXEC[name](**a)
                except Exception as e:  # noqa: BLE001
                    valid_call = False
                    result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": c.get("id", name),
                             "name": name, "content": json.dumps(result)})
    # ---- score ----
    want = scen["calls"]
    if not want:  # no-tool scenario
        correct_tool = (len(made) == 0)
        correct_args = correct_tool
        valid_call = True
    else:
        want_names = [n for n, _ in want]
        got_names = [n for n, _ in made]
        correct_tool = got_names[: len(want_names)] == want_names
        correct_args = correct_tool and all(
            args_match(made[i][1], want[i][1]) for i in range(min(len(made), len(want))))
    ans_ok = all(s.lower() in final.lower() for s in scen["answer"]) if scen["answer"] else True
    return {"id": scen["id"], "valid_call": valid_call, "correct_tool": correct_tool,
            "correct_args": correct_args, "task_complete": bool(ans_ok and correct_args),
            "made": made, "final": final[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:57455")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--out", default="/tmp/agentic_eval_results.json")
    args = ap.parse_args()

    rows = []
    t0 = time.time()
    for s in SCEN:
        try:
            r = run_scenario(args.url, args.model, s, temp=args.temp)
        except Exception as e:  # noqa: BLE001
            r = {"id": s["id"], "valid_call": False, "correct_tool": False,
                 "correct_args": False, "task_complete": False, "err": str(e), "made": []}
        rows.append(r)
        print(f"{r['id']:20s} valid={int(r['valid_call'])} tool={int(r['correct_tool'])} "
              f"args={int(r['correct_args'])} complete={int(r['task_complete'])}  "
              f"calls={r.get('made')}")

    n = len(rows)
    def rate(k): return sum(1 for r in rows if r.get(k))
    summary = {
        "model": args.model, "scenarios": n,
        "valid_call": f"{rate('valid_call')}/{n}",
        "correct_tool": f"{rate('correct_tool')}/{n}",
        "correct_args": f"{rate('correct_args')}/{n}",
        "task_complete": f"{rate('task_complete')}/{n}",
        "task_complete_pct": round(100 * rate("task_complete") / n, 1),
        "wall_s": round(time.time() - t0, 1),
    }
    print("\n=== Agentic tool-use (standalone) ===")
    for k, v in summary.items():
        print(f"  {k:20s} {v}")
    json.dump({"summary": summary, "results": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
