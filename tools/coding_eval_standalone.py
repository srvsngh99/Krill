#!/usr/bin/env python3
"""Standalone HumanEval pass@1 for a KrillLM-served model. No Ollama, no
comparison - a single-engine capability + throughput number.

Drives the OpenAI-compatible /v1/chat/completions endpoint: for each HumanEval
problem it asks the model to complete the function, extracts the code block, and
executes the canonical test in a sandboxed subprocess with a wall-clock timeout.
Reports pass@1 and aggregate decode tok/s.

Usage:
  python3 tools/coding_eval_standalone.py \
      --model gemma-4-12b-coder-nvfp4 --data /tmp/he.jsonl --limit 0 --temp 0
  # --limit 0 = all 164 problems; --limit N = first N (quick noisy estimate)
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request

CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


_THINK_TMPL = None  # lazily-compiled jinja chat template (think mode)


def _think_prompt(template_path, user):
    """Render the model's own chat template with enable_thinking=True. KrillLM's
    /v1/chat path does not thread enable_thinking, so to benchmark the model as a
    *reasoning* model we render the prompt ourselves and use raw /v1/completions."""
    global _THINK_TMPL
    if _THINK_TMPL is None:
        from jinja2 import Environment, BaseLoader
        env = Environment(loader=BaseLoader())
        _THINK_TMPL = env.from_string(open(template_path).read())
    return _THINK_TMPL.render(messages=[{"role": "user", "content": user}],
                              add_generation_prompt=True, enable_thinking=True,
                              bos_token="<bos>", eos_token="<eos>", tools=None)


def _strip_think(t):
    for mark in ("<channel|>", "</think>", "<|channel>"):
        if mark in t:
            t = t.split(mark)[-1]
    return t


def chat(url, model, prompt, temp, max_tokens, timeout=300, think=False,
         template_path=None):
    t0 = time.time()
    if think:
        body = json.dumps({"model": model, "prompt": _think_prompt(template_path, prompt),
                           "temperature": temp, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(url + "/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        dt = time.time() - t0
        msg = _strip_think(d["choices"][0]["text"])
        ct = (d.get("usage") or {}).get("completion_tokens", 0)
        return msg, ct, dt
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    msg = d["choices"][0]["message"]["content"]
    ct = (d.get("usage") or {}).get("completion_tokens", 0)
    return msg, ct, dt


def extract_code(text):
    # Concatenate ALL python blocks (a solution may define a helper in one block
    # and the entry point in another) and dedent (stray leading indent from prose
    # framing causes IndentationError on otherwise-correct code).
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return "\n\n".join(textwrap.dedent(b) for b in blocks)
    return text  # model ignored the fence instruction; run the raw text


def _exec_once(program, timeout):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
        f.write(program)
        f.flush()
        try:
            p = subprocess.run([sys.executable, f.name], capture_output=True,
                               timeout=timeout, text=True)
            return p.returncode == 0, ((p.stderr or "").strip().splitlines()[-1:] or [""])[0]
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:  # noqa: BLE001
            return False, f"RUNNER:{e}"


def run_test(code, test, entry_point, prompt="", timeout=15):
    # Dual-strategy assembly: a returned solution may be a FULL function
    # (self-contained) or a BODY-ONLY completion (needs the prompt's signature +
    # given helpers). Count a pass if EITHER assembly executes - the model wrote
    # correct code, so harness assembly choice shouldn't decide pass/fail.
    suffix = f"\n\n{test}\n\ncheck({entry_point})\n"
    ok, why = _exec_once(f"{prompt}\n\n{code}{suffix}", timeout)   # prompt + code
    if ok:
        return True, why
    ok2, why2 = _exec_once(f"{code}{suffix}", timeout)            # code only
    return (ok2, why2 if not ok2 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:57455")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/tmp/he.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all problems")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--think", action="store_true",
                    help="enable the model's reasoning channel (renders the chat "
                         "template with enable_thinking=True via /v1/completions)")
    ap.add_argument("--template", default=None,
                    help="path to chat_template.jinja (required with --think)")
    ap.add_argument("--out", default="/tmp/coding_eval_results.json")
    args = ap.parse_args()
    if args.think and not args.template:
        sys.exit("--think requires --template <path to chat_template.jinja>")

    problems = [json.loads(l) for l in open(args.data)]
    if args.limit:
        problems = problems[: args.limit]

    PROMPT = ("Complete the following Python function. Return ONLY the complete "
              "function in a single ```python code block, including any necessary "
              "imports.\n\n```python\n{p}\n```")

    n = len(problems)
    passed = 0
    tok_sum = 0.0
    time_sum = 0.0
    results = []
    t_start = time.time()
    for i, prob in enumerate(problems, 1):
        try:
            msg, ct, dt = chat(args.url, args.model, PROMPT.format(p=prob["prompt"]),
                               args.temp, args.max_tokens, think=args.think,
                               template_path=args.template)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{n}] {prob['task_id']}  REQUEST-ERR {e}")
            results.append({"task": prob["task_id"], "pass": False, "err": str(e)})
            continue
        code = extract_code(msg)
        ok, why = run_test(code, prob["test"], prob["entry_point"], prompt=prob["prompt"])
        passed += ok
        tok_sum += ct
        time_sum += dt
        results.append({"task": prob["task_id"], "pass": ok, "tokens": ct,
                        "why": ("" if ok else why), "code": code, "raw": msg})
        rate = ct / dt if dt > 0 else 0
        print(f"[{i}/{n}] {prob['task_id']:14s} {'PASS' if ok else 'FAIL':4s} "
              f"{ct:4d} tok @ {rate:5.1f} tok/s"
              f"{'' if ok else '  <- ' + str(why)[:60]}")

    wall = time.time() - t_start
    agg_rate = tok_sum / time_sum if time_sum else 0
    summary = {
        "model": args.model, "n": n, "passed": passed,
        "pass_at_1": round(100 * passed / n, 1),
        "agg_decode_tok_s": round(agg_rate, 1),
        "total_completion_tokens": int(tok_sum),
        "wall_s": round(wall, 1),
    }
    print("\n=== HumanEval pass@1 (standalone) ===")
    for k, v in summary.items():
        print(f"  {k:24s} {v}")
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
