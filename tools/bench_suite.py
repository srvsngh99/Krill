#!/usr/bin/env python3
"""Krill vs Ollama capability+latency benchmark suite (text / vision / voice /
tools, hot + cold). Sequential by design — it hits ONE engine at a time so the
GPU/RAM is never contended, which keeps the numbers honest.

It reads each engine's own Ollama-compatible timing block (`prompt_eval_*`,
`eval_*`, `load_duration`) so the comparison is apples-to-apples. Krill cold
numbers come from the native one-shot CLI (`krill run`), which loads the model
fresh and prints its own load/prefill/decode/TTFT; Ollama cold comes from
`ollama stop <model>` followed by a timed request (its `load_duration`).

This is a companion to:
  - tools/krill_concurrent_benchmark.py  (the N-stream concurrency sweep)
  - tools/gemma4_multimodal_benchmark.py  (the release-gate multimodal harness)
Unlike those, this one is the quick "show me the head-to-head numbers" driver
used to refresh docs/BENCHMARKS.md.

Prereqs:
  - Krill server up:  krill serve --model <m>   (default port 57455; hot numbers)
  - Ollama up on 11434 with the matching GGUF model pulled
  - .build/release/krill built (make release)            (Krill cold)
  - Pillow in the active venv ONLY if you let it auto-generate the image asset

Usage:
  python3 tools/bench_suite.py --axis all \
      --krill-model gemma-4-e2b --ollama-model gemma4:e2b
  python3 tools/bench_suite.py --axis text --krill-model llama-3.2-3b --ollama-model llama3.2:3b
"""
import argparse, base64, json, os, re, statistics, subprocess, sys, time, urllib.request

KRILL_URL = "http://127.0.0.1:57455"   # Krill default ("KRILL" on a keypad)
OLLAMA_URL = "http://127.0.0.1:11434"  # Ollama default
KRILL_BIN = ".build/release/krill"


def post(url, path, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url + path, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, (time.perf_counter() - t0) * 1000


def server_metrics(body, wall_ms):
    ns = 1e9
    pe_c, pe_d = body.get("prompt_eval_count") or 0, body.get("prompt_eval_duration") or 0
    e_c, e_d = body.get("eval_count") or 0, body.get("eval_duration") or 0
    return {
        "prefill_tps": round(pe_c / (pe_d / ns), 1) if pe_d else None,
        "decode_tps": round(e_c / (e_d / ns), 1) if e_d else None,
        "prefill_count": pe_c, "eval_count": e_c,
        "load_ms": round((body.get("load_duration") or 0) / 1e6, 1),
        "total_ms": round((body.get("total_duration") or 0) / 1e6, 1),
        "wall_ms": round(wall_ms, 1),
    }


def hot(url, path, payload, runs=3, warmup=1):
    """Warm, then median over `runs` (batch/timing noise -> median, not mean)."""
    for _ in range(warmup):
        post(url, path, payload)
    samples = [server_metrics(*post(url, path, payload)) for _ in range(runs)]
    def med(k):
        v = [s[k] for s in samples if isinstance(s.get(k), (int, float))]
        return round(statistics.median(v), 1) if v else None
    out = {k: med(k) for k in ["prefill_tps", "decode_tps", "total_ms", "wall_ms"]}
    out["eval_count"] = samples[-1]["eval_count"]
    out["prefill_count"] = samples[-1]["prefill_count"]
    return out


def krill_cli_cold(model, prompt, np_, image=None, audio=None, cwd="."):
    """One-shot CLI = cold load + first inference; parse its printed metrics.

    `KRILL_NO_AUTO_DAEMON=1` is REQUIRED: with a server running for the hot
    numbers, `krill run` otherwise auto-routes the prompt through that warm
    daemon (printing `--- (via daemon @ :PORT) N chunks ...`) instead of doing a
    real cold in-process load, so the cold metrics never appear and every field
    parses to None. The flag forces the in-process load + the
    `Ready (Xs load time)` / `decode: N tokens at X tok/s` / `total: Xs` lines.
    """
    cmd = [KRILL_BIN, "run", model, prompt, "--max-tokens", str(np_), "--temp", "0"]
    if image:
        cmd += ["--image", image]
    if audio:
        cmd += ["--audio", audio]
    env = {**os.environ, "KRILL_NO_AUTO_DAEMON": "1"}
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300, env=env)
    out = p.stdout + p.stderr
    if "(via daemon @" in out:
        print("  [cold] WARNING: krill run routed through the daemon despite "
              "KRILL_NO_AUTO_DAEMON=1 - cold metrics will be missing.")
    g = lambda rx: (re.search(rx, out) or [None, None])
    load = re.search(r"Ready \(([\d.]+)s load time\)", out)
    pre = re.search(r"prefill: ([\d.]+) tok/s", out)
    dec = re.search(r"decode: \d+ tokens at ([\d.]+) tok/s", out)
    ttft = re.search(r"TTFT: ([\d.]+)ms", out)
    tot = re.search(r"total: ([\d.]+)s", out)
    answer = " ".join(l for l in p.stdout.splitlines()
                      if l.strip() and not l.startswith(("Loading", "Ready", "prompt:", "---")))
    if not (load and dec and tot):
        # Surface WHY instead of silently emitting None cells: a non-zero exit,
        # a timeout, or an output format the regexes no longer match.
        print(f"  [cold] WARNING: could not parse cold metrics (exit={p.returncode}); "
              f"tail: {out[-160:]!r}")
    return {
        "load_ms": round(float(load.group(1)) * 1000, 1) if load else None,
        "prefill_tps": float(pre.group(1)) if pre else None,
        "decode_tps": float(dec.group(1)) if dec else None,
        "ttft_ms": float(ttft.group(1)) if ttft else None,
        "total_ms": round(float(tot.group(1)) * 1000, 1) if tot else None,
        "answer": answer[:100],
    }


def ollama_cold(model, path, payload):
    subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)
    time.sleep(1.0)
    return server_metrics(*post(OLLAMA_URL, path, payload))


def b64file(p):
    return base64.b64encode(open(p, "rb").read()).decode()


# ---- axes ----------------------------------------------------------------

def bench_text(a):
    print(f"\n=== TEXT  Krill {a.krill_model} vs Ollama {a.ollama_model} (np={a.text_tokens}) ===")
    prompt = "Write a detailed paragraph about the ocean and its importance to life on Earth."
    kp = {"model": a.krill_model, "prompt": prompt, "stream": False,
          "options": {"temperature": 0, "num_predict": a.text_tokens, "seed": 0}}
    op = dict(kp, model=a.ollama_model)
    kh, oh = hot(KRILL_URL, "/api/generate", kp), hot(OLLAMA_URL, "/api/generate", op)
    print(f"  HOT  Krill: decode {kh['decode_tps']} tok/s | total {kh['total_ms']} ms")
    print(f"       Ollama : decode {oh['decode_tps']} tok/s | total {oh['total_ms']} ms")
    if kh['decode_tps'] and oh['decode_tps']:
        print(f"       >> decode {kh['decode_tps']/oh['decode_tps']:.2f}x | total {oh['total_ms']/kh['total_ms']:.2f}x faster")
    kc = krill_cli_cold(a.krill_model, prompt, a.text_tokens, cwd=a.repo)
    oc = ollama_cold(a.ollama_model, "/api/generate", op)
    print(f"  COLD Krill: load {kc['load_ms']} ms | total {kc['total_ms']} ms")
    print(f"       Ollama : load {oc['load_ms']} ms | total {oc['total_ms']} ms")


def bench_vision(a):
    print(f"\n=== VISION  {a.krill_model} vs {a.ollama_model} (image={a.image}) ===")
    prompt = "Describe this image in detail. What is the main object and its color?"
    img = b64file(a.image)
    kp = {"model": a.krill_model, "prompt": prompt, "images": [img], "stream": False,
          "options": {"temperature": 0, "num_predict": a.image_tokens, "seed": 0}}
    op = dict(kp, model=a.ollama_model)
    kh, oh = hot(KRILL_URL, "/api/generate", kp), hot(OLLAMA_URL, "/api/generate", op)
    ka, _ = post(KRILL_URL, "/api/generate", kp)
    oa, _ = post(OLLAMA_URL, "/api/generate", op)

    # Flag empty output explicitly (mirrors bench_voice): Ollama's gemma4:e2b
    # processes the image but emits no text, which makes a latency comparison on
    # that model meaningless. Use a model where the OTHER engine actually answers
    # (e.g. qwen2.5vl:3b vs Qwen2.5-VL-3B-Instruct-4bit) so the comparison is on
    # latency, not "the baseline returned nothing".
    def ans(b):
        t = (b.get("response") or "").strip()
        return repr(t[:60]) if t else "EMPTY OUTPUT (processed image, produced no text)"
    print(f"  HOT  Krill: total {kh['total_ms']} ms | answer {ans(ka)}")
    print(f"       Ollama : total {oh['total_ms']} ms | answer {ans(oa)}")
    if not (oa.get("response") or "").strip():
        print("       >> NOTE: Ollama returned empty for this model; the latency "
              "ratio below is not a fair comparison. Re-run with a model Ollama "
              "renders (e.g. --krill-model Qwen2.5-VL-3B-Instruct-4bit "
              "--ollama-model qwen2.5vl:3b).")
    if kh['total_ms'] and oh['total_ms']:
        print(f"       >> total {oh['total_ms']/kh['total_ms']:.2f}x faster")
    kc = krill_cli_cold(a.krill_model, prompt, a.image_tokens, image=a.image, cwd=a.repo)
    oc = ollama_cold(a.ollama_model, "/api/generate", op)
    print(f"  COLD Krill: load {kc['load_ms']} ms | total {kc['total_ms']} ms")
    print(f"       Ollama : load {oc['load_ms']} ms | total {oc['total_ms']} ms")


def bench_voice(a):
    print(f"\n=== VOICE  {a.krill_model} vs {a.ollama_model} (audio={a.audio}) ===")
    prompt = "Transcribe or describe what is said in this audio."
    kc = krill_cli_cold(a.krill_model, prompt, a.audio_tokens, audio=a.audio, cwd=a.repo)
    print(f"  Krill (native CLI): decode {kc['decode_tps']} tok/s | TTFT {kc['ttft_ms']} ms")
    print(f"    transcript: {repr(kc['answer'])}")
    # Ollama: feed audio via the media field; record whether it returns text.
    op = {"model": a.ollama_model, "prompt": prompt, "images": [b64file(a.audio)],
          "stream": False, "options": {"temperature": 0, "num_predict": a.audio_tokens}}
    ob, _ = post(OLLAMA_URL, "/api/generate", op)
    txt = (ob.get("response") or "").strip()
    print(f"  Ollama: {'transcript: ' + repr(txt[:80]) if txt else 'EMPTY OUTPUT (ingested audio, produced no text)'}")


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "add", "description": "Add two numbers",
     "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the web",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]
CASES = [
    ("Weather in Berlin?", "get_weather"),
    ("What is 14 plus 28?", "add"),
    ("Search the web for the tallest mountain.", "search_web"),
    ("Weather in Cairo and what is 5 plus 9?", None),
]


def bench_tools(a):
    print(f"\n=== TOOLS  {a.krill_model} vs {a.ollama_model} (single-shot, scored) ===")
    def score(url, model):
        valid = lat = 0
        lats = []
        for q, exp in CASES:
            p = {"model": model, "messages": [{"role": "user", "content": q}], "tools": TOOLS,
                 "stream": False, "options": {"temperature": 0, "num_predict": 120, "seed": 0}}
            t0 = time.perf_counter()
            b, _ = post(url, "/api/chat", p)
            lats.append((time.perf_counter() - t0) * 1000)
            calls = (b.get("message") or {}).get("tool_calls") or []
            if calls and all(c.get("function", {}).get("name") for c in calls):
                valid += 1
        return valid, round(statistics.median(lats))
    kv, kl = score(KRILL_URL, a.krill_model)
    ov, ol = score(OLLAMA_URL, a.ollama_model)
    print(f"  Krill: valid_tool_call {kv}/{len(CASES)} | median latency {kl} ms")
    print(f"  Ollama : valid_tool_call {ov}/{len(CASES)} | median latency {ol} ms")


def main():
    p = argparse.ArgumentParser(description="Krill vs Ollama text/vision/voice/tools bench.")
    p.add_argument("--axis", choices=["text", "vision", "voice", "tools", "all"], default="all")
    p.add_argument("--krill-model", default="gemma-4-e2b")
    p.add_argument("--ollama-model", default="gemma4:e2b")
    p.add_argument("--image", default="/tmp/klmbench/red.png")
    p.add_argument("--audio", default="/tmp/klmbench/speech.wav")
    p.add_argument("--text-tokens", type=int, default=64)
    p.add_argument("--image-tokens", type=int, default=48)
    p.add_argument("--audio-tokens", type=int, default=50)
    p.add_argument("--repo", default=".", help="Krill repo dir (for the CLI cold path).")
    a = p.parse_args()
    print(f"Krill {KRILL_URL} ({a.krill_model})  vs  Ollama {OLLAMA_URL} ({a.ollama_model})")
    print("Sequential: one engine at a time. Hot = warm server median-of-3; "
          "cold = fresh model load (Krill CLI / `ollama stop`).")
    axes = ["text", "vision", "voice", "tools"] if a.axis == "all" else [a.axis]
    for ax in axes:
        try:
            globals()[f"bench_{ax}"](a)
        except Exception as e:  # noqa: BLE001 — one axis failing shouldn't abort the rest
            print(f"  [{ax}] skipped: {e}")
    print("\nFor the concurrency sweep (the 'scales under load' axis), run:\n"
          "  make bench-concurrent  (or tools/krill_concurrent_benchmark.py)")


if __name__ == "__main__":
    sys.exit(main())
