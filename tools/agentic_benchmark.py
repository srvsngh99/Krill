#!/usr/bin/env python3
"""Agentic-workload benchmark: KrillLM vs Ollama on the axis the market is moving
to, not single-user short chat.

A realistic agent/RAG request = a long SHARED context (system prompt + tool
schema + retrieved docs, reused across many calls) + a short varying question +
structured (JSON) output, served to MANY concurrent users. This is where
KrillLM's continuous batcher + shared prefix cache + grammar-constrained decoding
beat Ollama's per-slot, serialized model — and where 'miles ahead' is achievable
(single-stream short chat is RAM-bandwidth-bound parity; see docs/BENCHMARKS.md).

Measures, per concurrency level N: aggregate decode tok/s, p50/p99 TTFT
(end-to-end first token, the latency a user feels), tasks/sec, and JSON validity.
Sequential between engines (one engine's sweep finishes before the other starts).

Usage:
  python3 tools/agentic_benchmark.py --krill-model qwen2.5-3b --ollama-model qwen2.5:3b \
      --concurrency 1,4,8 --context-repeat 60
"""
import argparse, json, statistics, threading, time, urllib.request

KRILL = "http://127.0.0.1:57455"   # KrillLM default ("KRILL" on a keypad)
OLLAMA = "http://127.0.0.1:11434"  # Ollama default

CONTEXT_UNIT = (
    "KrillLM is a native Swift and MLX inference engine for Apple Silicon. It "
    "serves text, vision, audio, embeddings, rerankers, tool calling, and "
    "grammar-constrained structured output. Its continuous batcher serves many "
    "concurrent decode rows from a single weight read, and it shares prefix KV "
    "across requests. ")
QUESTIONS = [
    "What hardware does KrillLM target?", "List two modalities it supports.",
    "What does the continuous batcher do?", "Does it support tool calling?",
    "Name one structured-output feature.", "What language is it written in?",
    "Does it run on Apple Silicon?", "What is one thing it shares across requests?",
]


def stream_request(url, model, prompt, max_tokens, want_json):
    """Fire one streaming /api/generate; return (ttft_ms, total_ms, n_tokens, text)."""
    payload = {"model": model, "prompt": prompt, "stream": True,
               "options": {"temperature": 0, "num_predict": max_tokens, "seed": 0}}
    if want_json:
        payload["format"] = "json"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url + "/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n = 0
    chunks = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            piece = obj.get("response", "")
            if piece:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                n += 1
                chunks.append(piece)
            if obj.get("done"):
                break
    total = (time.perf_counter() - t0) * 1000
    return ttft or total, total, n, "".join(chunks)


def run_concurrency(url, model, n, max_tokens, want_json, context):
    """Fire N concurrent agentic requests (shared context, varied question)."""
    results = [None] * n
    def worker(i):
        q = QUESTIONS[i % len(QUESTIONS)]
        prompt = f"{context}\n\nUsing the context above, answer as JSON " \
                 f"{{\"answer\": string}}.\nQuestion: {q}\nJSON:"
        results[i] = stream_request(url, model, prompt, max_tokens, want_json)
    wall0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall0
    ttfts = [r[0] for r in results if r]
    toks = sum(r[2] for r in results if r)
    valid = 0
    for r in results:
        if not r:
            continue
        try:
            json.loads(r[3]); valid += 1
        except Exception:
            pass
    def pct(xs, q):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))], 0) if xs else None
    return {
        "agg_tps": round(toks / wall, 1) if wall else 0,
        "p50_ttft": pct(ttfts, 0.5), "p99_ttft": pct(ttfts, 0.99),
        "tasks_per_s": round(n / wall, 2), "json_valid": f"{valid}/{n}",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--krill-model", default="qwen2.5-3b")
    p.add_argument("--ollama-model", default="qwen2.5:3b")
    p.add_argument("--concurrency", default="1,4,8")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--context-repeat", type=int, default=60,
                   help="Repeat the context unit N times (~25 tok each).")
    p.add_argument("--no-json", action="store_true", help="Disable structured-output mode.")
    a = p.parse_args()
    context = CONTEXT_UNIT * a.context_repeat
    sweep = [int(x) for x in a.concurrency.split(",")]
    want_json = not a.no_json
    ctx_tok = len(context.split())
    print(f"Agentic workload: shared context ~{ctx_tok} words + JSON output "
          f"({'on' if want_json else 'off'}), max_tokens={a.max_tokens}")
    print("Warming each engine's prefix cache on the shared context first.\n")

    for name, url, model in [("KrillLM", KRILL, a.krill_model), ("Ollama", OLLAMA, a.ollama_model)]:
        # Warm the shared prefix once.
        stream_request(url, model, context + "\n\nQuestion: warmup\nJSON:", 8, want_json)
        print(f"=== {name} ({model}) ===")
        print(f"  {'N':>2} | {'agg tok/s':>9} | {'p50 TTFT':>9} | {'p99 TTFT':>9} | {'tasks/s':>7} | json")
        for n in sweep:
            r = run_concurrency(url, model, n, a.max_tokens, want_json, context)
            print(f"  {n:>2} | {r['agg_tps']:>9} | {str(r['p50_ttft'])+' ms':>9} | "
                  f"{str(r['p99_ttft'])+' ms':>9} | {r['tasks_per_s']:>7} | {r['json_valid']}")
        print()


if __name__ == "__main__":
    main()
