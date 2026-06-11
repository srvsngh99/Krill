#!/usr/bin/env python3
"""Long-context head-to-head: KrillLM vs Ollama, same prompt, same box.

Mirrors Tests/KLMEngineTests/Gemma4DecodeSweepTests.swift's prompt builder:
repeated ~55-token structured sections with a needle ("tangerine satellite")
planted at the START and a direct question at the end. Both engines are hit
through their Ollama-compatible POST /api/generate, so the timing fields are
read identically on both sides.

CRITICAL Ollama trap: it silently TRUNCATES the prompt to options.num_ctx
(default 2k-8k). A fast-but-truncated run is a FAILED run. We always send
num_ctx >= target+512 and verify the returned prompt_eval_count.

Run one engine at a time (24GB box):
  python3 tools/longctx_head2head.py --engine krillm --port 57455 \
      --model gemma-4-12b --ctx 2000,16000,32000,53000,99000
  python3 tools/longctx_head2head.py --engine ollama --port 11434 \
      --model gemma4:12b-mlx --ctx 2000,16000,32000,53000,99000

Rows print as they complete (crash-safe).
"""

import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request

PARA = (
    "The KrillLM engine runs large language models natively on Apple Silicon "
    "using MLX. It supports continuous batching, prefix-cache reuse, grammar "
    "constrained decoding, native tool calling, and multimodal input. The "
    'secret pass phrase for this document is "tangerine satellite".'
)

TOKENS_PER_SECTION = 55


def build_prompt(target_tokens: int, nonce: str, question: str) -> str:
    sections = max(1, target_tokens // TOKENS_PER_SECTION)
    blocks = [f"Run {nonce}. Document start. Remember the pass phrase."]
    for i in range(sections):
        blocks.append(f"Section {i}: {PARA}")
    if question == "direct":
        # ~8-token answer: validates retrieval, but the decode rate is then
        # dominated by fixed post-prefill overhead -- do NOT read tok/s here.
        blocks.append(
            "\nQuestion: What is the secret pass phrase mentioned above? Answer:"
        )
    else:
        # Long-form answer: enough decoded tokens that tok/s measures the
        # per-token decode rate. Ends with the phrase so needle still checks.
        blocks.append(
            "\nQuestion: Describe in detail, over several paragraphs, what the "
            "KrillLM engine supports and how its features help local "
            "deployment. End your answer by stating the secret pass phrase. "
            "Answer:"
        )
    return "\n".join(blocks)


class RSSampler(threading.Thread):
    """Polls the peak RSS of the engine's heaviest process during a request."""

    def __init__(self, pattern: str):
        super().__init__(daemon=True)
        self.pattern = pattern
        self.peak_mb = 0.0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["ps", "axo", "rss,command"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, timeout=5,
                ).stdout
                for line in out.splitlines()[1:]:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2 and self.pattern in parts[1] \
                            and "longctx_head2head" not in parts[1]:
                        self.peak_mb = max(self.peak_mb, int(parts[0]) / 1024)
            except Exception:
                pass
            self._stop.wait(2.0)

    def stop(self) -> float:
        self._stop.set()
        self.join(timeout=5)
        return self.peak_mb


def one_run(base: str, model: str, prompt: str, num_ctx: int,
            max_tokens: int, timeout_s: int, rss_pattern: str,
            think: bool = False) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Ollama gemma4 tags default to thinking -> the answer lands in the
        # `thinking` field and `response` comes back EMPTY. Always disable.
        "think": think,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    req = urllib.request.Request(
        f"{base}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    sampler = RSSampler(rss_pattern)
    sampler.start()
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
    except Exception as e:  # OOM / crash / timeout is itself a result
        return {"error": f"{type(e).__name__}: {e}",
                "wall_s": time.time() - t0, "rss_mb": sampler.stop()}
    wall = time.time() - t0
    rss = sampler.stop()
    pe_count = data.get("prompt_eval_count", 0)
    pe_dur = data.get("prompt_eval_duration", 0) / 1e9
    ev_count = data.get("eval_count", 0)
    ev_dur = data.get("eval_duration", 0) / 1e9
    return {
        "prompt_eval_count": pe_count,
        "prefill_s": pe_dur,
        "eval_count": ev_count,
        "decode_tps": (ev_count / ev_dur) if ev_dur > 0 else 0.0,
        "needle": "tangerine" in data.get("response", "").lower(),
        "response_head": data.get("response", "")[:80].replace("\n", " "),
        "wall_s": wall,
        "rss_mb": rss,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["krillm", "ollama"])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", default="2000,16000,32000,53000,99000")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--question", choices=["direct", "summary"], default="direct")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--nonce", default=str(int(time.time())))
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    rss_pattern = "ollama" if args.engine == "ollama" else "krillm"
    ctxs = [int(c) for c in args.ctx.split(",")]

    print(f"\n===== longctx head2head: engine={args.engine} model={args.model} "
          f"nonce={args.nonce} =====", flush=True)
    print("| target ctx | prompt_eval_count | prefill s | decode tok/s | "
          "needle | peak RSS GB | wall s | note |", flush=True)
    print("|---|---|---|---|---|---|---|---|", flush=True)

    for ctx in ctxs:
        prompt = build_prompt(ctx, f"{args.nonce}-{ctx}", args.question)
        # Actual gemma token count runs ~1.12x the section-based target
        # (measured: 110000-target -> 122921 tokens); num_ctx must cover the
        # ACTUAL count or Ollama silently truncates.
        r = one_run(base, args.model, prompt, num_ctx=int(ctx * 1.25) + 1024,
                    max_tokens=args.max_tokens, timeout_s=args.timeout,
                    rss_pattern=rss_pattern)
        if "error" in r:
            print(f"| {ctx} | - | - | - | - | "
                  f"{r['rss_mb']/1024:.2f} | {r['wall_s']:.0f} | "
                  f"FAILED: {r['error'][:90]} |", flush=True)
            continue
        # Actual tokens ~1.12x target, so anything at or below the target
        # means the engine dropped part of the prompt.
        truncated = r["prompt_eval_count"] < 1.0 * ctx
        note = "TRUNCATED!" if truncated \
            else f"ev={r['eval_count']} {r['response_head'][:48]}"
        print(f"| {ctx} | {r['prompt_eval_count']} | {r['prefill_s']:.1f} | "
              f"{r['decode_tps']:.1f} | {'y' if r['needle'] else 'N'} | "
              f"{r['rss_mb']/1024:.2f} | {r['wall_s']:.0f} | {note} |",
              flush=True)

    print("=" * 45, flush=True)


if __name__ == "__main__":
    main()
