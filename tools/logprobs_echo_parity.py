#!/usr/bin/env python3
"""Numeric parity check for `/v1/completions` `echo` + `logprobs` (Phase 3,
docs/LOGPROBS_PLAN.md §3.2/§5.4): Krill's reported PROMPT-token logprobs vs
mlx_lm computed directly from the same checkpoint's raw forward-pass logits.

Design (matches `InferenceEngine.echoPromptLogprobs`'s own documented
choices, see its doc comment in Sources/KrillEngine/InferenceEngine.swift):
  - The prompt is tokenized with mlx_lm's own `tokenizer.encode(prompt)` -
    NO chat template - exactly what Krill's echo path does (raw prompt
    tokens, since /v1/completions echoes the prompt AS SENT, not whatever
    the actual completion call's chat-template wrapping produces).
  - A single full-sequence forward (`model(ids)`), log_softmax in float32 at
    every position: position i predicts token i+1. This is the one-shot
    reference; Krill computes the SAME thing in chunks through a real
    (uncached-by-PrefixCache) KVCache, so a small chunk-boundary floor is
    expected and measured the same way `logprobs_parity.py` measures its own
    "full-sequence vs incremental" floor.
  - Krill's response drops a leading BOS token from `tokens`/`token_logprobs`
    (so `tokens` joined reproduces the returned `text` exactly) but the
    FIRST reported token's logprob is still a REAL value (scored against the
    hidden BOS context, not null) - so with a BOS-prepending tokenizer, this
    script compares Krill's `tokens[0]` against mlx_lm's position-0 (BOS)
    prediction of the true first token, and every subsequent entry i against
    position i (0-indexed into the FULL id sequence including BOS).

Usage: logprobs_echo_parity.py <model_dir> <prompt> [<krill_url> <krill_model> <api_key>]

With only <model_dir> and <prompt>, this only reports mlx_lm's own
full-sequence vs incremental floor (no Krill comparison). Pass the last three
args to also fetch a live `echo=true,logprobs=5,max_tokens=0` response from a
running `krill serve` and compare against it.
"""
import json
import statistics
import sys
import urllib.request

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load


def full_sequence_logprobs(model, ids):
    inp = mx.array([ids])
    logits = model(inp).astype(mx.float32)
    logsumexp = mx.logsumexp(logits, axis=-1, keepdims=True)
    logprobs = logits - logsumexp
    mx.eval(logprobs)
    return logprobs[0]


def incremental_logprobs(model, ids):
    """Chunked-through-a-real-KVCache reference, mirroring how Krill's
    echo path forwards the prompt (one real cache, chunk by chunk) rather
    than a single batched call - the apples-to-apples floor comparison."""
    cache = make_prompt_cache(model)
    out = []
    chunk = 512
    start = 0
    while start < len(ids):
        end = min(start + chunk, len(ids))
        inp = mx.array([ids[start:end]])
        logits = model(inp, cache=cache).astype(mx.float32)
        logsumexp = mx.logsumexp(logits, axis=-1, keepdims=True)
        lp = (logits - logsumexp)[0]
        mx.eval(lp)
        out.extend([lp[i] for i in range(lp.shape[0])])
        start = end
    return out


def fetch_krill_echo(url, model, prompt, api_key, top_logprobs=5):
    req = urllib.request.Request(
        url.rstrip("/") + "/completions",
        data=json.dumps({
            "model": model, "prompt": prompt, "echo": True,
            "logprobs": top_logprobs, "max_tokens": 0,
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main():
    model_dir, prompt = sys.argv[1], sys.argv[2]
    model, tokenizer = load(model_dir)
    ids = tokenizer.encode(prompt)
    print(f"prompt {prompt!r} -> {len(ids)} raw tokens (incl. any BOS): {ids}")

    method_a = full_sequence_logprobs(model, ids)
    method_b = incremental_logprobs(model, ids)
    floor_diffs = []
    for i in range(len(ids) - 1):
        tid = ids[i + 1]
        a_lp = float(method_a[i, tid].item())
        b_lp = float(method_b[i][tid].item())
        floor_diffs.append(abs(a_lp - b_lp))
    floor_max = max(floor_diffs) if floor_diffs else 0.0
    floor_median = statistics.median(floor_diffs) if floor_diffs else 0.0
    print(f"mlx_lm intrinsic floor (full-sequence vs chunked-incremental, "
          f"{len(floor_diffs)} positions): max={floor_max:.6e} median={floor_median:.6e}")

    if len(sys.argv) < 6:
        print("(no krill_url/model/api_key given - floor-only run)")
        return

    krill_url, krill_model, api_key = sys.argv[3], sys.argv[4], sys.argv[5]
    resp = fetch_krill_echo(krill_url, krill_model, prompt, api_key)
    lp = resp["choices"][0]["logprobs"]
    krill_tokens = lp["tokens"]
    krill_logprobs = lp["token_logprobs"]
    print(f"\nKrill reported {len(krill_tokens)} prompt tokens: {krill_tokens}")
    print(f"Krill text_offset: {lp['text_offset']}")
    assert resp["choices"][0]["text"] == prompt, (
        f"echoed text {resp['choices'][0]['text']!r} != prompt {prompt!r}")

    # Krill's tokens[] has the leading BOS stripped (if the tokenizer added
    # one); map Krill's entry i back onto the FULL `ids` sequence's position
    # (i + offset), offset = len(ids) - len(krill_tokens).
    offset = len(ids) - len(krill_tokens)
    assert offset in (0, 1), f"unexpected offset {offset} (expected 0 or 1 for a BOS strip)"

    diffs = []
    worst = None
    for i, (tok, krill_lp) in enumerate(zip(krill_tokens, krill_logprobs)):
        if krill_lp is None:
            continue
        global_i = i + offset
        # position (global_i - 1) predicts token at global_i, using method_b
        # (the apples-to-apples chunked-cache comparison, same convention as
        # logprobs_parity.py's incremental floor).
        tid = ids[global_i]
        mlx_lp = float(method_b[global_i - 1][tid].item())
        d = abs(krill_lp - mlx_lp)
        diffs.append(d)
        if worst is None or d > worst[0]:
            worst = (d, i, tok, krill_lp, mlx_lp)
    if diffs:
        print(f"\nKrill vs mlx_lm chunked-incremental - prompt-token max abs diff "
              f"({len(diffs)} compared): {max(diffs):.6e}, median: {statistics.median(diffs):.6e}")
        if floor_max > 0:
            print(f"ratio to floor: {max(diffs) / floor_max:.2f}x")
        print("worst case (diff, index, token, krill_lp, mlx_lp):", worst)
    else:
        print("\nNo comparable (non-null) prompt-token logprobs found.")


if __name__ == "__main__":
    main()
