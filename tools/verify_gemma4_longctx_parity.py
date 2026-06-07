#!/usr/bin/env python3
"""Generate the mlx-vlm reference for the Gemma 4 long-context parity gate.

Writes `reference_logits.json` into the gemma-4-e2b model dir: the exact prompt
token ids (a >1024-token VARIED RAG context, well past 2x the 512 sliding
window), the prefill last-token logits/argmax, and a 32-step greedy
continuation. `Gemma4LongCtxParityTests` then replays those EXACT ids through
the native KrillLM runtime and asserts the long-context decode tracks this
reference - the regression gate for the KV-shared decode RoPE-offset bug
(shared-layer Q must rotate at the donor's true position, not 0).

Why this exists: the native gemma4 is a custom arch; mlx-lm has only gemma3.
The canonical reference is mlx-vlm's `models/gemma4`. Two non-obvious setup
points, both load-bearing:
  * the mlx-community/gemma-4-e2b-it-4bit checkpoint ships k/v weights for the
    KV-shared layers that the model legitimately ignores -> load strict=False;
  * its tokenizer has no chat_template and does NOT add BOS -> prepend BOS (id
    2) and the <start_of_turn> wrapping by hand, or the model degenerates.

Usage:
  python3 -m venv /tmp/gemma4ref && /tmp/gemma4ref/bin/pip install mlx-vlm
  /tmp/gemma4ref/bin/python tools/verify_gemma4_longctx_parity.py \
      --model ~/.krillm/models/blobs/gemma-4-e2b
  KLM_GEMMA4_PARITY_DIR=~/.krillm/models/blobs/gemma-4-e2b \
      swift test --filter Gemma4LongCtxParityTests
"""
import argparse
import json
import os

import mlx.core as mx
import mlx.nn as nn

# The checkpoint carries 140 extra (KV-shared) k/v params the model ignores.
_orig = nn.Module.load_weights
nn.Module.load_weights = lambda self, w, strict=True: _orig(self, w, strict=False)

from mlx_vlm import load  # noqa: E402

CONTEXT_SENTENCES = [
    "KrillLM is a native Swift and MLX inference engine for Apple Silicon.",
    "It serves text, vision, audio, embeddings, rerankers, and tool calling.",
    "One structured-output feature it provides is grammar-constrained JSON decoding.",
    "Its continuous batcher serves many concurrent decode rows from a single weight read.",
    "KrillLM shares prefix KV cache across requests to avoid re-prefilling shared context.",
    "The project aims to be a drop-in Ollama replacement on macOS.",
    "It runs entirely on Apple Silicon using the MLX array framework and Metal.",
    "Cold model load and total request latency are among its measured wins over Ollama.",
    "Vision and voice are handled by native Swift pipelines, not a Python bridge.",
    "Gemma 4, Llama 3.x, Qwen 2.5/3, Mistral, and Phi families run natively.",
    "Tool calling uses per-family adapters that emit the model's native call format.",
    "Speculative decoding includes an opt-in n-gram prompt-lookup path for repetitive output.",
]


def build_prompt(units=60):
    ctx = " ".join(CONTEXT_SENTENCES[i % len(CONTEXT_SENTENCES)] for i in range(units)) + " "
    return (f'{ctx}\n\nUsing the context above, answer as JSON {{"answer": string}}.\n'
            f'Question: What hardware does KrillLM target?\nJSON:')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="gemma-4-e2b model dir")
    ap.add_argument("--units", type=int, default=60, help="context sentences (>~17 -> past 1024 tok)")
    ap.add_argument("--max-tokens", type=int, default=32)
    a = ap.parse_args()
    model_dir = os.path.expanduser(a.model)

    model, proc = load(model_dir)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    lm = model.language_model

    prompt = build_prompt(a.units)
    ids = [2] + tok.encode("<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n")

    logits = lm(mx.array([ids])).logits[0, -1]
    mx.eval(logits)
    last = logits.tolist()
    argmax = int(mx.argmax(logits).item())

    cache = lm.make_cache()
    lg = lm(mx.array([ids]), cache=cache).logits[:, -1, :]
    greedy = []
    for _ in range(a.max_tokens):
        nt = int(mx.argmax(lg[0]).item())
        greedy.append(nt)
        if nt in (1, 106, 50):
            break
        lg = lm(mx.array([[nt]]), cache=cache).logits[:, -1, :]

    out = os.path.join(model_dir, "reference_logits.json")
    json.dump(dict(tokens=ids, vocab_size=len(last), last_token_logits=last,
                   argmax=argmax, greedy_ids=greedy), open(out, "w"))
    print(f"wrote {out}: ptok={len(ids)} argmax={argmax} greedy={tok.decode(greedy)[:60]!r}")


if __name__ == "__main__":
    main()
