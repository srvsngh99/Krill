#!/usr/bin/env python3
"""Stdin/stdout sidecar bridge for mixture-of-experts text models via mlx-lm.

KrillLM does not yet ship a native Swift+MLX router + expert-FFN
dispatch (WS6 native runtime is a follow-up). This bridge keeps
MoE models usable as a compatible-fallback tier: KrillLM spawns
the bridge once per session, feeds it text requests over stdin,
and reads streamed token text from stdout.

Protocol (one JSON object per line, identical shape to the WS5
qwen25vl_bridge so the Swift side can share its LineReader /
JSON-frame plumbing):
    request:  { "id": <int>, "max_tokens": int,
                "messages": [ {"role": "system|user|assistant",
                               "content": str}, ... ] }
        (the legacy `prompt: str` form is accepted as a shorthand)
    response: { "id": <int>, "token": str }    (full text in one frame)
                { "id": <int>, "done": true,
                  "prompt_tokens": int, "completion_tokens": int }
                { "id": <int>, "error": str }  (on failure)

The bridge loads the model once on startup (path is the single CLI
argument). Fail-fast: any mlx-lm exception terminates the bridge so
KrillLM observes an EOF instead of stuck state.

mlx-lm transparently handles router + expert dispatch for known
MoE architectures (Mixtral, Qwen3-MoE, Qwen2-MoE, OLMoE), so this
bridge is intentionally architecture-agnostic - the same protocol
works for any model mlx-lm can load. KrillLM's family-detection
arm decides which models route here.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any


def emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to the mlx-community MoE checkpoint")
    args = ap.parse_args()

    try:
        from mlx_lm import load, generate  # type: ignore[import-not-found]
        # Apply chat templates are also needed for chat-style turns.
        from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: F401
    except ImportError as e:
        emit({"error": f"mlx-lm not installed in this venv: {e}"})
        return 1

    try:
        model, tokenizer = load(args.model)
    except Exception as e:
        emit({"error": f"mlx-lm load failed: {e}\n{traceback.format_exc()}"})
        return 1

    emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id", 0)
            messages = req.get("messages")
            if not messages:
                text = req.get("prompt", "")
                messages = [{"role": "user", "content": text}]
            max_tokens = int(req.get("max_tokens", 256))
        except Exception as e:
            emit({"error": f"invalid request: {e}"})
            continue

        try:
            # Most MoE chat tunes ship a working chat template;
            # apply it via the tokenizer. Falls back to a plain
            # join when the tokenizer has no chat template
            # configured.
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False)
            except Exception:
                prompt = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in messages)
                prompt += "\nassistant:"
            text_out = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=max_tokens, verbose=False)
            # mlx-lm's generate() returns the raw completion text.
            # Token counts are not in the return; approximate via the
            # tokenizer for usage reporting (cheap relative to
            # generate itself).
            try:
                prompt_tokens = len(tokenizer.encode(prompt))
            except Exception:
                prompt_tokens = max(1, len(prompt) // 4)
            try:
                completion_tokens = len(tokenizer.encode(text_out))
            except Exception:
                completion_tokens = max(1, len(text_out) // 4)
            emit({"id": req_id, "token": text_out})
            emit({
                "id": req_id, "done": True,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })
        except Exception as e:
            emit({
                "id": req_id, "error": f"generate failed: {e}",
            })
    return 0


if __name__ == "__main__":
    sys.exit(main())
