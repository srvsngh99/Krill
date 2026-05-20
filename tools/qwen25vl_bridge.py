#!/usr/bin/env python3
"""Stdin/stdout sidecar bridge for Qwen 2.5-VL inference via mlx-vlm.

KrillLM does not yet ship a native Swift+MLX vision tower / mRoPE
implementation for Qwen 2.5-VL (WS5 native runtime is a follow-up).
This bridge keeps the model usable as a compatible-fallback tier:
KrillLM spawns the bridge once per session, feeds it (text, image)
requests over stdin, and reads streamed token text from stdout.

Protocol (one JSON object per line):
    request:  { "id": <int>, "prompt": str, "image_path": str|null,
                "max_tokens": int }
    response: { "id": <int>, "token": str }    (streamed)
                { "id": <int>, "done": true,
                  "prompt_tokens": int, "completion_tokens": int }
                { "id": <int>, "error": str }  (on failure)

The bridge process loads the model once on startup (the path is the
single CLI argument). It is fail-fast: any mlx-vlm exception
terminates the bridge so KrillLM observes an EOF instead of stuck
state. KrillLM's RerankEngine-style retry-once policy applies.

This is a deliberate `compatible_fallback`-tier path - it is slower
than a native Swift+MLX implementation would be and adds a Python
process to the runtime. The native port (custom vision tower,
patch merger, 3D mRoPE, masked embedding injection) is tracked as
the follow-up to this PR.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any


def emit(obj: dict[str, Any]) -> None:
    """Write one JSON object to stdout, flushed immediately."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def build_prompt(text: str, has_image: bool) -> str:
    """Construct the Qwen 2.5-VL chat template manually.

    The version of mlx-vlm we ship against (0.5.0) has a Jinja
    template bug for content-list prompts; we sidestep it by
    writing the raw chat sequence using the official Qwen 2.5-VL
    special tokens. This is the same shape `mlx_vlm.prompt_utils`
    constructs internally when the template works.
    """
    parts = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"]
    parts.append("<|im_start|>user\n")
    if has_image:
        parts.append("<|vision_start|><|image_pad|><|vision_end|>")
    parts.append(text)
    parts.append("<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to the mlx-community Qwen 2.5-VL checkpoint")
    args = ap.parse_args()

    # Defer mlx-vlm import until the bridge actually starts so a
    # bridge-spawn failure does not stall KrillLM with import-time
    # MLX kernel compilation logs.
    try:
        from mlx_vlm import load, generate  # type: ignore[import-not-found]
    except ImportError as e:
        emit({"error": f"mlx-vlm not installed in this venv: {e}"})
        return 1

    try:
        model, processor = load(args.model)
    except Exception as e:
        emit({"error": f"mlx-vlm load failed: {e}\n{traceback.format_exc()}"})
        return 1

    emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id", 0)
            text = req.get("prompt", "")
            image_path = req.get("image_path")
            max_tokens = int(req.get("max_tokens", 256))
        except Exception as e:
            emit({"error": f"invalid request: {e}"})
            continue

        try:
            prompt = build_prompt(text, has_image=bool(image_path))
            kwargs: dict[str, Any] = {"max_tokens": max_tokens, "verbose": False}
            if image_path:
                kwargs["image"] = [image_path]
            # mlx-vlm.generate is non-streaming today; return the full
            # text and emit it as a single token chunk plus a `done`
            # frame. Streaming token-by-token would require driving
            # `stream_generate` which has the same chat-template bug
            # in 0.5.0; left as a follow-up once mlx-vlm is upgraded.
            result = generate(model, processor, prompt, **kwargs)
            text_out = result.text if hasattr(result, "text") else str(result)
            usage = getattr(result, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            if completion_tokens == 0:
                # Approximate: 1 token ~ 4 chars for the response.
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
