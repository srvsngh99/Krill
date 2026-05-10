#!/usr/bin/env python3
"""Long-running mlx-vlm sidecar for KrillLM.

Reads line-delimited JSON requests on stdin and writes line-delimited JSON
responses on stdout. The model is loaded once at startup and reused across
requests to avoid paying the Python + mlx_vlm import cost on every call.

Protocol:
  Request:  {"id": "<str>", "prompt": "...", "max_tokens": N,
             "image_path": "<path or null>", "audio_path": "<path or null>"}
  Response: {"id": "<same id>", "ok": true,  "output": "..."}
        or  {"id": "<same id>", "ok": false, "error": "<msg>"}

stderr is used for human-readable logging only. A single line "READY" is
written to stderr after model load completes so the caller can detect
readiness.
"""

import argparse
import json
import sys
import traceback


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def write_response(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def build_prompt(processor, user_prompt: str, image_path, audio_path) -> str:
    tok = processor.tokenizer
    bos = tok.decode([2])
    turn_start = tok.decode([105])
    turn_end = tok.decode([106])
    newline = tok.decode([107])
    media_prefix = ""
    if image_path:
        media_prefix += "<|image|>"
    if audio_path:
        media_prefix += "<|audio|>"
    return (
        f"{bos}{turn_start}user{newline}"
        f"{media_prefix}{user_prompt}"
        f"{turn_end}{newline}{turn_start}model{newline}"
    )


def handle_request(model, processor, generate_fn, req: dict) -> dict:
    rid = req.get("id", "")
    try:
        prompt = req.get("prompt", "")
        max_tokens = int(req.get("max_tokens", 512))
        image_path = req.get("image_path")
        audio_path = req.get("audio_path")
        full_prompt = build_prompt(processor, prompt, image_path, audio_path)
        kwargs = {}
        if image_path:
            kwargs["image"] = [image_path]
        if audio_path:
            kwargs["audio"] = audio_path
        result = generate_fn(
            model,
            processor,
            prompt=full_prompt,
            max_tokens=max_tokens,
            verbose=False,
            **kwargs,
        )
        text = result.text if hasattr(result, "text") else str(result)
        return {"id": rid, "ok": True, "output": text.strip()}
    except Exception as exc:
        return {
            "id": rid,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="KrillLM mlx-vlm sidecar")
    ap.add_argument("--model-path", required=True, help="Path to model directory")
    args = ap.parse_args()

    log(f"loading model: {args.model_path}")
    try:
        from mlx_vlm import load, generate as generate_fn
    except Exception as exc:  # pragma: no cover - import failure path
        log(f"FATAL: cannot import mlx_vlm: {exc}")
        return 2

    try:
        model, processor = load(args.model_path)
    except Exception as exc:
        log(f"FATAL: model load failed: {exc}")
        log(traceback.format_exc())
        return 3

    log("READY")

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            write_response({"id": "", "ok": False, "error": f"invalid JSON: {exc}"})
            continue
        resp = handle_request(model, processor, generate_fn, req)
        write_response(resp)

    log("EOF on stdin; exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
