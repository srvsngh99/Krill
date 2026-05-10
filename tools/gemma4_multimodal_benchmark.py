#!/usr/bin/env python3
"""Benchmark KrillLM Gemma4 via mlx-vlm against Ollama Gemma4.

This benchmark intentionally treats text, image, and audio as separate tasks.
It records exact quantization metadata and only labels the comparison as
4-bit-class equivalent, because MLX affine 4-bit and GGUF Q4_K_M are different
quantizers even though both are 4-bit families.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import re
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw

try:
    from mlx_vlm import generate, load  # type: ignore
    _MLX_VLM_AVAILABLE = True
    _MLX_VLM_IMPORT_ERROR: Optional[Exception] = None
except Exception as _exc:  # pragma: no cover — optional dep when only using native paths
    generate = None  # type: ignore[assignment]
    load = None  # type: ignore[assignment]
    _MLX_VLM_AVAILABLE = False
    _MLX_VLM_IMPORT_ERROR = _exc


DEFAULT_KRILL_MODEL = str(Path.home() / ".krillm/models/blobs/gemma-4-e2b")
DEFAULT_OLLAMA_MODEL = "gemma4:e2b"
DEFAULT_OUTPUT = ".build/benchmarks/gemma4-e2b-multimodal-4bit.json"

# Mirrors the regex in tools/krillm_vs_ollama_benchmark.py:parse_krillm_result
KRILLM_CLI_STATS_RE = re.compile(
    r"prompt:\s+(?P<prompt_tokens>\d+)\s+tokens,\s+"
    r"prefill:\s+(?P<prefill_tps>[0-9.]+)\s+tok/s,\s+"
    r"decode:\s+(?P<generated_tokens>\d+)\s+tokens\s+at\s+"
    r"(?P<decode_tps>[0-9.]+)\s+tok/s,\s+"
    r"TTFT:\s+(?P<ttft_ms>[0-9.]+)ms,\s+"
    r"total:\s+(?P<total_s>[0-9.]+)s"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma4 text/image/audio benchmark.")
    parser.add_argument("--krill-model", default=DEFAULT_KRILL_MODEL)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-bin", default="ollama")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--engine", choices=["both", "krillm", "ollama"], default="both")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-max-tokens", type=int, default=32)
    parser.add_argument("--image-max-tokens", type=int, default=16)
    parser.add_argument("--audio-max-tokens", type=int, default=24)
    parser.add_argument("--image", help="Optional image asset path.")
    parser.add_argument("--audio", help="Optional audio asset path.")
    parser.add_argument("--krillm-url", help="KrillLM server URL for native-path benchmarking (e.g. http://127.0.0.1:11435).")
    parser.add_argument(
        "--krillm-image-mode",
        choices=["bridge", "native_cli", "native_server"],
        default="native_cli",
        help=(
            "How to run KrillLM image (and text) tasks. 'bridge' uses the in-process "
            "mlx-vlm Python path (legacy). 'native_cli' invokes the krillm CLI binary "
            "as a subprocess (matches what users run; fastest). 'native_server' sends "
            "multimodal HTTP requests to --krillm-url. Audio always falls back to bridge."
        ),
    )
    parser.add_argument(
        "--krillm-bin",
        default=".build/release/krillm",
        help="Path to krillm CLI binary used by --krillm-image-mode native_cli.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--cache-mode",
        choices=["cold", "warm", "cache_hit", "auto"],
        default="auto",
        help=(
            "Cache labelling mode for results. 'auto' infers per-run labels "
            "from --warmup and prefix-cache heuristics. When set to an "
            "explicit value, every result is force-tagged with that label "
            "and cache_mode_source is recorded as 'explicit'."
        ),
    )
    return parser.parse_args()


def run_cmd(command: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def ensure_assets(output_path: str, image: Optional[str], audio: Optional[str]) -> dict[str, str]:
    asset_dir = Path(output_path).parent / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image) if image else asset_dir / "gemma4-red-box.png"
    if not image:
        img = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle((45, 70, 211, 190), fill=(220, 40, 40))
        draw.text((62, 108), "RED BOX", fill="white")
        img.save(image_path)

    audio_path = Path(audio) if audio else asset_dir / "gemma4-tone-5s.wav"
    if not audio:
        sample_rate = 16_000
        duration_s = 5.0
        frequency = 440.0
        with wave.open(str(audio_path), "w") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            frames = bytearray()
            for i in range(int(sample_rate * duration_s)):
                sample = int(0.35 * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
                frames += struct.pack("<h", sample)
            handle.writeframes(frames)

    return {"image": str(image_path), "audio": str(audio_path)}


def krill_quantization(model_path: str) -> dict[str, Any]:
    path = Path(model_path)
    if (path / "config.json").exists():
        config_path = path / "config.json"
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(repo_id=model_path, filename="config.json"))
        except Exception as exc:
            return {
                "model": model_path,
                "quantization": {},
                "quantization_class": "unknown",
                "metadata_error": str(exc),
            }
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant = config.get("quantization") or config.get("quantization_config") or {}
    quant_class = "bf16" if str(config.get("dtype", "")).lower() == "bfloat16" and not quant else "unknown"
    if quant.get("bits"):
        quant_class = f"{quant.get('bits')}-bit"
    return {
        "model": model_path,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "dtype": config.get("dtype"),
        "quantization": quant,
        "quantization_class": quant_class,
        "has_vision_config": "vision_config" in config,
        "has_audio_config": "audio_config" in config,
        "image_token_id": config.get("image_token_id"),
        "audio_token_id": config.get("audio_token_id"),
    }


def ollama_show(ollama_bin: str, model: str, timeout: float) -> dict[str, Any]:
    completed = run_cmd([ollama_bin, "show", model], timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    text = completed.stdout
    quant_match = re.search(r"quantization\s+(\S+)", text)
    parameter_match = re.search(r"parameters\s+(.+)", text)
    capabilities = []
    in_capabilities = False
    for line in text.splitlines():
        if line.strip() == "Capabilities":
            in_capabilities = True
            continue
        if in_capabilities:
            stripped = line.strip()
            if not stripped:
                continue
            if line.startswith("  ") and not line.startswith("    "):
                break
            capabilities.append(stripped)
    quant = quant_match.group(1) if quant_match else "unknown"
    quant_class = "bf16" if quant.upper() == "BF16" else ("f16" if quant.upper() == "F16" else ("4-bit" if quant.startswith("Q4") else quant))
    return {
        "raw": text,
        "parameters": parameter_match.group(1).strip() if parameter_match else None,
        "quantization": quant,
        "quantization_class": quant_class,
        "capabilities": capabilities,
    }


def quantization_comparison(krill_quant: Optional[dict[str, Any]], ollama_quant: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not krill_quant or not ollama_quant:
        return {
            "strict_equal": False,
            "class_equal": False,
            "label": "single-engine run",
            "note": "Only one engine was benchmarked in this report.",
        }
    strict_equal = krill_quant.get("quantization") == ollama_quant.get("quantization")
    class_equal = krill_quant.get("quantization_class") == ollama_quant.get("quantization_class")
    note = ""
    if class_equal and not strict_equal:
        note = "Precision class matches, but model formats and runtime layouts may differ."
    return {
        "strict_equal": strict_equal,
        "class_equal": class_equal,
        "label": "precision-class equivalent" if class_equal else "mismatch",
        "note": note,
    }


def build_krill_prompt(processor: Any, prompt: str, media_prefix: str = "") -> str:
    tokenizer = processor.tokenizer
    bos = tokenizer.decode([2])
    turn_start = tokenizer.decode([105])
    turn_end = tokenizer.decode([106])
    newline = tokenizer.decode([107])
    return f"{bos}{turn_start}user{newline}{media_prefix}{prompt}{turn_end}{newline}{turn_start}model{newline}"


def run_krill_task(
    model: Any,
    processor: Any,
    task: dict[str, Any],
    temperature: float,
    top_p: float,
    measured: bool,
) -> Optional[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "max_tokens": task["max_tokens"],
        "temperature": temperature,
        "top_p": top_p,
        "verbose": False,
    }
    media_prefix = ""
    if task.get("image"):
        kwargs["image"] = [task["image"]]
        media_prefix += "<|image|>"
    if task.get("audio"):
        kwargs["audio"] = task["audio"]
        media_prefix += "<|audio|>"

    prompt = build_krill_prompt(processor, task["prompt"], media_prefix)
    start = time.perf_counter()
    result = generate(model, processor, prompt=prompt, **kwargs)
    wall_s = time.perf_counter() - start
    if not measured:
        return None

    text = result.text if hasattr(result, "text") else str(result)
    return {
        "wall_time_s": wall_s,
        "prompt_tokens": getattr(result, "prompt_tokens", None),
        "generated_tokens": getattr(result, "generation_tokens", None),
        "prefill_tokens_per_second": getattr(result, "prompt_tps", None),
        "decode_tokens_per_second": getattr(result, "generation_tps", None),
        "peak_memory_gb": getattr(result, "peak_memory", None),
        "output_sha256": sha256_text(text),
        "output_preview": text[:200],
    }


def run_krill_server_task(
    args: argparse.Namespace,
    task: dict[str, Any],
    measured: bool,
) -> Optional[dict[str, Any]]:
    """Run a benchmark request against a persistent KrillLM server using /api/generate.

    Supports text, image, and audio. Image is sent as base64 in the Ollama
    `images` array; audio is sent as base64 in the `audio` field with
    `audio_format`. Audio runs through the server's mlx-vlm bridge, which
    does not stream — the server emits a single content chunk + final done
    chunk for compatibility.
    """
    import base64 as _b64

    prompt = task["prompt"]

    payload: dict[str, Any] = {
        "model": "gemma-4-e2b",
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "num_predict": task["max_tokens"],
        },
    }
    image_path = task.get("image")
    audio_path = task.get("audio")
    if image_path:
        payload["images"] = [_b64.b64encode(Path(image_path).read_bytes()).decode("ascii")]
    if audio_path:
        payload["audio"] = _b64.b64encode(Path(audio_path).read_bytes()).decode("ascii")
        ext = Path(audio_path).suffix.lstrip(".").lower() or "wav"
        payload["audio_format"] = ext
    url = args.krillm_url.rstrip("/") + "/api/generate"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_token_s: Optional[float] = None
    chunks: list[str] = []
    final: Optional[dict[str, Any]] = None
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line)
            content = event.get("response") or ""
            if content and first_token_s is None:
                first_token_s = time.perf_counter() - start
            chunks.append(content)
            if event.get("done"):
                final = event
    wall_s = time.perf_counter() - start
    if final is None:
        raise RuntimeError("KrillLM server stream ended without final stats")
    if not measured:
        return None

    prompt_eval_s = int(final.get("prompt_eval_duration") or 0) / 1_000_000_000
    eval_s = int(final.get("eval_duration") or 0) / 1_000_000_000
    prompt_tokens = int(final.get("prompt_eval_count") or 0)
    generated_tokens = int(final.get("eval_count") or 0)
    text = "".join(chunks)
    return {
        "wall_time_s": wall_s,
        "total_s": int(final.get("total_duration") or 0) / 1_000_000_000,
        "ttft_ms_wall": first_token_s * 1000 if first_token_s is not None else None,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prefill_tokens_per_second": prompt_tokens / prompt_eval_s if prompt_eval_s > 0 else None,
        "decode_tokens_per_second": generated_tokens / eval_s if eval_s > 0 else None,
        "output_sha256": sha256_text(text),
        "output_preview": text[:200],
        "mode": "server",
    }


def run_krill_native_cli_task(
    args: argparse.Namespace,
    task: dict[str, Any],
    measured: bool,
) -> Optional[dict[str, Any]]:
    """Invoke the krillm CLI binary for a (text or image) task.

    Audio is not supported by this path and must be handled by the bridge.
    """
    if task.get("audio"):
        raise RuntimeError("native_cli path does not support audio tasks")

    command: list[str] = [
        args.krillm_bin,
        "run",
        args.krill_model,
        task["prompt"],
        "--temp",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-tokens",
        str(task["max_tokens"]),
        "--seed",
        str(args.seed),
    ]
    if task.get("image"):
        command += ["--image", task["image"]]

    start = time.perf_counter()
    completed = run_cmd(command, timeout=args.timeout)
    wall_s = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"krillm CLI failed with exit {completed.returncode}: "
            f"{completed.stderr or completed.stdout}"
        )
    if not measured:
        return None

    stdout = completed.stdout
    match = KRILLM_CLI_STATS_RE.search(stdout)
    if not match:
        raise RuntimeError("krillm CLI output did not contain a parseable stats line")
    generated = stdout.split("\n---\n", 1)[0]
    generated_lines = [
        line for line in generated.splitlines()
        if line and not line.startswith("Loading model") and not line.startswith("Ready (")
    ]
    generated_text = "\n".join(generated_lines)
    return {
        "wall_time_s": wall_s,
        "total_s": float(match.group("total_s")),
        "ttft_ms_wall": float(match.group("ttft_ms")),
        "prompt_tokens": int(match.group("prompt_tokens")),
        "generated_tokens": int(match.group("generated_tokens")),
        "prefill_tokens_per_second": float(match.group("prefill_tps")),
        "decode_tokens_per_second": float(match.group("decode_tps")),
        "output_sha256": sha256_text(generated_text),
        "output_preview": generated_text[:200],
        "mode": "native_cli",
    }


def run_krill_native_server_task(
    args: argparse.Namespace,
    task: dict[str, Any],
    measured: bool,
) -> Optional[dict[str, Any]]:
    """Send a multimodal request to the KrillLM server using the Ollama /api/generate shape.

    Image bytes are sent base64-encoded in the `images` field. Audio is not supported
    on this path and must use the bridge.
    """
    if task.get("audio"):
        raise RuntimeError("native_server path does not support audio tasks")

    payload: dict[str, Any] = {
        "model": "gemma-4-e2b",
        "prompt": task["prompt"],
        "stream": True,
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "num_predict": task["max_tokens"],
        },
    }
    if task.get("image"):
        payload["images"] = [
            base64.b64encode(Path(task["image"]).read_bytes()).decode("ascii")
        ]
    url = args.krillm_url.rstrip("/") + "/api/generate"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_token_s: Optional[float] = None
    chunks: list[str] = []
    final: Optional[dict[str, Any]] = None
    try:
        response_ctx = urllib.request.urlopen(request, timeout=args.timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            raise RuntimeError(
                "KrillLM server returned 400 for multimodal request — "
                "server multimodal support has likely not landed yet "
                f"(url={url}): {exc.read().decode('utf-8', errors='replace')}"
            ) from exc
        raise
    with response_ctx as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line)
            content = event.get("response") or ""
            if content and first_token_s is None:
                first_token_s = time.perf_counter() - start
            chunks.append(content)
            if event.get("done"):
                final = event
    wall_s = time.perf_counter() - start
    if final is None:
        raise RuntimeError("KrillLM server stream ended without final stats")
    if not measured:
        return None

    prompt_eval_s = int(final.get("prompt_eval_duration") or 0) / 1_000_000_000
    eval_s = int(final.get("eval_duration") or 0) / 1_000_000_000
    prompt_tokens = int(final.get("prompt_eval_count") or 0)
    generated_tokens = int(final.get("eval_count") or 0)
    text = "".join(chunks)
    return {
        "wall_time_s": wall_s,
        "total_s": int(final.get("total_duration") or 0) / 1_000_000_000,
        "ttft_ms_wall": first_token_s * 1000 if first_token_s is not None else None,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prefill_tokens_per_second": prompt_tokens / prompt_eval_s if prompt_eval_s > 0 else None,
        "decode_tokens_per_second": generated_tokens / eval_s if eval_s > 0 else None,
        "output_sha256": sha256_text(text),
        "output_preview": text[:200],
        "mode": "native_server",
    }


def ollama_chat_payload(args: argparse.Namespace, task: dict[str, Any], stream: bool) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "user", "content": task["prompt"]}
    media_path = task.get("image") or task.get("audio")
    if media_path:
        message["images"] = [base64.b64encode(Path(media_path).read_bytes()).decode("ascii")]
    return {
        "model": args.ollama_model,
        "messages": [message],
        "stream": stream,
        "think": False,
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "num_predict": task["max_tokens"],
        },
    }


def run_ollama_task(args: argparse.Namespace, task: dict[str, Any], measured: bool) -> Optional[dict[str, Any]]:
    payload = ollama_chat_payload(args, task, stream=True)
    request = urllib.request.Request(
        args.ollama_host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_token_s: Optional[float] = None
    chunks: list[str] = []
    final: Optional[dict[str, Any]] = None
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line)
            content = ((event.get("message") or {}).get("content")) or ""
            if content and first_token_s is None:
                first_token_s = time.perf_counter() - start
            chunks.append(content)
            if event.get("done"):
                final = event
    wall_s = time.perf_counter() - start
    if final is None:
        raise RuntimeError("Ollama stream ended without final stats")
    if not measured:
        return None

    prompt_eval_s = int(final.get("prompt_eval_duration") or 0) / 1_000_000_000
    eval_s = int(final.get("eval_duration") or 0) / 1_000_000_000
    prompt_tokens = int(final.get("prompt_eval_count") or 0)
    generated_tokens = int(final.get("eval_count") or 0)
    text = "".join(chunks)
    return {
        "wall_time_s": wall_s,
        "total_s": int(final.get("total_duration") or 0) / 1_000_000_000,
        "load_ms": int(final.get("load_duration") or 0) / 1_000_000,
        "ttft_ms_wall": first_token_s * 1000 if first_token_s is not None else None,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prompt_eval_ms": prompt_eval_s * 1000,
        "prefill_tokens_per_second": prompt_tokens / prompt_eval_s if prompt_eval_s > 0 else None,
        "decode_tokens_per_second": generated_tokens / eval_s if eval_s > 0 else None,
        "output_sha256": sha256_text(text),
        "output_preview": text[:200],
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "wall_time_s",
        "total_s",
        "load_ms",
        "ttft_ms_wall",
        "prompt_tokens",
        "generated_tokens",
        "prompt_eval_ms",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "peak_memory_gb",
    ]
    summary: dict[str, Any] = {"runs": len(runs)}
    for key in keys:
        values = [run[key] for run in runs if isinstance(run.get(key), (int, float))]
        if values:
            summary[f"{key}_median"] = statistics.median(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)
    return summary


def environment() -> dict[str, Any]:
    def output_or_none(command: list[str]) -> Optional[str]:
        try:
            completed = run_cmd(command, timeout=15.0)
        except Exception:
            return None
        if completed.returncode != 0:
            return None
        return (completed.stdout or completed.stderr).strip() or None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "ollama_version": output_or_none(["ollama", "--version"]),
        "git_commit": output_or_none(["git", "rev-parse", "HEAD"]),
        "git_status": output_or_none(["git", "status", "--short"]),
    }


def infer_cache_mode(index: int, warmup: int, use_server: bool) -> str:
    if use_server and (warmup > 0 or index > 0):
        return "cache_hit"
    if warmup == 0 and index == 0:
        return "cold"
    return "warm"


def apply_cache_mode(
    runs: list[Optional[dict[str, Any]]],
    cli_mode: str,
    warmup: int,
    use_server: bool,
) -> tuple[str, str]:
    source = "explicit" if cli_mode != "auto" else "auto"
    labels: list[str] = []
    for i, run in enumerate(runs):
        if run is None:
            continue
        label = cli_mode if cli_mode != "auto" else infer_cache_mode(i, warmup, use_server)
        run["cache_mode"] = label
        labels.append(label)
    if not labels:
        group = cli_mode if cli_mode != "auto" else "warm"
    elif all(l == labels[0] for l in labels):
        group = labels[0]
    else:
        group = "mixed"
    return group, source


def media_signature(path: Optional[str]) -> Optional[dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    suffix = p.suffix.lower()
    info: dict[str, Any] = {"path": str(p), "size_bytes": p.stat().st_size}
    if suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
        try:
            with Image.open(p) as img:
                info["kind"] = "image"
                info["width"] = img.width
                info["height"] = img.height
        except Exception as exc:
            info["kind"] = "image"
            info["error"] = str(exc)
    elif suffix == ".wav":
        try:
            with wave.open(str(p), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate() or 1
                info["kind"] = "audio"
                info["frames"] = frames
                info["sample_rate"] = rate
                info["duration_s"] = frames / rate
        except Exception as exc:
            info["kind"] = "audio"
            info["error"] = str(exc)
    else:
        info["kind"] = "other"
    return info


def parity_field(
    krillm_runs: list[dict[str, Any]],
    ollama_runs: list[dict[str, Any]],
    prompt: str,
    media_path: Optional[str],
    media_kind: Optional[str],
) -> dict[str, Any]:
    def median_tokens(runs: list[dict[str, Any]]) -> Optional[float]:
        values = [r["prompt_tokens"] for r in runs if r and isinstance(r.get("prompt_tokens"), (int, float)) and r["prompt_tokens"]]
        if not values:
            return None
        return statistics.median(values)

    krill_tokens = median_tokens(krillm_runs)
    ollama_tokens = median_tokens(ollama_runs)
    if krill_tokens and ollama_tokens:
        basis = "prompt_tokens"
        a, b = krill_tokens, ollama_tokens
    else:
        basis = "prompt_chars"
        a = b = float(len(prompt))

    prompt_status = "ok"
    delta_ratio = 0.0
    if max(a, b) > 0:
        delta_ratio = abs(a - b) / max(a, b)
        if delta_ratio > 0.10:
            prompt_status = "mismatch"

    result: dict[str, Any] = {
        "prompt": {
            "status": prompt_status,
            "basis": basis,
            "krillm": a,
            "ollama": b,
            "delta_ratio": round(delta_ratio, 4),
        },
    }

    if media_path:
        sig = media_signature(media_path)
        result["media"] = {
            "status": "ok",
            "kind": media_kind,
            "signature": sig,
            "note": "Both engines receive the same fixture file by path/bytes.",
        }

    overall = "ok"
    if prompt_status == "mismatch":
        overall = "mismatch"
    if overall == "mismatch":
        result["details"] = (
            f"prompt size differs by {delta_ratio*100:.1f}% between engines "
            f"({basis}: krillm={a}, ollama={b})"
        )
    result["status"] = overall
    return result


def krill_path_for_task(args: argparse.Namespace, task: dict[str, Any], use_server: bool) -> str:
    """Decide which KrillLM path to use for a single task.

    Audio always falls back to the bridge because no native audio path exists.
    Text/image follow the explicit --krillm-image-mode selection, with
    --krillm-url forcing the legacy server path when set.
    """
    if task.get("audio"):
        return "bridge"
    if use_server:
        return "native_server"
    return args.krillm_image_mode


def dispatch_krill_task(
    args: argparse.Namespace,
    task: dict[str, Any],
    model: Any,
    processor: Any,
    path: str,
    measured: bool,
) -> Optional[dict[str, Any]]:
    if path == "native_cli":
        result = run_krill_native_cli_task(args, task, measured=measured)
    elif path == "native_server":
        result = run_krill_native_server_task(args, task, measured=measured)
    else:
        if model is None or processor is None:
            raise RuntimeError(
                "bridge path requested but mlx-vlm model is not loaded; "
                "this typically means --krillm-image-mode bridge was selected "
                "without a local model load — check that mlx-vlm is installed."
            )
        result = run_krill_task(model, processor, task, args.temperature, args.top_p, measured=measured)
    if result is not None:
        result["krillm_path"] = path
    return result


def main() -> int:
    args = parse_args()
    if args.runs < 1 or args.warmup < 0:
        raise SystemExit("--runs must be >= 1 and --warmup must be >= 0")

    # Validate the krillm CLI binary up-front when any task may need it.
    if args.engine in ("both", "krillm") and args.krillm_image_mode == "native_cli":
        bin_path = Path(args.krillm_bin)
        if not bin_path.exists() or not os.access(bin_path, os.X_OK):
            raise SystemExit(
                f"krillm CLI binary not found or not executable at '{args.krillm_bin}'. "
                "Build it with `make release` (or pass --krillm-bin <path>)."
            )

    assets = ensure_assets(args.output, args.image, args.audio)
    tasks = [
        {
            "name": "text",
            "prompt": "Explain quantum computing in simple terms.",
            "max_tokens": args.text_max_tokens,
        },
        {
            "name": "image",
            "prompt": "What is shown in this image? Answer briefly.",
            "max_tokens": args.image_max_tokens,
            "image": assets["image"],
        },
        {
            "name": "audio",
            "prompt": "What sound is in this audio? Answer briefly.",
            "max_tokens": args.audio_max_tokens,
            "audio": assets["audio"],
        },
    ]

    include_krillm = args.engine in ("both", "krillm")
    include_ollama = args.engine in ("both", "ollama")
    use_server = bool(getattr(args, "krillm_url", None))
    krill_quant = krill_quantization(args.krill_model) if include_krillm else None
    ollama_quant = ollama_show(args.ollama_bin, args.ollama_model, args.timeout) if include_ollama else None

    # Determine if we need the in-process mlx-vlm model (used for bridge path,
    # which is required for audio in all modes and for text/image when
    # --krillm-image-mode bridge is selected).
    audio_task_present = any(t.get("audio") for t in tasks)
    needs_bridge_model = include_krillm and not use_server and (
        args.krillm_image_mode == "bridge" or audio_task_present
    )

    model = None
    processor = None
    krill_load_s = None
    if needs_bridge_model:
        if not _MLX_VLM_AVAILABLE:
            raise SystemExit(
                "mlx-vlm is required for the bridge path "
                f"(audio fallback or --krillm-image-mode bridge) but failed to import: "
                f"{_MLX_VLM_IMPORT_ERROR}. Install mlx-vlm or pick a krillm-image-mode "
                "that does not include audio."
            )
        load_start = time.perf_counter()
        model, processor = load(args.krill_model)
        krill_load_s = time.perf_counter() - load_start
    if include_krillm and use_server:
        # Verify server is reachable
        try:
            health_req = urllib.request.Request(args.krillm_url.rstrip("/") + "/healthz")
            with urllib.request.urlopen(health_req, timeout=10) as resp:
                health = json.loads(resp.read())
            print(f"KrillLM server: {health.get('status')} (model: {health.get('model')})")
        except Exception as exc:
            raise SystemExit(f"KrillLM server not reachable at {args.krillm_url}: {exc}")

    results: dict[str, Any] = {}
    for task in tasks:
        krill_path = krill_path_for_task(args, task, use_server)
        # Skip warmup for server media tasks (server doesn't handle media yet)
        skip_krill_task = use_server and (task.get("image") or task.get("audio"))
        for _ in range(args.warmup):
            if include_krillm and not skip_krill_task:
                dispatch_krill_task(args, task, model, processor, krill_path, measured=False)
            if include_ollama:
                run_ollama_task(args, task, measured=False)

        task_results: dict[str, Any] = {
            "prompt": task["prompt"],
            "max_tokens": task["max_tokens"],
            "media": {k: task[k] for k in ("image", "audio") if k in task},
        }
        krill_runs_for_parity: list[dict[str, Any]] = []
        ollama_runs_for_parity: list[dict[str, Any]] = []
        if include_krillm:
            if use_server and (task.get("image") or task.get("audio")):
                # Legacy --krillm-url skip-media branch (server agent owns this region).
                krill_runs = [
                    run_krill_server_task(args, task, measured=True)
                    for _ in range(args.runs)
                ]
                valid_runs = [run for run in krill_runs if run]
                if valid_runs:
                    for run in valid_runs:
                        run["krillm_path"] = "native_server"
                    group, source = apply_cache_mode(valid_runs, args.cache_mode, args.warmup, True)
                    task_results["krillm"] = {
                        "runs": valid_runs,
                        "summary": summarize(valid_runs),
                        "cache_mode": group,
                        "cache_mode_source": source,
                    }
                    krill_runs_for_parity = valid_runs
                else:
                    task_results["krillm_skipped"] = "Server mode returned no valid runs for this task"
            else:
                krill_runs = [
                    dispatch_krill_task(args, task, model, processor, krill_path, measured=True)
                    for _ in range(args.runs)
                ]
                is_cache_hitting = krill_path == "native_server"
                group, source = apply_cache_mode(krill_runs, args.cache_mode, args.warmup, is_cache_hitting)
                task_results["krillm"] = {
                    "runs": krill_runs,
                    "summary": summarize([run for run in krill_runs if run]),
                    "cache_mode": group,
                    "cache_mode_source": source,
                    "krillm_path": krill_path,
                }
                krill_runs_for_parity = [run for run in krill_runs if run]
        if include_ollama:
            ollama_runs = [run_ollama_task(args, task, measured=True) for _ in range(args.runs)]
            group, source = apply_cache_mode(ollama_runs, args.cache_mode, args.warmup, False)
            task_results["ollama"] = {
                "runs": ollama_runs,
                "summary": summarize([run for run in ollama_runs if run]),
                "cache_mode": group,
                "cache_mode_source": source,
            }
            ollama_runs_for_parity = [run for run in ollama_runs if run]

        if include_krillm and include_ollama and krill_runs_for_parity and ollama_runs_for_parity:
            media_path = task.get("image") or task.get("audio")
            media_kind = "image" if task.get("image") else ("audio" if task.get("audio") else None)
            parity = parity_field(
                krill_runs_for_parity,
                ollama_runs_for_parity,
                task["prompt"],
                media_path,
                media_kind,
            )
            task_results["input_parity"] = parity
            if parity.get("status") == "mismatch":
                print(
                    f"WARN: input parity mismatch for task '{task['name']}': {parity.get('details')}",
                    file=sys.stderr,
                )

        results[task["name"]] = task_results

    report = {
        "status": "ok",
        "benchmark": {
            "krill_model": args.krill_model,
            "ollama_model": args.ollama_model,
            "engine": args.engine,
            "krillm_mode": "server" if use_server else "mlx-vlm",
            "krillm_url": args.krillm_url if use_server else None,
            "krillm_image_mode": args.krillm_image_mode,
            "krillm_bin": args.krillm_bin if args.krillm_image_mode == "native_cli" else None,
            "runs": args.runs,
            "warmup": args.warmup,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "cache_mode_requested": args.cache_mode,
        },
        "environment": environment(),
        "assets": {
            "image": {"path": assets["image"], "sha256": sha256_bytes(Path(assets["image"]).read_bytes())},
            "audio": {"path": assets["audio"], "sha256": sha256_bytes(Path(assets["audio"]).read_bytes())},
        },
        "quantization": {
            "krillm": krill_quant,
            "ollama": ollama_quant,
            "comparison": quantization_comparison(krill_quant, ollama_quant),
        },
        "krillm_model_load_s": krill_load_s,
        "krillm_image_mode": args.krillm_image_mode,
        "results": results,
    }
    if args.krillm_image_mode == "native_cli":
        report["krillm_bin"] = args.krillm_bin

    if args.cache_mode != "auto":
        report["cache_mode"] = args.cache_mode
        report["cache_mode_source"] = "explicit"
    else:
        per_task_modes: list[str] = []
        for task_data in results.values():
            for engine in ("krillm", "ollama"):
                engine_data = task_data.get(engine)
                if isinstance(engine_data, dict) and engine_data.get("cache_mode"):
                    per_task_modes.append(engine_data["cache_mode"])
        if per_task_modes and all(m == per_task_modes[0] for m in per_task_modes):
            report["cache_mode"] = per_task_modes[0]
        else:
            report["cache_mode"] = "mixed" if per_task_modes else "warm"
        report["cache_mode_source"] = "auto"

    if use_server:
        report["server_media"] = {
            "status": "supported",
            "image": "native",
            "audio": "bridge",
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Benchmark report: {output}")
    compact = {}
    for name, value in results.items():
        compact[name] = {
            engine: engine_result["summary"]
            for engine, engine_result in value.items()
            if engine in ("krillm", "ollama")
        }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
