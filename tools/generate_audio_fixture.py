#!/usr/bin/env python3
"""Generate deterministic, semantically obvious audio fixtures for benchmarks.

The previous fixture (`gemma4-tone-5s.wav`) was ambiguous: mlx-vlm reported a
"click/percussive" sound while Ollama reported "dog barking". Release readiness
blocker 4 (see docs/RELEASE_READINESS_REMEDIATION.md) calls for fixtures that
any reasonable audio model should agree on.

This script writes these fixtures alongside the existing tone file:

- ``gemma4-sine-1khz-5s.wav``: a pure 1 kHz sine, 5 s, 16 kHz mono, 16-bit PCM.
  Expected description: a single steady tone / sine wave / continuous beep.
- ``gemma4-silence-2s.wav``: 2 s of digital silence at 16 kHz mono, 16-bit PCM.
  Expected description: silence / no sound / nothing audible.
- ``gemma4-speech-pangram.wav``: a fixed spoken pangram, 16 kHz mono, 16-bit
  PCM, synthesised via macOS ``say`` + ``afconvert``. This is the WS6
  numerical-parity fixture: a pure tone/silence is out-of-distribution for a
  speech-understanding model (Gemma 4 E2B hallucinates "cat"/"dog" on it,
  non-deterministically, in both the mlx-vlm oracle and the native path), so
  the sine/silence fixtures are non-empty smokes only. Speech with a known
  transcript is the deterministic semantic gate.

The sine/silence fixtures are deterministic byte-for-byte (no randomness,
fixed amplitude/phase, integer sample math). The speech fixture is
*content*-deterministic, not byte-deterministic: a pinned voice + sentence
yields a stable transcript (which is what the term-based rubric checks),
but the encoded bytes vary by macOS/voice version. It is macOS-only and is
skipped with a clear message when ``say``/``afconvert`` are unavailable.

Standard library plus, for the speech fixture only, the macOS ``say`` and
``afconvert`` binaries (no third-party Python deps).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Callable, Iterable

# WS6 speech fixture: a fixed pangram spoken by a pinned macOS voice. The
# transcript is what the term-based rubric checks, so this only needs to be
# content-deterministic. Keep the sentence and voice stable across runs.
SPEECH_SENTENCE = "The quick brown fox jumps over the lazy dog."
SPEECH_VOICE = "Samantha"  # ships with macOS; stable, clearly intelligible

SAMPLE_RATE_HZ = 16_000
BITS_PER_SAMPLE = 16
NUM_CHANNELS = 1
SAMPLE_WIDTH_BYTES = BITS_PER_SAMPLE // 8

# Use a conservative amplitude (~0.5 of full-scale) to avoid any clipping
# concerns and to keep the encoded bytes identical across platforms regardless
# of dithering choices in downstream decoders.
SINE_PEAK_AMPLITUDE = 16_000  # int16 peak; full-scale is 32767


def sine_samples(frequency_hz: float, duration_s: float) -> Iterable[int]:
    """Yield int16 PCM samples for a pure sine wave.

    Math is performed in float64 then quantised with ``round`` so the output
    is bit-identical regardless of the host platform's libm rounding modes for
    the sample range we care about.
    """
    total_samples = int(round(SAMPLE_RATE_HZ * duration_s))
    two_pi_f_over_sr = (2.0 * math.pi * frequency_hz) / SAMPLE_RATE_HZ
    for n in range(total_samples):
        value = SINE_PEAK_AMPLITUDE * math.sin(two_pi_f_over_sr * n)
        # int() truncates toward zero; round() gives nearest-integer which is
        # what we want for symmetric sine output.
        sample = int(round(value))
        if sample > 32_767:
            sample = 32_767
        elif sample < -32_768:
            sample = -32_768
        yield sample


def silence_samples(duration_s: float) -> Iterable[int]:
    total_samples = int(round(SAMPLE_RATE_HZ * duration_s))
    for _ in range(total_samples):
        yield 0


def write_speech_wav(path: Path) -> bool:
    """Synthesise the speech pangram via macOS ``say`` + ``afconvert``.

    Returns True if written, False if skipped (non-macOS or tools missing).
    Output is 16 kHz mono 16-bit PCM WAV to match the other fixtures and the
    Gemma 4 USM feature extractor's expected sampling rate.
    """
    if sys.platform != "darwin" or not (
        shutil.which("say") and shutil.which("afconvert")
    ):
        print(
            f"SKIP {path.name}: requires macOS 'say' + 'afconvert' "
            f"(speech fixture is content-deterministic, macOS-only)",
            file=sys.stderr,
        )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "speech_raw.aiff"
        subprocess.run(
            ["say", "-v", SPEECH_VOICE, "-o", str(raw), SPEECH_SENTENCE],
            check=True,
        )
        subprocess.run(
            [
                "afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE_HZ}",
                "-c", str(NUM_CHANNELS), str(raw), str(path),
            ],
            check=True,
        )
    return True


def write_wav(path: Path, samples: Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Buffer all samples first so we can write them in one pass; this keeps
    # the resulting file's chunk sizes deterministic and avoids any chance of
    # partial writes affecting the wave header.
    payload = b"".join(struct.pack("<h", s) for s in samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE_HZ)
        wf.writeframes(payload)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


# Each entry: (filename, builder, rubric description)
FixtureBuilder = Callable[[], Iterable[int]]

FIXTURES: list[tuple[str, FixtureBuilder, dict]] = [
    (
        "gemma4-sine-1khz-5s.wav",
        lambda: sine_samples(frequency_hz=1_000.0, duration_s=5.0),
        {
            "description": "Pure 1 kHz sine wave, 5 s, 16 kHz mono, 16-bit PCM.",
            "smoke_only": True,
            "expected_any": [
                "tone", "sine", "beep", "single", "steady", "continuous",
                "hum", "buzz",
            ],
            "forbidden": ["dog", "bark", "music", "speech", "voice"],
        },
    ),
    (
        "gemma4-silence-2s.wav",
        lambda: silence_samples(duration_s=2.0),
        {
            "description": "Digital silence, 2 s, 16 kHz mono, 16-bit PCM.",
            "smoke_only": True,
            "expected_any": [
                "silence", "silent", "nothing", "no sound", "quiet",
            ],
            "forbidden": ["dog", "bark", "music", "tone", "speech"],
        },
    ),
    (
        # WS6 numerical-parity gate. builder is None: self-written via
        # write_speech_wav (macOS say+afconvert), not the sample iterator.
        "gemma4-speech-pangram.wav",
        None,
        {
            "description": (
                f"Spoken pangram \"{SPEECH_SENTENCE}\" "
                f"(voice={SPEECH_VOICE}), 16 kHz mono, 16-bit PCM. "
                "Content-deterministic, macOS-only."
            ),
            "prompt": "Transcribe this audio exactly.",
            "expected_any": [
                "fox", "dog", "quick", "brown", "jump", "lazy",
            ],
            "forbidden": [
                "cat", "music", "silence", "tone", "bark", "i cannot",
            ],
        },
    ),
]

# Builder names whose mismatch on out-of-distribution non-speech input is
# expected; they are non-empty smokes, not semantic rubric gates.
SMOKE_ONLY = {"gemma4-sine-1khz-5s.wav", "gemma4-silence-2s.wav"}


def list_fixtures() -> None:
    payload = {
        "fixtures": [
            {
                "filename": name,
                "rubric": rubric,
            }
            for name, _builder, rubric in FIXTURES
        ]
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=".build/benchmarks/assets/",
        help="Directory to write fixtures into (default: .build/benchmarks/assets/).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the planned fixture names and rubric without writing anything.",
    )
    args = parser.parse_args()

    if args.list:
        list_fixtures()
        return 0

    out_dir = Path(args.output)
    for name, builder, _rubric in FIXTURES:
        path = out_dir / name
        if builder is None:
            # Self-writing speech fixture (macOS say+afconvert).
            if not write_speech_wav(path):
                continue
        else:
            write_wav(path, builder())
        digest = sha256_of(path)
        print(f"{path}  sha256={digest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
