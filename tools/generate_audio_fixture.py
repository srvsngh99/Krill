#!/usr/bin/env python3
"""Generate deterministic, semantically obvious audio fixtures for benchmarks.

The previous fixture (`gemma4-tone-5s.wav`) was ambiguous: mlx-vlm reported a
"click/percussive" sound while Ollama reported "dog barking". Release readiness
blocker 4 (see docs/RELEASE_READINESS_REMEDIATION.md) calls for fixtures that
any reasonable audio model should agree on.

This script writes two additional fixtures alongside the existing tone file:

- ``gemma4-sine-1khz-5s.wav``: a pure 1 kHz sine, 5 s, 16 kHz mono, 16-bit PCM.
  Expected description: a single steady tone / sine wave / continuous beep.
- ``gemma4-silence-2s.wav``: 2 s of digital silence at 16 kHz mono, 16-bit PCM.
  Expected description: silence / no sound / nothing audible.

Outputs are deterministic byte-for-byte: no randomness, no timestamps, fixed
amplitude and phase, integer sample math. The script prints a SHA-256 for each
written file so determinism can be verified across runs.

Standard library only (``wave``, ``struct``, ``math``, ``hashlib``, ``argparse``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import wave
from pathlib import Path
from typing import Callable, Iterable

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
            "expected_any": [
                "silence", "silent", "nothing", "no sound", "quiet",
            ],
            "forbidden": ["dog", "bark", "music", "tone", "speech"],
        },
    ),
]


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
        write_wav(path, builder())
        digest = sha256_of(path)
        print(f"{path}  sha256={digest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
