#!/usr/bin/env python3
"""Check that committed Krill release metadata names the same version.

This deliberately does not contact GitHub: it is a fast PR check for metadata
drift. Verifying that the formula digest matches the published asset remains a
release-publication responsibility in the canonical Homebrew tap.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def capture(pattern: str, text: str, label: str, errors: list[str]) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        errors.append(f"{label}: expected version field was not found")
        return None
    return match.group(1)


def main() -> int:
    errors: list[str] = []
    version = read("VERSION").strip()
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", version) is None:
        errors.append(f"VERSION: {version!r} is not a supported semantic version")

    checks = {
        "Sources/KrillRegistry/KrillVersion.swift": capture(
            r'^public let KrillVersion: String = "([^"]+)"',
            read("Sources/KrillRegistry/KrillVersion.swift"),
            "Swift KrillVersion",
            errors,
        ),
        "Formula/krill.rb version": capture(
            r'^\s*version "([^"]+)"',
            read("Formula/krill.rb"),
            "Homebrew formula version",
            errors,
        ),
        "Formula/krill.rb release URL tag": capture(
            r'/releases/download/v([^/]+)/krill-[^/]+-arm64-apple-macos\.tar\.gz"',
            read("Formula/krill.rb"),
            "Homebrew formula URL tag",
            errors,
        ),
        "Formula/krill.rb artifact name": capture(
            r'/releases/download/v[^/]+/krill-(.+?)-arm64-apple-macos\.tar\.gz"',
            read("Formula/krill.rb"),
            "Homebrew formula artifact version",
            errors,
        ),
        "RELEASES.md newest entry": capture(
            r'^## v([^ ]+)\s+[—-]',
            read("RELEASES.md"),
            "RELEASES newest entry",
            errors,
        ),
        "CHANGELOG.md newest release": capture(
            r'^## \[([^]]+)\](?:\s+-|$)',
            read("CHANGELOG.md").split("## [Unreleased]", 1)[-1],
            "CHANGELOG newest release",
            errors,
        ),
    }

    for label, found in checks.items():
        if found is not None and found != version:
            errors.append(f"{label}: {found!r} does not match VERSION {version!r}")

    formula = read("Formula/krill.rb")
    digest = capture(r'^\s*sha256 "([0-9a-f]+)"', formula, "Homebrew formula sha256", errors)
    if digest is not None and len(digest) != 64:
        errors.append("Formula/krill.rb sha256 must contain exactly 64 lowercase hex digits")

    if errors:
        print("release metadata is inconsistent:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"release metadata agrees on {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
