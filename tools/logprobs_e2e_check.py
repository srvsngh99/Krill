#!/usr/bin/env python3
"""End-to-end real-model check for the `logprobs` aggregator fix
(docs/LOGPROBS_PLAN.md finding #1).

Against a real running `krill serve`, for both a Hindi/Devanagari prompt and
a code prompt containing `<` (which the streaming reasoning filter holds
while disambiguating a possible tag), greedy + logprobs + top_logprobs=3:

  1. Non-streaming: concatenating every `logprobs.content[]` entry's `bytes`
     equals the UTF-8 bytes of `message.content` exactly.
  2. Streaming: same byte-concat check across all chunks' `logprobs.content[]`
     entries, AND the streamed `content` (concatenation of every chunk's
     `delta.content`) is byte-identical to the non-streaming `content` for
     the same (greedy) request.

Usage: logprobs_e2e_check.py [base_url] [api_key] [model]
"""
import sys

from openai import OpenAI

base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:57480/v1"
api_key = sys.argv[2] if len(sys.argv) > 2 else "dummy"
model = sys.argv[3] if len(sys.argv) > 3 else "llama-3.2-1b"

client = OpenAI(base_url=base_url, api_key=api_key)

PROMPTS = {
    # NOTE: some models emit a leading "\n\n" before real content on certain
    # prompts. Krill's non-streaming path trims it (`ReasoningParser.strip`);
    # the streaming path does not - a PRE-EXISTING inconsistency, unrelated
    # to logprobs (reproduces identically with logprobs off) and out of
    # scope for this fix (it would need StreamingReasoningFilter, shared by
    # every CLI/TUI caller, to also buffer-and-trim leading whitespace).
    # Worded to avoid triggering it so this script's stream==non-stream
    # check isolates what THIS fix is actually responsible for.
    "hindi": "Translate to Hindi and reply with ONLY the translation, "
             "no leading newline, no punctuation commentary: Hello world",
    "code": (
        "Reply with only this exact code, nothing else, no markdown fences:\n"
        "if x < y:\n    return \"<div>ok</div>\""
    ),
}

failures = []


def check_bytes_concat(label, content, entries):
    concat = bytearray()
    for e in entries:
        b = e.get("bytes")
        if b:
            concat.extend(b)
    expected = content.encode("utf-8")
    if bytes(concat) != expected:
        failures.append(
            f"[{label}] bytes-concat != content.utf8\n"
            f"  content : {content!r}\n"
            f"  from utf8: {expected!r}\n"
            f"  concat   : {bytes(concat)!r}"
        )
        return False
    return True


def run_non_streaming(name, prompt):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0,
        logprobs=True,
        top_logprobs=3,
    )
    content = r.choices[0].message.content or ""
    lp = r.choices[0].logprobs
    entries = [
        {"token": e.token, "bytes": e.bytes, "logprob": e.logprob}
        for e in (lp.content if lp else [])
    ]
    ok = check_bytes_concat(f"{name}/non-stream", content, entries)
    n_visible_tokens = len(entries)
    print(f"[{name}] non-stream: {'OK' if ok else 'FAIL'} - "
          f"{n_visible_tokens} entries, content={content!r}")
    return content, entries


def run_streaming(name, prompt):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0,
        logprobs=True,
        top_logprobs=3,
        stream=True,
    )
    full_text = ""
    all_entries = []
    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue
        if choice.delta and choice.delta.content:
            full_text += choice.delta.content
        if choice.logprobs is not None and choice.logprobs.content:
            for e in choice.logprobs.content:
                all_entries.append({"token": e.token, "bytes": e.bytes, "logprob": e.logprob})
    ok = check_bytes_concat(f"{name}/stream", full_text, all_entries)
    print(f"[{name}] stream:     {'OK' if ok else 'FAIL'} - "
          f"{len(all_entries)} entries, content={full_text!r}")
    return full_text, all_entries


def main():
    for name, prompt in PROMPTS.items():
        ns_content, ns_entries = run_non_streaming(name, prompt)
        s_content, s_entries = run_streaming(name, prompt)
        if ns_content != s_content:
            failures.append(
                f"[{name}] stream content != non-stream content\n"
                f"  non-stream: {ns_content!r}\n"
                f"  stream    : {s_content!r}"
            )
        else:
            print(f"[{name}] stream content == non-stream content: OK")
        if len(ns_entries) != len(s_entries):
            failures.append(
                f"[{name}] entry count mismatch: non-stream={len(ns_entries)} "
                f"stream={len(s_entries)}"
            )

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("All logprobs e2e checks passed.")


if __name__ == "__main__":
    main()
