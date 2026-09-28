#!/usr/bin/env python3
"""OpenAI Python SDK round-trip against a local dev `krill serve`.

Verifies:
  1. A request WITHOUT logprobs still works (regression check).
  2. logprobs=True, top_logprobs=5, non-streaming: response parses into
     typed ChatCompletionTokenLogprob objects with no SDK validation error.
  3. Same, streaming=True: each chunk parses into typed
     ChoiceDeltaLogprob-style objects with no SDK validation error.
"""
import sys

from openai import OpenAI

base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:57480/v1"
api_key = sys.argv[2] if len(sys.argv) > 2 else "dummy"
model = sys.argv[3] if len(sys.argv) > 3 else "llama-3.2-1b"

client = OpenAI(base_url=base_url, api_key=api_key)

print("=== 1. Plain request (no logprobs) still works ===")
r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say OK."}],
    max_tokens=8,
    temperature=0,
)
assert r.choices[0].logprobs is None, f"expected logprobs=None, got {r.choices[0].logprobs!r}"
print("OK - content:", repr(r.choices[0].message.content), "logprobs:", r.choices[0].logprobs)

print("\n=== 2. Non-streaming logprobs=True, top_logprobs=5 ===")
r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    max_tokens=20,
    temperature=0,
    logprobs=True,
    top_logprobs=5,
)
lp = r.choices[0].logprobs
assert lp is not None, "expected logprobs object, got None"
assert len(lp.content) > 0, "expected non-empty logprobs.content"
for entry in lp.content:
    assert isinstance(entry.token, str)
    assert isinstance(entry.logprob, float)
    assert entry.bytes is None or isinstance(entry.bytes, list)
    assert isinstance(entry.top_logprobs, list)
    for alt in entry.top_logprobs:
        assert isinstance(alt.token, str)
        assert isinstance(alt.logprob, float)
print(f"OK - {len(lp.content)} typed ChatCompletionTokenLogprob entries, "
      f"first: token={lp.content[0].token!r} logprob={lp.content[0].logprob} "
      f"top_logprobs={len(lp.content[0].top_logprobs)}")
print("content:", repr(r.choices[0].message.content))

print("\n=== 3. Streaming logprobs=True, top_logprobs=5 ===")
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    max_tokens=20,
    temperature=0,
    logprobs=True,
    top_logprobs=5,
    stream=True,
)
chunk_count = 0
content_chunks_with_logprobs = 0
full_text = ""
for chunk in stream:
    chunk_count += 1
    choice = chunk.choices[0] if chunk.choices else None
    if choice is None:
        continue
    if choice.delta.content:
        full_text += choice.delta.content
    if choice.logprobs is not None and choice.logprobs.content:
        content_chunks_with_logprobs += 1
        for entry in choice.logprobs.content:
            assert isinstance(entry.token, str)
            assert isinstance(entry.logprob, float)
            assert isinstance(entry.top_logprobs, list)
print(f"OK - {chunk_count} chunks total, {content_chunks_with_logprobs} carried "
      f"typed logprobs.content entries")
print("streamed content:", repr(full_text))

print("\nAll OpenAI SDK round-trip checks passed.")
