#!/usr/bin/env python3
"""End-to-end real-model check for the 2026-09-30 logprobs follow-up:
Ollama `/api/chat` + `/api/generate`, and legacy OpenAI `/v1/completions`
(docs/LOGPROBS_PLAN.md's "Ollama + legacy completions" addendum).

Against a real running `krill serve`, greedy (`temperature: 0`), checks:

  1. /v1/completions (`logprobs: int`, non-streaming only - this endpoint
     has no streaming support at all in Krill, pre-existing and unrelated
     to logprobs): `choices[0].logprobs.tokens` joined equals the returned
     `text`; `token_logprobs`/`tokens`/`top_logprobs`/`text_offset` all the
     same length; `text_offset` is a strictly-nondecreasing cumulative sum
     matching each token's own length; `top_logprobs[i]` has at most
     `logprobs+1` entries and always contains the sampled token's own key
     with the same value as `token_logprobs[i]`; a request WITHOUT
     `logprobs` gets no `logprobs` key at all (byte-identity check via key
     absence, not just null).
  2. /api/chat (`logprobs: bool`, `top_logprobs: int`), non-streaming: the
     top-level `logprobs` array's `bytes`-concat equals `message.content`'s
     UTF-8 bytes; a request without logprobs gets no `logprobs` key.
  3. /api/chat, streaming (NDJSON): concatenating every line's `logprobs`
     entries' bytes equals the concatenation of every line's
     `message.content`; streamed content equals the non-streaming content
     for the same greedy request; entry count matches between the two.
  4. /api/generate, same checks as /api/chat but with `response` instead of
     `message.content`.
  5. Cross-endpoint parity: for the SAME prompt (rendered as a single-turn
     chat message so the templated prompt is identical), the FIRST
     generated token's logprob from /v1/completions, /api/chat, and
     /api/generate all equal (to float32 precision) what
     /v1/chat/completions reports - same engine, same plain decode path,
     so they should match exactly, not just approximately.

Usage: logprobs_ollama_completions_e2e_check.py [base_url] [api_key] [model]
  base_url defaults to the OpenAI-style base (http://host:port/v1); the
  Ollama-dialect endpoints are derived by stripping the trailing /v1.
"""
import json
import sys
import urllib.request

from openai import OpenAI

base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:57480/v1"
api_key = sys.argv[2] if len(sys.argv) > 2 else "dummy"
model = sys.argv[3] if len(sys.argv) > 3 else "llama-3.2-1b"
root = base_url[:-3] if base_url.endswith("/v1") else base_url

client = OpenAI(base_url=base_url, api_key=api_key)

PROMPT = "Reply with only this exact code, nothing else, no markdown fences:\nif x < y:\n    return \"<div>ok</div>\""
MAX_TOKENS = 40

failures = []


def post(path, body):
    req = urllib.request.Request(
        root + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def post_ndjson(path, body):
    req = urllib.request.Request(
        root + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    lines = []
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:
            raw = raw.strip()
            if raw:
                lines.append(json.loads(raw))
    return lines


def check(label, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{label}] {status}" + (f" - {detail}" if detail else ""))
    if not cond:
        failures.append(f"{label}: {detail}")
    return cond


# --- 1. /v1/completions ---------------------------------------------------

def check_legacy_completions():
    # Byte-identity: no logprobs key at all when not requested.
    r = post("/v1/completions", {
        "model": model, "prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": 0,
    })
    check("completions/off-has-no-key", "logprobs" not in r["choices"][0],
          f"keys={sorted(r['choices'][0].keys())}")

    r = post("/v1/completions", {
        "model": model, "prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": 0,
        "logprobs": 3,
    })
    text = r["choices"][0]["text"]
    lp = r["choices"][0]["logprobs"]
    tokens, tok_lp, top_lp, offsets = (
        lp["tokens"], lp["token_logprobs"], lp["top_logprobs"], lp["text_offset"])
    n = len(tokens)
    check("completions/lengths-match", len(tok_lp) == n and len(top_lp) == n and len(offsets) == n,
          f"n={n} tok_lp={len(tok_lp)} top_lp={len(top_lp)} offsets={len(offsets)}")
    check("completions/tokens-join-equals-text", "".join(tokens) == text,
          f"joined={''.join(tokens)!r} text={text!r}")
    expected_offset = 0
    offsets_ok = True
    for tok, off in zip(tokens, offsets):
        if off != expected_offset:
            offsets_ok = False
        expected_offset += len(tok)
    check("completions/text-offset-cumulative", offsets_ok, f"offsets={offsets} tokens={tokens}")
    dict_ok = True
    for i, (tok, sampled_lp, dct) in enumerate(zip(tokens, tok_lp, top_lp)):
        if len(dct) > 4:  # logprobs=3 -> up to 3+1
            dict_ok = False
        if tok not in dct or dct[tok] != sampled_lp:
            dict_ok = False
    check("completions/dict-contains-sampled-token", dict_ok,
          f"top_lp={top_lp} tok_lp={tok_lp}")
    print(f"[completions] {n} tokens, text={text!r}")
    return text, tokens, tok_lp


# --- 2/3. /api/chat ---------------------------------------------------------

def bytes_concat_matches(content, entries):
    concat = bytearray()
    for e in entries:
        b = e.get("bytes")
        if b:
            concat.extend(b)
    return bytes(concat) == content.encode("utf-8")


def check_ollama_chat():
    r = post("/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False, "options": {"temperature": 0, "num_predict": MAX_TOKENS},
    })
    check("api-chat/off-has-no-key", "logprobs" not in r, f"keys={sorted(r.keys())}")

    r = post("/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False, "options": {"temperature": 0, "num_predict": MAX_TOKENS},
        "logprobs": True, "top_logprobs": 3,
    })
    content = r["message"]["content"]
    entries = r.get("logprobs", [])
    check("api-chat/non-stream-bytes-concat", bytes_concat_matches(content, entries),
          f"content={content!r}")
    for e in entries:
        if "top_logprobs" in e and len(e["top_logprobs"]) == 0:
            failures.append("api-chat: empty top_logprobs list present instead of omitted")
    print(f"[api-chat] non-stream: {len(entries)} entries, content={content!r}")

    lines = post_ndjson("/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True, "options": {"temperature": 0, "num_predict": MAX_TOKENS},
        "logprobs": True, "top_logprobs": 3,
    })
    stream_content = "".join(
        (ln.get("message", {}) or {}).get("content", "") for ln in lines)
    stream_entries = []
    for ln in lines:
        stream_entries.extend(ln.get("logprobs", []) or [])
    check("api-chat/stream-bytes-concat", bytes_concat_matches(stream_content, stream_entries),
          f"content={stream_content!r}")
    check("api-chat/stream-equals-non-stream-content", stream_content == content,
          f"stream={stream_content!r} non-stream={content!r}")
    check("api-chat/stream-equals-non-stream-entry-count", len(stream_entries) == len(entries),
          f"stream={len(stream_entries)} non-stream={len(entries)}")
    print(f"[api-chat] stream: {len(lines)} NDJSON lines, {len(stream_entries)} entries")
    return content, entries


def check_ollama_generate():
    r = post("/api/generate", {
        "model": model, "prompt": PROMPT,
        "stream": False, "options": {"temperature": 0, "num_predict": MAX_TOKENS},
    })
    check("api-generate/off-has-no-key", "logprobs" not in r, f"keys={sorted(r.keys())}")

    r = post("/api/generate", {
        "model": model, "prompt": PROMPT,
        "stream": False, "options": {"temperature": 0, "num_predict": MAX_TOKENS},
        "logprobs": True, "top_logprobs": 3,
    })
    response = r["response"]
    entries = r.get("logprobs", [])
    check("api-generate/non-stream-bytes-concat", bytes_concat_matches(response, entries),
          f"response={response!r}")
    print(f"[api-generate] non-stream: {len(entries)} entries, response={response!r}")

    lines = post_ndjson("/api/generate", {
        "model": model, "prompt": PROMPT,
        "stream": True, "options": {"temperature": 0, "num_predict": MAX_TOKENS},
        "logprobs": True, "top_logprobs": 3,
    })
    stream_response = "".join(ln.get("response", "") for ln in lines)
    stream_entries = []
    for ln in lines:
        stream_entries.extend(ln.get("logprobs", []) or [])
    check("api-generate/stream-bytes-concat", bytes_concat_matches(stream_response, stream_entries),
          f"response={stream_response!r}")
    check("api-generate/stream-equals-non-stream-response", stream_response == response,
          f"stream={stream_response!r} non-stream={response!r}")
    check("api-generate/stream-equals-non-stream-entry-count", len(stream_entries) == len(entries),
          f"stream={len(stream_entries)} non-stream={len(entries)}")
    # Confirm a system override still works with logprobs on (no crash / no
    # rejection) - a pre-existing feature this change must not break.
    r2 = post("/api/generate", {
        "model": model, "prompt": PROMPT, "system": "Reply tersely.",
        "stream": False, "options": {"temperature": 0, "num_predict": MAX_TOKENS},
        "logprobs": True, "top_logprobs": 1,
    })
    check("api-generate/system-override-with-logprobs", "response" in r2)
    return response, entries


def check_chat_completions_reference():
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=MAX_TOKENS, temperature=0,
        logprobs=True, top_logprobs=3,
    )
    lp = r.choices[0].logprobs
    entries = [{"token": e.token, "logprob": e.logprob} for e in (lp.content if lp else [])]
    print(f"[chat/completions reference] {len(entries)} entries, "
          f"content={r.choices[0].message.content!r}")
    return entries


def main():
    # Order matters for the cross-endpoint parity check below: the SAME
    # prompt is sent to every endpoint, so Krill's prefix cache (a real,
    # pre-existing, logprobs-unrelated engine property) is cold on the very
    # FIRST call of the run and warm on every call after it - a "full
    # prefill" vs "cache hit, reforward last position" numerical difference
    # of a few e-3 nats, the SAME class of noise floor the numeric-parity
    # investigation in docs/LOGPROBS_PLAN.md's Resolutions section already
    # characterizes (full-sequence vs incremental KV-cache decode). Running
    # the /v1/chat/completions reference call LAST (after the other three
    # have already warmed the cache with the identical prompt) puts all four
    # on the SAME (warm) compute path, so the comparison is apples-to-apples.
    text, tokens, tok_lp = check_legacy_completions()
    chat_content, chat_entries = check_ollama_chat()
    gen_response, gen_entries = check_ollama_generate()
    ref_entries = check_chat_completions_reference()

    # Cross-endpoint parity: first generated token's logprob must be
    # IDENTICAL (same engine, same plain decode path, same greedy request,
    # same templated single-turn prompt, same warm-cache compute path)
    # across all four surfaces.
    if ref_entries and tok_lp and chat_entries and gen_entries:
        ref0 = ref_entries[0]["logprob"]
        vals = {
            "chat/completions (reference)": ref0,
            "/v1/completions": tok_lp[0],
            "/api/chat": chat_entries[0]["logprob"],
            "/api/generate": gen_entries[0]["logprob"],
        }
        print("\nFirst-token logprob by endpoint:", vals)
        max_diff = max(abs(v - ref0) for v in vals.values())
        check("cross-endpoint/first-token-logprob-matches", max_diff < 1e-6,
              f"max diff from reference={max_diff!r}, values={vals}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("All Ollama + legacy-completions logprobs e2e checks passed.")


if __name__ == "__main__":
    main()
