#!/usr/bin/env python3
"""Numeric parity check: Krill's reported `logprobs` vs mlx_lm computed
directly from the same checkpoint's raw forward-pass logits.

Reads a Krill /v1/chat/completions JSON response (requested with
logprobs+top_logprobs, temperature 0 / greedy) plus the messages used to
produce it, re-applies the model's own chat template via mlx_lm's tokenizer
(the same Jinja source Krill reads), then teacher-forces the exact greedy
token sequence step by step through the model, computing
log_softmax(logits) in float32 at each step and comparing to Krill's
reported logprob for the sampled token AND its top-N alternates, matched by
exact BYTE identity (not rank position, which can flip on near-ties) via a
GPT-2 byte-level-BPE bytes->id table built from the tokenizer's own vocab.

Also measures mlx_lm's OWN intrinsic numeric noise floor, independent of
Krill entirely: the same logprobs computed two ways - (A) one full forward
pass over the whole prompt+generated sequence at once, and (B) incremental,
one new token at a time, through a real KV cache (mirroring how both Krill
and mlx_lm's own `generate` actually decode) - and reports the max diff
between A and B. Krill's diff against B (the apples-to-apples incremental
comparison) should be within roughly 2x of the A-vs-B floor; a much larger
gap points at a real bug rather than ordinary floating-point non-
associativity between batched and incremental attention.

Fixes applied in this revision (docs/LOGPROBS_PLAN.md finding #3):
  - Teacher-forcing previously advanced the context with mlx's OWN greedy
    pick every step, not Krill's actual sampled id - once the two diverge
    even once (a near-tie flips either way from ordinary fp noise), every
    later position was scored from a genuinely different prefix than the
    one Krill's own forward pass saw, which manufactures large, spurious
    diffs that compound with distance from the divergence point. Now uses
    Krill's own id (resolved via the bytes table) when available, exactly
    matching what Krill's decode loop actually conditioned on.
  - `build_bytes_to_id` used plain last-wins dict assignment, so if two
    vocab pieces map to the same byte string, the first is silently
    shadowed and a comparison could resolve to the WRONG token id. Now
    detects such collisions, excludes them from the lookup table, and
    reports them instead of guessing.

Usage: logprobs_parity.py <krill_response.json> <model_dir> <messages.json>
"""
import json
import sys
from collections import defaultdict

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load


def gpt2_byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def piece_to_bytes(piece, byte_decoder):
    if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
        try:
            return bytes([int(piece[3:5], 16)])
        except ValueError:
            pass
    out = bytearray()
    ok = True
    for ch in piece:
        b = byte_decoder.get(ch)
        if b is None:
            ok = False
            break
        out.append(b)
    if ok:
        return bytes(out)
    return piece.encode("utf-8", errors="ignore")


def build_bytes_to_id(tokenizer):
    """Bytes -> token id, with COLLISION DETECTION (finding #3a): if more
    than one vocab piece maps to the same byte string, that byte string is
    ambiguous and is excluded from the table entirely (rather than silently
    resolving to whichever piece iterated last), and reported separately so
    the caller can see how many comparisons that cost."""
    byte_decoder = gpt2_byte_decoder()
    vocab = tokenizer._tokenizer.get_vocab()
    by_bytes = defaultdict(list)
    for piece, tok_id in vocab.items():
        by_bytes[piece_to_bytes(piece, byte_decoder)].append((piece, tok_id))
    table = {}
    collisions = {}
    for b, entries in by_bytes.items():
        if len(entries) == 1:
            table[b] = entries[0][1]
        else:
            collisions[b] = entries
    return table, collisions


def load_ids(resp, messages, tokenizer, bytes_to_id):
    """The exact id sequence Krill's own decode loop conditioned on: the
    prompt ids (via the model's own chat template) followed by KRILL'S OWN
    sampled id at each step (resolved via `bytes_to_id`), falling back to
    None (caller re-derives via argmax) only when a byte string is
    unresolved or ambiguous."""
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    krill_ids = []
    unresolved = []
    for i, entry in enumerate(resp["choices"][0]["logprobs"]["content"]):
        b = bytes(entry["bytes"])
        tid = bytes_to_id.get(b)
        if tid is None:
            unresolved.append((i, entry["token"], entry["bytes"]))
        krill_ids.append(tid)
    return list(prompt_ids), krill_ids, unresolved


def full_sequence_logprobs(model, ids):
    """Method A: ONE forward pass over the whole sequence; log_softmax at
    every position in float32. Returns a list of [vocab] float32 arrays,
    one per position (position i predicts token i+1)."""
    inp = mx.array([ids])
    logits = model(inp).astype(mx.float32)
    logsumexp = mx.logsumexp(logits, axis=-1, keepdims=True)
    logprobs = logits - logsumexp
    mx.eval(logprobs)
    return logprobs[0]


def incremental_logprobs(model, prompt_ids, gen_ids):
    """Method B: prefill the prompt once, then step one NEW token at a time
    through a real KV cache (mirrors how Krill and mlx_lm's own `generate`
    actually decode) - the intrinsic-noise comparison point for method A."""
    cache = make_prompt_cache(model)
    out = []
    inp = mx.array([prompt_ids])
    logits = model(inp, cache=cache).astype(mx.float32)
    last = logits[0, -1, :]
    out.append(last - mx.logsumexp(last))
    for tid in gen_ids[:-1]:
        inp = mx.array([[tid]])
        logits = model(inp, cache=cache).astype(mx.float32)
        last = logits[0, -1, :]
        out.append(last - mx.logsumexp(last))
    mx.eval(out)
    return out


def main():
    resp_path, model_dir, messages_path = sys.argv[1], sys.argv[2], sys.argv[3]
    resp = json.load(open(resp_path))
    messages = json.load(open(messages_path))

    choice = resp["choices"][0]
    lp_content = choice["logprobs"]["content"]
    print(f"Krill reported {len(lp_content)} token logprobs")

    model, tokenizer = load(model_dir)
    bytes_to_id, collisions = build_bytes_to_id(tokenizer)
    print(f"bytes->id table: {len(bytes_to_id)} unambiguous, "
          f"{len(collisions)} colliding byte-strings excluded")
    if collisions:
        sample = list(collisions.items())[:5]
        for b, entries in sample:
            print(f"  collision {b!r}: {entries}")

    prompt_ids, krill_ids, unresolved = load_ids(resp, messages, tokenizer, bytes_to_id)
    if unresolved:
        print(f"Unresolved sampled-token byte->id lookups ({len(unresolved)}): {unresolved[:10]}")

    # Teacher-forced full id sequence: Krill's own resolved ids where
    # available; for an unresolved position, re-derive via the model's own
    # greedy argmax at that step so the WALK stays on Krill's actual path
    # for every step after (this only matters for `full_sequence_logprobs`,
    # which needs concrete ids up front - the incremental loop below
    # recomputes on the fly).
    gen_ids = list(krill_ids)
    if any(t is None for t in gen_ids):
        # Resolve unresolved slots by walking incrementally with a cache,
        # using the greedy argmax only at the unresolved step itself.
        cache = make_prompt_cache(model)
        inp = mx.array([prompt_ids])
        logits = model(inp, cache=cache)
        resolved_ids = []
        for i, tid in enumerate(gen_ids):
            if tid is None:
                tid = int(mx.argmax(logits[0, -1, :]).item())
            resolved_ids.append(tid)
            if i < len(gen_ids) - 1:
                inp = mx.array([[tid]])
                logits = model(inp, cache=cache)
        gen_ids = resolved_ids

    full_ids = prompt_ids + gen_ids
    n_prompt = len(prompt_ids)

    # --- Floor: mlx_lm vs itself, full-sequence vs incremental ---
    method_a = full_sequence_logprobs(model, full_ids)  # position i -> next-token dist
    method_b = incremental_logprobs(model, prompt_ids, gen_ids)

    floor_max_diff = 0.0
    floor_detail = None
    for i, tid in enumerate(gen_ids):
        pos = n_prompt - 1 + i  # method_a's position that predicts gen_ids[i]
        a_lp = float(method_a[pos, tid].item())
        b_lp = float(method_b[i][tid].item())
        d = abs(a_lp - b_lp)
        if d > floor_max_diff:
            floor_max_diff = d
            floor_detail = (i, tid, a_lp, b_lp)
    print(f"\nmlx_lm intrinsic noise floor (full-sequence vs incremental, "
          f"{len(gen_ids)} positions): max abs diff = {floor_max_diff:.6e}")
    if floor_detail:
        print("  worst case (step, token id, full-seq logprob, incremental logprob):", floor_detail)

    # --- Krill vs incremental (apples-to-apples: Krill also decodes
    #     incrementally with a growing KV cache) ---
    max_abs_diff = 0.0
    max_abs_diff_detail = None
    top_alt_max_diff = 0.0
    top_alt_max_diff_detail = None
    n_alt_compared = 0
    n_sampled_compared = 0

    for i, entry in enumerate(lp_content):
        logprobs = method_b[i]
        krill_id = krill_ids[i] if i < len(krill_ids) else None
        if krill_id is None:
            continue
        krill_logprob = entry["logprob"]
        mlx_logprob = float(logprobs[krill_id].item())
        diff = abs(krill_logprob - mlx_logprob)
        n_sampled_compared += 1
        if diff > max_abs_diff:
            max_abs_diff = diff
            max_abs_diff_detail = (i, entry["token"], krill_logprob, mlx_logprob)

        for alt in entry.get("top_logprobs", []):
            alt_id = bytes_to_id.get(bytes(alt["bytes"]))
            if alt_id is None:
                continue
            mlx_alt_logprob = float(logprobs[alt_id].item())
            n_alt_compared += 1
            d = abs(alt["logprob"] - mlx_alt_logprob)
            if d > top_alt_max_diff:
                top_alt_max_diff = d
                top_alt_max_diff_detail = (i, alt["token"], alt["logprob"], mlx_alt_logprob)

    print(f"\nKrill vs mlx_lm INCREMENTAL (apples-to-apples) - "
          f"sampled-token max abs diff ({n_sampled_compared} compared): {max_abs_diff:.6e}")
    if max_abs_diff_detail:
        print("  worst case (step, token, krill, mlx):", max_abs_diff_detail)
    print(f"Krill vs mlx_lm INCREMENTAL - top-N alternate max abs diff "
          f"({n_alt_compared} compared): {top_alt_max_diff:.6e}")
    if top_alt_max_diff_detail:
        print("  worst case (step, token, krill, mlx):", top_alt_max_diff_detail)

    if floor_max_diff > 0:
        ratio = max_abs_diff / floor_max_diff
        print(f"\nKrill diff / floor ratio (sampled token): {ratio:.2f}x "
              f"({'within' if ratio <= 2.0 else 'ABOVE'} the ~2x expectation)")


if __name__ == "__main__":
    main()
