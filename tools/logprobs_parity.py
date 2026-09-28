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

Usage: logprobs_parity.py <krill_response.json> <model_dir> <messages.json>
"""
import json
import sys

import mlx.core as mx
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
    byte_decoder = gpt2_byte_decoder()
    vocab = tokenizer._tokenizer.get_vocab()
    table = {}
    for piece, tok_id in vocab.items():
        table[piece_to_bytes(piece, byte_decoder)] = tok_id
    return table


def main():
    resp_path, model_dir, messages_path = sys.argv[1], sys.argv[2], sys.argv[3]
    resp = json.load(open(resp_path))
    messages = json.load(open(messages_path))

    choice = resp["choices"][0]
    lp_content = choice["logprobs"]["content"]
    print(f"Krill reported {len(lp_content)} token logprobs")

    model, tokenizer = load(model_dir)
    bytes_to_id = build_bytes_to_id(tokenizer)

    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    ids = list(prompt_ids)

    max_abs_diff = 0.0
    max_abs_diff_detail = None
    top_alt_max_diff = 0.0
    top_alt_max_diff_detail = None
    unresolved = []
    n_alt_compared = 0
    n_sampled_compared = 0

    for i, entry in enumerate(lp_content):
        inp = mx.array([ids])
        logits = model(inp)
        last = logits[0, -1, :].astype(mx.float32)
        logprobs = last - mx.logsumexp(last)

        greedy_id = int(mx.argmax(last).item())

        krill_id = bytes_to_id.get(bytes(entry["bytes"]))
        if krill_id is None:
            unresolved.append(("sampled", i, entry["token"], entry["bytes"]))
        else:
            if krill_id != greedy_id:
                print(f"  NOTE step {i}: krill sampled id {krill_id} "
                      f"({entry['token']!r}) != mlx greedy id {greedy_id} "
                      f"({tokenizer.decode([greedy_id])!r}) - comparing krill's own id's logprob anyway")
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
                unresolved.append(("alt", i, alt["token"], alt["bytes"]))
                continue
            mlx_alt_logprob = float(logprobs[alt_id].item())
            n_alt_compared += 1
            d = abs(alt["logprob"] - mlx_alt_logprob)
            if d > top_alt_max_diff:
                top_alt_max_diff = d
                top_alt_max_diff_detail = (i, alt["token"], alt["logprob"], mlx_alt_logprob)

        ids.append(greedy_id)

    print(f"Sampled-token max abs logprob diff ({n_sampled_compared} compared): {max_abs_diff:.6e}")
    if max_abs_diff_detail:
        print("  worst case (step, token, krill, mlx):", max_abs_diff_detail)
    print(f"Top-N alternate max abs logprob diff ({n_alt_compared} compared): {top_alt_max_diff:.6e}")
    if top_alt_max_diff_detail:
        print("  worst case (step, token, krill, mlx):", top_alt_max_diff_detail)
    if unresolved:
        print(f"Unresolved byte->id lookups ({len(unresolved)}): {unresolved[:10]}")


if __name__ == "__main__":
    main()
