#!/usr/bin/env python3
"""Build a tiny random Nanbeige 4.2 (`NanbeigeForCausalLM`, model_type
"nanbeige") checkpoint, run it through the UPSTREAM PyTorch reference, and
record the resulting logits so the native Krill `NanbeigeForCausalLM` runtime
can be gated for logit parity on the exact same weights.

This is the OFFLINE TEST ORACLE only -- it never runs at inference time and is
not part of the Krill engine. The runtime itself is pure Swift + MLX
(Sources/KrillCore/NanbeigeModel.swift); this fixture only exists to gate it,
like tools/verify_glm4_parity.py gates the native GLM-4 runtime.

The oracle is the model's OWN `modeling_nanbeige.py` (mlx-lm has no nanbeige
port), so this gates against the authors' semantics rather than against a
reimplementation of them.

Why tiny + synthetic: a real Nanbeige4.2-3B does not need to load to prove the
port is correct. A tiny random model exercises exactly what is easy to get wrong
in this port:

  - **The LOOP.** `num_loops=2` runs the whole layer stack twice over the SAME
    weights. Getting this wrong (running once, or re-running with fresh weights)
    changes every logit.
  - **The per-loop KV cache.** The reference indexes the cache as
    `layer_idx + loop_idx * num_hidden_layers`, so a 2-layer/2-loop model holds
    FOUR cache entries. Sharing one cache per layer across loops silently
    corrupts decode - which is why this fixture records an incremental-decode
    reference (`step_logits`) in addition to the full-sequence prefill, and the
    Swift test replays both.
  - **The end-of-loop norm.** With `skip_loop_final_norm=false` (the shipped
    default) `model.norm` is applied at the end of EVERY loop, so loop 2 consumes
    a normed residual. Applying it only once at the end is a subtle, plausible,
    and wrong reading.
  - **head_dim decoupled from hidden_size.** The fixture deliberately sets
    `head_dim * num_attention_heads != hidden_size` (the real 3B does: 48*128 vs
    3072), so a runtime that derives `hidden_size / num_heads` mis-shapes q/o and
    fails to load.

Usage:
    python3 tools/verify_nanbeige_parity.py /tmp/krill-nanbeige-parity

Writes into that dir: config.json, model.safetensors, reference_logits.json.
Then run the gated Swift test against it:
    KRILL_NANBEIGE_PARITY_DIR=/tmp/krill-nanbeige-parity \\
        swift test --filter NanbeigeParityTests

`--source <dir>` points at a local snapshot of the HF repo supplying
`modeling_nanbeige.py` / `configuration_nanbeige.py` (defaults to
~/.cache/krill-nanbeige/Nanbeige4.2-3B).

`--real` instead writes `reference_logits.json` straight INTO that snapshot, so
the same Swift test gates the native runtime against the REAL 3B weights:
    python3 tools/verify_nanbeige_parity.py --real ignored
    KRILL_NANBEIGE_PARITY_DIR=~/.cache/krill-nanbeige/Nanbeige4.2-3B \\
        swift test --filter NanbeigeParityTests

!!! READ THIS BEFORE TRUSTING A `--real` RUN !!!

`NanbeigeRotaryEmbedding` registers `inv_freq` with `persistent=False`, so it is
NOT in the checkpoint. transformers 5.x materializes a `from_pretrained` model
from checkpoint tensors, which leaves that buffer at its ZERO initialization -
all 22 layers. RoPE then computes cos(0)=1, sin(0)=0 and applies NO rotation at
all, and the reference emits confidently positionless garbage. (Building the
model directly, as the synthetic fixture above does, computes `inv_freq`
normally - which is why only the `--real` path needs this.)

`repair_rope` recomputes the buffer and reports how many layers were zeroed. If
it ever reports 0 on a `--real` run, the upstream/transformers combination has
changed and the repair should be re-examined rather than silently dropped.
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile

import torch

# Fixed token sequence the Swift side replays.
TOKENS = [3, 11, 47, 2, 99, 5, 61, 7]

DEFAULT_SOURCE = os.path.expanduser("~/.cache/krill-nanbeige/Nanbeige4.2-3B")


def load_upstream(source: str):
    """Import the repo's `modeling_nanbeige` / `configuration_nanbeige`.

    They use RELATIVE imports (`from .configuration_nanbeige import ...`), so
    they only import as a package. Copy both into a temp package dir and import
    that, rather than mutating the snapshot.
    """
    for fn in ("modeling_nanbeige.py", "configuration_nanbeige.py"):
        if not os.path.exists(os.path.join(source, fn)):
            sys.exit(f"error: {fn} not found in {source}\n"
                     f"       pass --source <dir with the HF snapshot>")
    pkgdir = tempfile.mkdtemp(prefix="nanbeige_ref_")
    pkg = os.path.join(pkgdir, "nanbeige_ref")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    for fn in ("modeling_nanbeige.py", "configuration_nanbeige.py"):
        shutil.copy(os.path.join(source, fn), os.path.join(pkg, fn))
    sys.path.insert(0, pkgdir)
    from nanbeige_ref import configuration_nanbeige as cfg_mod  # noqa: E402
    from nanbeige_ref import modeling_nanbeige as mod_mod       # noqa: E402
    return cfg_mod, mod_mod


def repair_rope(model) -> None:
    """Recompute the zero-initialized `inv_freq` buffers. See the module docstring:
    without this the reference applies NO rotary embedding at all."""
    zeroed = 0
    layers = model.model.layers
    for layer in layers:
        r = layer.self_attn.rotary_emb
        inv = 1.0 / (r.base ** (torch.arange(0, r.dim, 2, dtype=torch.int64).float() / r.dim))
        if bool((r.inv_freq == 0).all()):
            zeroed += 1
        r.inv_freq.copy_(inv.to(r.inv_freq.dtype))
    print(f"repair_rope: recomputed inv_freq; {zeroed}/{len(layers)} layers were ZERO")
    if zeroed == 0:
        print("  note: nothing was zeroed - re-check whether this repair is still needed")


def build_real(source: str) -> None:
    """Emit reference logits for the REAL checkpoint, in place."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from transformers.cache_utils import DynamicCache

    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(source, trust_remote_code=True)
    # The shipped config has `rope_scaling: null`; transformers 5.x repopulates it
    # and the reference's `_init_rope` then indexes a missing "type" key.
    cfg.rope_scaling = None
    if hasattr(cfg, "rope_parameters"):
        cfg.rope_parameters = None
    model = AutoModelForCausalLM.from_pretrained(
        source, config=cfg, trust_remote_code=True, dtype=torch.float32).eval()
    repair_rope(model)

    text = "The capital of France is Paris. It is a major center of art and culture."
    tokens = tok(text, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([tokens])
    with torch.no_grad():
        full = model(ids, use_cache=False).logits[0]
        prefill_last = full[-1]
        cache = DynamicCache()
        model(ids[:, :-1], past_key_values=cache, use_cache=True)
        step = model(ids[:, -1:], past_key_values=cache, use_cache=True).logits[0, -1]

    print(f"cache entries = {len(cache)} "
          f"(expected {cfg.num_loops * cfg.num_hidden_layers})")
    print(f"reference self-consistency = {(prefill_last - step).abs().max().item():.3e}")

    out = os.path.join(source, "reference_logits.json")
    with open(out, "w") as f:
        json.dump({
            "tokens": tokens,
            "vocab_size": int(cfg.vocab_size),
            "num_loops": int(cfg.num_loops),
            "num_hidden_layers": int(cfg.num_hidden_layers),
            "expected_cache_entries": int(cfg.num_loops * cfg.num_hidden_layers),
            "last_token_logits": prefill_last.tolist(),
            "argmax": int(prefill_last.argmax().item()),
            "step_logits": step.tolist(),
            # The checkpoint is bf16 and Krill computes in bf16, while this
            # reference runs fp32; `num_loops` doubles the layer executions that
            # accumulate rounding. Argmax + cosine > 0.9999 are the real gate.
            "max_abs_tolerance": 0.75,
        }, f)
    print(f"wrote {out}\nargmax = {int(prefill_last.argmax().item())}")
    print(f"\nNow run:\n  KRILL_NANBEIGE_PARITY_DIR={source} "
          f"swift test --filter NanbeigeParityTests")


def build(outdir: str, source: str) -> None:
    cfg_mod, mod_mod = load_upstream(source)
    os.makedirs(outdir, exist_ok=True)

    # Deliberately decoupled: 4 heads * 64 dims = 256 != hidden_size 128, so a
    # runtime deriving head_dim = hidden/heads (32) mis-shapes q_proj and o_proj.
    # head_dim stays 64 because MLX's fused SDPA only fast-paths a set of head
    # dims (64/80/96/128 - the real model's 128 is among them); an exotic width
    # like 24 returns zeros from the cached-decode path and would make this
    # fixture test MLX's kernel coverage rather than the port.
    config = cfg_mod.NanbeigeConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=128,
        num_loops=2,               # THE looped-transformer path
        skip_loop_final_norm=False,  # norm at the end of every loop
        rms_norm_eps=1e-5,
        rope_theta=70000000.0,       # the real model's unusual theta
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        attention_bias=False,
        qk_layernorm=False,
        torch_dtype="float32",
    )

    # transformers 5.x auto-populates `rope_scaling` / `rope_parameters` on the
    # base config, but the shipped Nanbeige config.json has `rope_scaling: null`
    # and the reference's `_init_rope` takes its plain-RoPE branch only when the
    # attribute is None. Force it back so the fixture exercises the SHIPPED path.
    config.rope_scaling = None
    if hasattr(config, "rope_parameters"):
        config.rope_parameters = None

    torch.manual_seed(0)
    model = mod_mod.NanbeigeForCausalLM(config).to(torch.float32).eval()
    # Random init: the default init leaves norms at 1.0 and would hide a
    # misplaced-norm bug, so give every parameter real structure.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.08 + 0.02)

    ids = torch.tensor([TOKENS], dtype=torch.long)

    with torch.no_grad():
        # (a) Full-sequence prefill, no cache.
        full = model(ids, use_cache=False).logits[0]          # [L, V]
        prefill_last = full[-1]

        # (b) Incremental decode: prefill all but the last token WITH cache,
        # then step the final token. This is what catches per-loop cache
        # indexing: with num_loops=2 the cache must hold
        # num_loops * num_hidden_layers entries, and loop 2 must not overwrite
        # loop 1's keys/values. The stepped logits must equal the prefill's
        # last-token logits.
        # Hand in an explicit `Cache`. Left as None, the reference takes its
        # legacy-tuple branch (`DynamicCache.from_legacy_cache`), which
        # transformers 5.x removed; passing a real Cache skips that dead path and
        # exercises the same per-loop cache indexing the runtime must reproduce.
        from transformers.cache_utils import DynamicCache
        pre = model(ids[:, :-1], use_cache=True, past_key_values=DynamicCache())
        step = model(ids[:, -1:], past_key_values=pre.past_key_values,
                     use_cache=True).logits[0, -1]
        cache_len = len(pre.past_key_values)

    drift = (prefill_last - step).abs().max().item()
    print(f"reference self-consistency (prefill vs cached step) max|d| = {drift:.3e}")
    if drift > 1e-3:
        sys.exit(f"error: the REFERENCE disagrees with itself by {drift:.3e}; "
                 f"the fixture is not trustworthy")
    print(f"reference cache entries = {cache_len} "
          f"(expected {config.num_loops * config.num_hidden_layers})")

    # Save weights. safetensors rejects shared storage, so clone every tensor.
    from safetensors.torch import save_file
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(outdir, "model.safetensors"))

    cfg = config.to_dict()
    cfg["architectures"] = ["NanbeigeForCausalLM"]
    cfg["model_type"] = "nanbeige"
    cfg["torch_dtype"] = "float32"
    # `auto_map` would make Krill's loader look for remote code it does not run.
    cfg.pop("auto_map", None)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    ref = {
        "tokens": TOKENS,
        "vocab_size": config.vocab_size,
        "num_loops": config.num_loops,
        "num_hidden_layers": config.num_hidden_layers,
        "expected_cache_entries": config.num_loops * config.num_hidden_layers,
        "last_token_logits": prefill_last.tolist(),
        "argmax": int(prefill_last.argmax().item()),
        "step_logits": step.tolist(),
    }
    with open(os.path.join(outdir, "reference_logits.json"), "w") as f:
        json.dump(ref, f)

    print(f"wrote {outdir}/config.json, model.safetensors, reference_logits.json")
    print(f"argmax = {ref['argmax']}")
    print(f"\nNow run:\n  KRILL_NANBEIGE_PARITY_DIR={outdir} "
          f"swift test --filter NanbeigeParityTests")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", nargs="?", default=None,
                    help="directory to write the synthetic fixture into")
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help=f"HF snapshot supplying the reference code "
                         f"(default: {DEFAULT_SOURCE})")
    ap.add_argument("--real", action="store_true",
                    help="gate against the REAL checkpoint: write "
                         "reference_logits.json into --source instead of "
                         "building a synthetic fixture")
    args = ap.parse_args()
    if args.real:
        build_real(args.source)
    elif args.outdir:
        build(args.outdir, args.source)
    else:
        ap.error("give an outdir, or pass --real")


if __name__ == "__main__":
    main()
