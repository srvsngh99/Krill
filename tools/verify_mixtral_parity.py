#!/usr/bin/env python3
"""Build a tiny *quantized* Mixtral checkpoint with mlx-lm and record its
reference logits, so the native Krill `MixtralForCausalLM` runtime can be
checked for logit parity against mlx-lm on the exact same weights.

Why tiny + synthetic: a real Mixtral-8x7B (~24 GB at 4-bit) does not fit on a
24 GB host. A tiny random model exercises the same code paths that are easy to
get wrong in the native port -- the `block_sparse_moe` router
(softmax-over-all -> top-k -> renormalize) and the `gatherQuantizedMM`
SwitchGLU expert dispatch -- against mlx-lm's reference on identical packed
4-bit expert tensors. Both runtimes use MLX, so parity should be ~bit-exact.

Usage:
    python3 tools/verify_mixtral_parity.py /tmp/krill-mixtral-parity

Writes into that dir: config.json, model.safetensors, reference_logits.json
(the last-token logits for a fixed token sequence). Then run the gated Swift
test against it:
    KLM_MIXTRAL_PARITY_DIR=/tmp/krill-mixtral-parity \\
        swift test -c release --filter MixtralParityTests
"""
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.mixtral import Model, ModelArgs

GROUP_SIZE = 64
BITS = 4
# Fixed token sequence the Swift side replays.
TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]


def build(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    args = ModelArgs(
        model_type="mixtral",
        vocab_size=320,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts_per_tok=2,
        num_local_experts=8,
        rms_norm_eps=1e-5,
        rope_theta=1_000_000.0,
        tie_word_embeddings=False,
    )
    mx.random.seed(0)
    model = Model(args)
    mx.eval(model.parameters())

    # Quantize uniformly (matches Krill's loadWeights quantize pass). The
    # MoE SwitchGLU SwitchLinears become QuantizedSwitchLinear; the router
    # gate, embeddings, lm_head, and attention Linears become QuantizedLinear.
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(model.parameters())

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(outdir, "model.safetensors"), weights)

    config = {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "num_hidden_layers": args.num_hidden_layers,
        "vocab_size": args.vocab_size,
        "rms_norm_eps": args.rms_norm_eps,
        "rope_theta": args.rope_theta,
        "max_position_embeddings": 4096,
        "num_local_experts": args.num_local_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "tie_word_embeddings": False,
        "quantization": {"group_size": GROUP_SIZE, "bits": BITS},
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Reference logits for the fixed token sequence: full last-token row.
    tokens = mx.array([TOKENS])
    logits = model(tokens)
    mx.eval(logits)
    last = logits[0, -1, :]
    ref = {
        "tokens": TOKENS,
        "vocab_size": args.vocab_size,
        "last_token_logits": [float(v) for v in last.tolist()],
        "argmax": int(mx.argmax(last).item()),
    }
    with open(os.path.join(outdir, "reference_logits.json"), "w") as f:
        json.dump(ref, f)

    print(f"Wrote tiny quantized Mixtral + reference logits to {outdir}")
    print(f"  argmax(last_token_logits) = {ref['argmax']}")
    keys = sorted(weights.keys())
    sample = [k for k in keys if "block_sparse_moe" in k][:4]
    print("  sample MoE keys:", *sample, sep="\n    ")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    build(sys.argv[1])
