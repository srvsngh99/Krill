#!/usr/bin/env python3
"""Build a tiny *quantized* OLMoE checkpoint with mlx-lm and record its
reference logits, to check the native KrillLM `OLMoEForCausalLM` runtime for
logit parity against mlx-lm on identical weights.

Exercises the OLMoE-specific surfaces: the whole-projection q/k RMSNorm
(over n_heads*head_dim / n_kv_heads*head_dim, before the head reshape), the
router (softmax over all experts -> top-K, renorm only if norm_topk_prob),
and the `gatherQuantizedMM` SwitchGLU. Both runtimes are MLX, so parity is
~bit-exact.

Usage:
    python3 tools/verify_olmoe_parity.py /tmp/krillm-olmoe-parity
    KLM_OLMOE_PARITY_DIR=/tmp/krillm-olmoe-parity \\
        swift test -c release --filter OLMoEParityTests
"""
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.olmoe import Model, ModelArgs

GROUP_SIZE = 64
BITS = 4
TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]


def build(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    args = ModelArgs(
        model_type="olmoe",
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=320,
        num_experts=8,
        num_experts_per_tok=4,
        norm_topk_prob=False,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    mx.random.seed(0)
    model = Model(args)
    mx.eval(model.parameters())
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(model.parameters())

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(outdir, "model.safetensors"), weights)

    config = {
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "num_hidden_layers": args.num_hidden_layers,
        "vocab_size": args.vocab_size,
        "rms_norm_eps": args.rms_norm_eps,
        "rope_theta": args.rope_theta,
        "max_position_embeddings": 4096,
        "num_experts": args.num_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "norm_topk_prob": args.norm_topk_prob,
        "tie_word_embeddings": False,
        "quantization": {"group_size": GROUP_SIZE, "bits": BITS},
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

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

    print(f"Wrote tiny quantized OLMoE + reference logits to {outdir}")
    print(f"  argmax(last_token_logits) = {ref['argmax']}")
    qk = [k for k in sorted(weights) if "q_norm" in k or "k_norm" in k][:2]
    print("  sample q/k-norm keys:", *qk, sep="\n    ")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    build(sys.argv[1])
