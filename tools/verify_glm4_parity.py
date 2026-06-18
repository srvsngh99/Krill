#!/usr/bin/env python3
"""Build a tiny *quantized* GLM-4 (`Glm4ForCausalLM`, model_type "glm4")
checkpoint with mlx-lm and record its reference logits, so the native Krill
`Glm4ForCausalLM` runtime can be checked for logit parity against mlx-lm on the
exact same weights.

This is the OFFLINE TEST ORACLE only -- it never runs at inference time and is
not part of the Krill engine. The runtime itself is pure Swift + MLX
(Sources/KrillCore/Glm4Model.swift); this Python fixture only exists to gate it,
exactly like tools/verify_mixtral_parity.py gates the native Mixtral runtime.

Why tiny + synthetic: a real GLM-4-9B-0414 does not need to load to prove the
port is correct. A tiny random model exercises the exact code paths that are
easy to get wrong in the native GLM-4 port:
  - separate q/k/v/o projections WITH bias (attention_bias=True)
  - fused gate_up_proj SwiGLU MLP
  - SANDWICH NORM: four RMSNorms per layer -- input_layernorm,
    post_self_attn_layernorm (on the attn output, pre-residual),
    post_attention_layernorm (pre-MLP), post_mlp_layernorm (on the MLP output,
    pre-residual)
  - PARTIAL RoPE (partial_rotary_factor=0.5, traditional=True)
  - untied lm_head
against mlx-lm's reference on identical packed 4-bit tensors. Both runtimes use
MLX, so parity should be ~bit-exact.

The Glm4 weight-key layout this fixture emits is the GROUND TRUTH the Swift
@ModuleInfo keys must match. It is DIFFERENT from the legacy ChatGLM layout in
Sources/KrillCore/GLMModel.swift (transformer.encoder.layers.*.self_attention.
query_key_value / dense_h_to_4h / word_embeddings / output_layer).

Usage:
    python3 tools/verify_glm4_parity.py /tmp/krill-glm4-parity

Writes into that dir: config.json, model.safetensors, reference_logits.json
(the last-token logits for a fixed token sequence). Then run the gated Swift
test against it:
    KRILL_GLM4_PARITY_DIR=/tmp/krill-glm4-parity \\
        swift test -c release --filter Glm4ParityTests
"""
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.glm4 import Model, ModelArgs

GROUP_SIZE = 64
BITS = 4
# Fixed token sequence the Swift side replays.
TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]


def build(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    args = ModelArgs(
        model_type="glm4",
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        attention_bias=True,        # GLM-4 puts bias on q/k/v (and o per mlx-lm)
        partial_rotary_factor=0.5,  # only half of head_dim is rotated
        rope_theta=10000.0,
        rope_traditional=True,
        rms_norm_eps=1e-5,
        vocab_size=320,
        max_position_embeddings=4096,
    )
    mx.random.seed(0)
    model = Model(args)
    mx.eval(model.parameters())

    # Quantize uniformly (matches Krill's loadWeights quantize pass): every
    # Linear (q/k/v/o, gate_up, down, lm_head) + the embedding become
    # QuantizedLinear / QuantizedEmbedding on identical packed tensors.
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(model.parameters())

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(outdir, "model.safetensors"), weights)

    config = {
        "architectures": ["Glm4ForCausalLM"],
        "model_type": "glm4",
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "num_hidden_layers": args.num_hidden_layers,
        "head_dim": args.head_dim,
        "attention_bias": args.attention_bias,
        "partial_rotary_factor": args.partial_rotary_factor,
        "rope_theta": args.rope_theta,
        "rope_traditional": args.rope_traditional,
        "rms_norm_eps": args.rms_norm_eps,
        "vocab_size": args.vocab_size,
        "max_position_embeddings": args.max_position_embeddings,
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

    print(f"Wrote tiny quantized GLM-4 + reference logits to {outdir}")
    print(f"  argmax(last_token_logits) = {ref['argmax']}")
    keys = sorted(weights.keys())
    print(f"  {len(keys)} weight tensors. Layer-0 keys (the Swift @ModuleInfo spec):")
    for k in keys:
        if k.startswith("model.layers.0.") or k.startswith("model.embed") \
                or k == "lm_head.weight" or k == "model.norm.weight":
            print("    ", k, list(weights[k].shape))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    build(sys.argv[1])
