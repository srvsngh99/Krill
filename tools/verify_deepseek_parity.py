#!/usr/bin/env python3
"""Build a tiny *quantized* DeepSeek checkpoint with mlx-lm and record its
reference logits, to check the native KrillLM `DeepSeekForCausalLM` runtime
for logit parity against mlx-lm on identical weights.

Two variants exercise both gating paths and both MLA Q paths:
  v2 - model_type deepseek_v2: softmax/greedy router, direct q_proj (no
       q_lora bottleneck), shared expert, first_k_dense_replace.
  v3 - model_type deepseek_v3: noaux_tc sigmoid router + e_score_correction_bias
       + group top-2-sum selection + norm_topk_prob, q_lora bottleneck.

Both share MLA (low-rank KV, split rope/nope head dims) and YaRN RoPE. Real
DeepSeek-V2-Lite fits on this host; DeepSeek-V3 671B does not, so a tiny
synthetic V3 stands in for its gating numerics. Both runtimes are MLX, so
parity is ~bit-exact.

Usage:
    python3 tools/verify_deepseek_parity.py /tmp/krillm-deepseek-v2 v2
    python3 tools/verify_deepseek_parity.py /tmp/krillm-deepseek-v3 v3
    KLM_DEEPSEEK_V2_PARITY_DIR=/tmp/krillm-deepseek-v2 \\
        KLM_DEEPSEEK_V3_PARITY_DIR=/tmp/krillm-deepseek-v3 \\
        swift test -c release --filter DeepSeekParityTests
"""
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

GROUP_SIZE = 64
BITS = 4
TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]

ROPE_SCALING = {
    "type": "yarn",
    "factor": 4.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 0.0,
    "original_max_position_embeddings": 256,
}


def build(outdir: str, variant: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    common = dict(
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=320,
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        v_head_dim=64,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=ROPE_SCALING,
    )
    if variant == "v2":
        from mlx_lm.models.deepseek_v2 import Model, ModelArgs
        args = ModelArgs(
            model_type="deepseek_v2",
            q_lora_rank=None,
            topk_method="greedy",
            n_group=1,
            topk_group=1,
            attention_bias=False,
            **common,
        )
        config_arch = "DeepseekV2ForCausalLM"
        extra = {"scoring_func": "softmax", "norm_topk_prob": False}
    elif variant == "v3":
        from mlx_lm.models.deepseek_v3 import Model, ModelArgs
        args = ModelArgs(
            model_type="deepseek_v3",
            q_lora_rank=128,
            topk_method="noaux_tc",
            scoring_func="sigmoid",
            norm_topk_prob=True,
            n_group=2,
            topk_group=1,
            attention_bias=False,
            **common,
        )
        config_arch = "DeepseekV3ForCausalLM"
        extra = {"scoring_func": "sigmoid", "norm_topk_prob": True}
    else:
        raise SystemExit(f"unknown variant {variant!r} (expected v2 or v3)")

    mx.random.seed(0)
    model = Model(args)
    # The MoE router gate (`MoEGate.weight`) and `e_score_correction_bias`
    # initialize to zeros and are not randomized by model construction. Left at
    # zero every routed score is sigmoid(0) = 0.5, so the group-select scores
    # tie exactly and the expert selection is decided purely by the runtime's
    # tie-break order -- which is an artifact, not the numerics a real
    # checkpoint exercises. Randomize them (V3 only; V2 has no group select) so
    # the synthetic model routes through a non-degenerate `noaux_tc` gate.
    if variant == "v3":
        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "gate"):
                mlp.gate.weight = mx.random.normal(mlp.gate.weight.shape) * 0.02
                mlp.gate.e_score_correction_bias = (
                    mx.random.normal(mlp.gate.e_score_correction_bias.shape) * 0.02)
    mx.eval(model.parameters())
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(model.parameters())

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(outdir, "model.safetensors"), weights)

    config = {
        "architectures": [config_arch],
        "model_type": args.model_type,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "moe_intermediate_size": args.moe_intermediate_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "vocab_size": args.vocab_size,
        "rms_norm_eps": args.rms_norm_eps,
        "rope_theta": args.rope_theta,
        "max_position_embeddings": args.max_position_embeddings,
        "kv_lora_rank": args.kv_lora_rank,
        "qk_rope_head_dim": args.qk_rope_head_dim,
        "qk_nope_head_dim": args.qk_nope_head_dim,
        "v_head_dim": args.v_head_dim,
        "n_routed_experts": args.n_routed_experts,
        "n_shared_experts": args.n_shared_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "routed_scaling_factor": args.routed_scaling_factor,
        "topk_method": args.topk_method,
        "n_group": args.n_group,
        "topk_group": args.topk_group,
        "first_k_dense_replace": args.first_k_dense_replace,
        "moe_layer_freq": args.moe_layer_freq,
        "attention_bias": False,
        "rope_scaling": ROPE_SCALING,
        "quantization": {"group_size": GROUP_SIZE, "bits": BITS},
        **extra,
    }
    if variant == "v3":
        config["q_lora_rank"] = args.q_lora_rank
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logits = model(mx.array([TOKENS]))
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

    print(f"Wrote tiny quantized DeepSeek-{variant} + reference logits to {outdir}")
    print(f"  argmax(last_token_logits) = {ref['argmax']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    build(sys.argv[1], sys.argv[2])
