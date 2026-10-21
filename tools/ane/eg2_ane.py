"""Trace-friendly, ANE-oriented reimplementation of the EmbeddingGemma-2 TEXT tower (mirrors HF modeling_embedding_gemma2).
Used only by tools/ane/build_ane_embedder.py (offline). The Krill runtime is Swift-only.
Static shapes (B,S). Inputs: embeds [B,S,H] (already * sqrt(H)) OR input_ids [B,S] int32 (embed_in_graph), mask [B,S] fp (1=real).
Output: L2-normalised [B,768] (mean pool incl. prompt, then projection; equals project-then-pool, no bias).
Sliding window 512 is a no-op for S<=512 (|i-j|<512 always), so only the padding bias is applied."""
import json, math, torch, torch.nn as nn, torch.nn.functional as F

class RMS(nn.Module):
    """RMSNorm with pre-scale c so x^2 cannot overflow fp16 (255^2 = 65k). y = x * rsqrt(mean(x^2)+eps) * w"""
    def __init__(s, d, eps=1e-6, scale=True, c=1/64.):
        super().__init__(); s.eps=eps; s.c=c; s.w = nn.Parameter(torch.ones(d)) if scale else None
    def forward(s, x):
        xs = x * s.c
        inv = torch.rsqrt((xs*xs).mean(-1, keepdim=True) + s.eps * s.c * s.c)   # = (1/c) * rsqrt(ms+eps)
        y = xs * inv                                                         # = x * rsqrt(ms+eps)
        return y * s.w if s.w is not None else y

def rot_half(x):
    a, b = torch.chunk(x, 2, -1)
    return torch.cat([-b, a], -1)

class Attn(nn.Module):
    def __init__(s, H, nh, nkv, hd, eps):
        super().__init__(); s.nh, s.nkv, s.hd = nh, nkv, hd
        s.q_proj = nn.Linear(H, nh*hd, bias=False); s.k_proj = nn.Linear(H, nkv*hd, bias=False)
        s.v_proj = nn.Linear(H, nkv*hd, bias=False); s.o_proj = nn.Linear(nh*hd, H, bias=False)
        s.q_norm = RMS(hd, eps); s.k_norm = RMS(hd, eps); s.v_norm = RMS(hd, eps, scale=False)
    B = 1; S = 1
    def forward(s, x, cos, sin, bias):
        B, S = s.B, s.S
        q = s.q_proj(x).view(B, S, s.nh, s.hd).transpose(1, 2)
        k = s.k_proj(x).view(B, S, s.nkv, s.hd).transpose(1, 2)
        v = s.v_proj(x).view(B, S, s.nkv, s.hd).transpose(1, 2)
        q = s.q_norm(q); k = s.k_norm(k); v = s.v_norm(v)
        q = q*cos + rot_half(q)*sin; k = k*cos + rot_half(k)*sin
        g = s.nh // s.nkv
        if g > 1:
            k = torch.cat([k[:, i:i+1].expand(B, g, S, s.hd) for i in range(s.nkv)], 1)
            v = torch.cat([v[:, i:i+1].expand(B, g, S, s.hd) for i in range(s.nkv)], 1)
        a = torch.matmul(q, k.transpose(2, 3)) + bias          # scaling = 1.0
        a = torch.softmax(a, -1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, S, s.nh*s.hd)
        return s.o_proj(o)

class Layer(nn.Module):
    def __init__(s, c, i):
        super().__init__(); H = c["hidden_size"]; eps = c["rms_norm_eps"]
        pl = c.get("per_layer_config", {}).get(f"{i:02d}", {})
        hd = pl.get("head_dim", c["head_dim"]); nkv = pl.get("num_key_value_heads", c["num_key_value_heads"])
        s.attn = Attn(H, c["num_attention_heads"], nkv, hd, eps)
        s.gate_proj = nn.Linear(H, c["intermediate_size"], bias=False); s.up_proj = nn.Linear(H, c["intermediate_size"], bias=False)
        s.down_proj = nn.Linear(c["intermediate_size"], H, bias=False)
        s.input_ln = RMS(H, eps); s.post_attn_ln = RMS(H, eps); s.pre_ff_ln = RMS(H, eps); s.post_ff_ln = RMS(H, eps)
        P = c["hidden_size_per_layer_input"]
        s.ple_gate = nn.Linear(H, P, bias=False); s.ple_proj = nn.Linear(P, H, bias=False); s.ple_norm = RMS(H, eps)
        s.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)
        s.full = c["layer_types"][i] == "full_attention"
    def forward(s, h, pli, cos, sin, bias):
        h = h + s.post_attn_ln(s.attn(s.input_ln(h), cos, sin, bias))
        y = s.pre_ff_ln(h)
        y = s.down_proj(F.gelu(s.gate_proj(y), approximate="tanh") * s.up_proj(y))
        h = h + s.post_ff_ln(y)
        g = F.gelu(s.ple_gate(h), approximate="tanh") * pli
        h = h + s.ple_norm(s.ple_proj(g))
        return h * s.layer_scalar

class EG2(nn.Module):
    def __init__(s, tc, S, B=1, embed_in_graph=False, rms_c=1/64.):
        super().__init__(); s.tc=tc; s.S=S; s.B=B; s.eig=embed_in_graph
        H = tc["hidden_size"]; eps = tc["rms_norm_eps"]; L = tc["num_hidden_layers"]; P = tc["hidden_size_per_layer_input"]
        s.H=H; s.L=L; s.P=P
        s.embed = nn.Embedding(tc["vocab_size"], H) if embed_in_graph else None
        s.layers = nn.ModuleList([Layer(tc, i) for i in range(L)])
        s.ple_model_proj = nn.Linear(H, L*P, bias=False); s.ple_norm = RMS(P, eps)
        s.norm = RMS(H, eps); s.emb_proj = nn.Linear(H, tc["embedding_dim"], bias=False)
        for m in s.modules():
            if isinstance(m, RMS): m.c = rms_c
            if isinstance(m, Attn): m.B=B; m.S=S
        s.types = {}
        for t in ("sliding_attention", "full_attention"):
            idx = tc["layer_types"].index(t) if t in tc["layer_types"] else None
            if idx is None: continue
            pl = tc.get("per_layer_config", {}).get(f"{idx:02d}", {}); hd = pl.get("head_dim", tc["head_dim"])
            base = tc["rope_parameters"][t]["rope_theta"]
            inv = 1.0/(base**(torch.arange(0, hd, 2).float()/hd)); fr = torch.arange(S).float()[:, None]*inv[None]
            e = torch.cat([fr, fr], -1)
            s.register_buffer(f"cos_{t[:4]}", e.cos()[None, None]); s.register_buffer(f"sin_{t[:4]}", e.sin()[None, None])
    def forward(s, x, mask):
        if s.eig: x = s.embed(x) * (s.H**0.5)
        pl = s.ple_norm((s.ple_model_proj(x) * (s.H**-0.5)).view(s.B, s.S, s.L, s.P))
        bias = ((1.0 - mask) * -30000.0)[:, None, None, :]
        h = x
        for i, l in enumerate(s.layers):
            t = "full" if l.full else "slid"
            h = l(h, pl[:, :, i], getattr(s, f"cos_{t}"), getattr(s, f"sin_{t}"), bias)
        h = s.norm(h)
        m = mask[:, :, None]
        pooled = (h*m).sum(1) / m.sum(1)
        e = s.emb_proj(pooled)
        return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)

def load_weights(model, sd):
    """sd: dict of language_model.* tensors (HF names)."""
    def g(k): return sd["language_model."+k].float()
    with torch.no_grad():
        if model.embed is not None: model.embed.weight.copy_(g("embed_tokens.weight"))
        model.emb_proj.weight.copy_(g("embedding_projection.weight")); model.norm.w.copy_(g("norm.weight"))
        model.ple_model_proj.weight.copy_(g("ple.per_layer_model_projection.weight")); model.ple_norm.w.copy_(g("ple.per_layer_projection_norm.weight"))
        for i, l in enumerate(model.layers):
            p = f"layers.{i}."
            for a, b in [("input_ln","input_layernorm"),("post_attn_ln","post_attention_layernorm"),("pre_ff_ln","pre_feedforward_layernorm"),("post_ff_ln","post_feedforward_layernorm")]:
                getattr(l, a).w.copy_(g(p+b+".weight"))
            for n in ("q_proj","k_proj","v_proj","o_proj"): getattr(l.attn, n).weight.copy_(g(p+f"self_attn.{n}.weight"))
            l.attn.q_norm.w.copy_(g(p+"self_attn.q_norm.weight")); l.attn.k_norm.w.copy_(g(p+"self_attn.k_norm.weight"))
            for n in ("gate_proj","up_proj","down_proj"): getattr(l, n).weight.copy_(g(p+f"mlp.{n}.weight"))
            l.ple_gate.weight.copy_(g(p+"ple_block.per_layer_input_gate.weight")); l.ple_proj.weight.copy_(g(p+"ple_block.per_layer_projection.weight"))
            l.ple_norm.w.copy_(g(p+"ple_block.post_per_layer_input_norm.weight")); l.layer_scalar.copy_(g(p+"layer_scalar"))

def text_cfg(path):
    return json.load(open(path))["text_config"]
