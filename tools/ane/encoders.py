"""Trace-friendly torch encoders for the Neural Engine packages (offline; the Krill runtime is Swift only).

Each class mirrors the matching Krill module in Sources/KrillCore (the semantic reference for the GPU path) and
the HF checkpoint key names, so `load_state_dict` takes the checkpoint as is. Why hand-written rather than
`torch.jit.trace(AutoModel)`: nomic_bert, gte "new" and jina-bert are custom-code architectures that transformers
cannot load offline, and the HF masks/SDPA do not trace to a fixed-shape fp16 graph. Static shapes (B=1, S);
inputs: embeds [1,S,H] (the host already gathered the word-embedding rows), mask [1,S] (1 = real token).
Everything position related (absolute / rotary / ALiBi / relative bias / sliding window) is a constant buffer.
"""
import json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

NEG = -30000.0   # fp16-safe additive mask


def pad_bias(mask):
    return ((1.0 - mask) * NEG)[:, None, None, :]


class LN(nn.Module):
    """LayerNorm; `c` pre-scales the input (LN is scale invariant) so x^2 cannot overflow fp16."""
    def __init__(s, d, eps, bias=True, c=1.0):
        super().__init__(); s.d, s.eps, s.c = d, eps, c
        s.weight = nn.Parameter(torch.ones(d)); s.bias = nn.Parameter(torch.zeros(d)) if bias else None
    def forward(s, x):
        if s.c != 1.0:
            return F.layer_norm(x * s.c, (s.d,), s.weight, s.bias, s.eps * s.c * s.c)
        return F.layer_norm(x, (s.d,), s.weight, s.bias, s.eps)


class RMS(nn.Module):
    def __init__(s, d, eps, c=1 / 64.):
        super().__init__(); s.eps, s.c = eps, c; s.weight = nn.Parameter(torch.ones(d))
    def forward(s, x):
        xs = x * s.c
        return xs * torch.rsqrt((xs * xs).mean(-1, keepdim=True) + s.eps * s.c * s.c) * s.weight


def rot_half(x):
    a, b = torch.chunk(x, 2, -1)
    return torch.cat([-b, a], -1)


def rope_tables(S, dim, inv_freq):
    fr = torch.arange(S).float()[:, None] * inv_freq[None]
    e = torch.cat([fr, fr], -1)
    return e.cos()[None, None], e.sin()[None, None]      # [1,1,S,dim]


def apply_rope(x, cos, sin):
    r = cos.shape[-1]
    if r == x.shape[-1]:
        return x * cos + rot_half(x) * sin
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * cos + rot_half(xr) * sin, xp], -1)


def sdpa(q, k, v, bias, scale):
    a = torch.matmul(q * scale, k.transpose(2, 3)) + bias
    return torch.matmul(torch.softmax(a, -1), v)


def heads(x, nh, S):
    return x.view(1, S, nh, -1).transpose(1, 2)


def merge(x, S):
    return x.transpose(1, 2).reshape(1, S, -1)


class Lin(nn.Linear):
    pass


# ---------------------------------------------------------------- BERT / RoBERTa / XLM-R (+ cross-encoder head)
class BertSelf(nn.Module):
    def __init__(s, H):
        super().__init__(); s.query, s.key, s.value = Lin(H, H), Lin(H, H), Lin(H, H)
class BertAttOut(nn.Module):
    def __init__(s, H, eps, c):
        super().__init__(); s.dense = Lin(H, H); s.LayerNorm = LN(H, eps, c=c)
class BertAtt(nn.Module):
    def __init__(s, H, eps, c):
        super().__init__(); setattr(s, "self", BertSelf(H)); s.output = BertAttOut(H, eps, c)
class BertInter(nn.Module):
    def __init__(s, H, I):
        super().__init__(); s.dense = Lin(H, I)
class BertOut(nn.Module):
    def __init__(s, H, I, eps, c):
        super().__init__(); s.dense = Lin(I, H); s.LayerNorm = LN(H, eps, c=c)
class BertLayer(nn.Module):
    def __init__(s, cfg, c):
        super().__init__(); H, I, eps = cfg["hidden_size"], cfg["intermediate_size"], cfg.get("layer_norm_eps", 1e-12)
        s.attention = BertAtt(H, eps, c); s.intermediate = BertInter(H, I); s.output = BertOut(H, I, eps, c)
        s.nh = cfg["num_attention_heads"]; s.act = cfg.get("hidden_act", "gelu")
    def forward(s, x, bias, S):
        a = getattr(s.attention, "self")
        q, k, v = heads(a.query(x), s.nh, S), heads(a.key(x), s.nh, S), heads(a.value(x), s.nh, S)
        o = merge(sdpa(q, k, v, bias, 1.0 / math.sqrt(q.shape[-1])), S)
        h = s.attention.output.LayerNorm(s.attention.output.dense(o) + x)
        i = s.intermediate.dense(h)
        i = F.gelu(i, approximate="tanh") if s.act == "gelu_new" else F.gelu(i)
        return s.output.LayerNorm(s.output.dense(i) + h)
class BertEmb(nn.Module):
    def __init__(s, cfg, c):
        super().__init__(); H = cfg["hidden_size"]
        s.position_embeddings = nn.Embedding(cfg["max_position_embeddings"], H)
        s.token_type_embeddings = nn.Embedding(max(cfg.get("type_vocab_size", 2), 1), H)
        s.LayerNorm = LN(H, cfg.get("layer_norm_eps", 1e-12), c=c)
class BertEnc(nn.Module):
    def __init__(s, cfg, c):
        super().__init__(); s.layer = nn.ModuleList([BertLayer(cfg, c) for _ in range(cfg["num_hidden_layers"])])
class ClsHead(nn.Module):
    def __init__(s, H):
        super().__init__(); s.dense = Lin(H, H); s.out_proj = Lin(H, 1)
class Bert(nn.Module):
    def __init__(s, cfg, S, c=1.0, cross_encoder=False):
        super().__init__(); s.S = S
        s.embeddings = BertEmb(cfg, c); s.encoder = BertEnc(cfg, c)
        s.classifier = ClsHead(cfg["hidden_size"]) if cross_encoder else None
        off = (cfg.get("pad_token_id", 1) + 1) if cfg.get("model_type") in ("roberta", "xlm-roberta") else 0
        s.off = off
        s.register_buffer("pos_ids", torch.arange(off, off + S))
    def forward(s, embeds, mask):
        e = s.embeddings
        x = embeds + (e.position_embeddings(s.pos_ids) + e.token_type_embeddings.weight[0])[None]
        x = e.LayerNorm(x)
        bias = pad_bias(mask)
        for l in s.encoder.layer:
            x = l(x, bias, s.S)
        return x
    def head(s, x):
        c = s.classifier
        return c.out_proj(torch.tanh(c.dense(x[:, 0])))


# ---------------------------------------------------------------- MPNet (relative attention bias)
class MPAtt(nn.Module):
    def __init__(s, H, eps, c):
        super().__init__(); s.attn = nn.Module(); s.attn.q, s.attn.k, s.attn.v, s.attn.o = Lin(H, H), Lin(H, H), Lin(H, H), Lin(H, H)
        s.LayerNorm = LN(H, eps, c=c)
class MPLayer(nn.Module):
    def __init__(s, cfg, c):
        super().__init__(); H, I, eps = cfg["hidden_size"], cfg["intermediate_size"], cfg.get("layer_norm_eps", 1e-5)
        s.attention = MPAtt(H, eps, c); s.intermediate = BertInter(H, I); s.output = BertOut(H, I, eps, c); s.nh = cfg["num_attention_heads"]
    def forward(s, x, bias, S):
        a = s.attention.attn
        q, k, v = heads(a.q(x), s.nh, S), heads(a.k(x), s.nh, S), heads(a.v(x), s.nh, S)
        att = s.attention.LayerNorm(a.o(merge(sdpa(q, k, v, bias, 1.0 / math.sqrt(q.shape[-1])), S)) + x)
        return s.output.LayerNorm(s.output.dense(F.gelu(s.intermediate.dense(att))) + att)
class MPEnc(nn.Module):
    def __init__(s, cfg, c):
        super().__init__(); s.layer = nn.ModuleList([MPLayer(cfg, c) for _ in range(cfg["num_hidden_layers"])])
        s.relative_attention_bias = nn.Embedding(cfg.get("relative_attention_num_buckets", 32), cfg["num_attention_heads"])
class MPNet(nn.Module):
    def __init__(s, cfg, S, c=1.0):
        super().__init__(); s.S = S; s.cfg = cfg; H = cfg["hidden_size"]
        s.embeddings = nn.Module(); s.embeddings.position_embeddings = nn.Embedding(cfg["max_position_embeddings"], H)
        s.embeddings.LayerNorm = LN(H, cfg.get("layer_norm_eps", 1e-5), c=c)
        s.encoder = MPEnc(cfg, c); s.off = cfg.get("pad_token_id", 1) + 1
        s.register_buffer("pos_ids", torch.arange(s.off, s.off + S)); s.register_buffer("buckets", s._buckets(S))
    def _buckets(s, S, nb=None, maxd=128):
        nb = (s.cfg.get("relative_attention_num_buckets", 32)) // 2
        out = np.zeros((S, S), dtype=np.int64)
        for i in range(S):
            for j in range(S):
                n = -(j - i); r = nb if n < 0 else 0; n = abs(n); me = nb // 2
                r += n if n < me else min(me + int(math.log(n / me) / math.log(maxd / me) * (nb - me)), nb - 1)
                out[i, j] = r
        return torch.from_numpy(out)
    def forward(s, embeds, mask):
        e = s.embeddings
        x = e.LayerNorm(embeds + e.position_embeddings(s.pos_ids)[None])
        rb = s.encoder.relative_attention_bias(s.buckets).permute(2, 0, 1)[None]
        bias = pad_bias(mask) + rb
        for l in s.encoder.layer:
            x = l(x, bias, s.S)
        return x


# ---------------------------------------------------------------- nomic_bert (RoPE, fused Wqkv, SwiGLU, post-norm)
class NAtt(nn.Module):
    def __init__(s, H):
        super().__init__(); s.Wqkv = Lin(H, 3 * H, bias=False); s.out_proj = Lin(H, H, bias=False)
class NMlp(nn.Module):
    def __init__(s, H, I):
        super().__init__(); s.fc11, s.fc12 = Lin(H, I, bias=False), Lin(H, I, bias=False); s.fc2 = Lin(I, H, bias=False)
class NLayer(nn.Module):
    def __init__(s, H, I, eps, nh, c):
        super().__init__(); s.attn = NAtt(H); s.mlp = NMlp(H, I); s.norm1 = LN(H, eps, c=c); s.norm2 = LN(H, eps, c=c); s.nh = nh
    def forward(s, x, bias, cos, sin, S):
        q, k, v = (heads(t, s.nh, S) for t in s.attn.Wqkv(x).chunk(3, -1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        h = s.norm1(s.attn.out_proj(merge(sdpa(q, k, v, bias, 1.0 / math.sqrt(q.shape[-1])), S)) + x)
        m = s.mlp
        return s.norm2(m.fc2(m.fc11(h) * F.silu(m.fc12(h))) + h)
class Nomic(nn.Module):
    def __init__(s, cfg, S, c=1.0):
        super().__init__(); s.S = S
        H = cfg.get("n_embd", cfg.get("hidden_size")); nh = cfg.get("n_head", cfg.get("num_attention_heads"))
        I = cfg.get("n_inner", cfg.get("intermediate_size")); eps = cfg.get("layer_norm_epsilon", cfg.get("layer_norm_eps", 1e-12))
        s.embeddings = nn.Module(); s.embeddings.token_type_embeddings = nn.Embedding(max(cfg.get("type_vocab_size", 2), 1), H)
        s.emb_ln = LN(H, eps, c=c); s.encoder = nn.Module()
        s.encoder.layers = nn.ModuleList([NLayer(H, I, eps, nh, c) for _ in range(cfg.get("n_layer", cfg.get("num_hidden_layers")))])
        hd = H // nh; rd = int(hd * cfg.get("rotary_emb_fraction", 1.0)); base = cfg.get("rotary_emb_base", 1000)
        cos, sin = rope_tables(S, rd, 1.0 / (base ** (torch.arange(0, rd, 2).float() / rd)))
        s.register_buffer("cos", cos); s.register_buffer("sin", sin)
    def forward(s, embeds, mask):
        x = s.emb_ln(embeds + s.embeddings.token_type_embeddings.weight[0][None, None])
        bias = pad_bias(mask)
        for l in s.encoder.layers:
            x = l(x, bias, s.cos, s.sin, s.S)
        return x


# ---------------------------------------------------------------- gte-v1.5 ("new": RoPE (NTK), GeGLU, post-norm, CLS)
class GAtt(nn.Module):
    def __init__(s, H):
        super().__init__(); s.qkv_proj = Lin(H, 3 * H); s.o_proj = Lin(H, H)
class GMlp(nn.Module):
    def __init__(s, H, I):
        super().__init__(); s.up_gate_proj = Lin(H, 2 * I, bias=False); s.down_proj = Lin(I, H)
class GLayer(nn.Module):
    def __init__(s, H, I, eps, nh, c):
        super().__init__(); s.attention = GAtt(H); s.mlp = GMlp(H, I); s.attn_ln = LN(H, eps, c=c); s.mlp_ln = LN(H, eps, c=c); s.nh = nh
    def forward(s, x, bias, cos, sin, S):
        q, k, v = (heads(t, s.nh, S) for t in s.attention.qkv_proj(x).chunk(3, -1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        h = s.attn_ln(s.attention.o_proj(merge(sdpa(q, k, v, bias, 1.0 / math.sqrt(q.shape[-1])), S)) + x)
        u, g = s.mlp.up_gate_proj(h).chunk(2, -1)
        return s.mlp_ln(s.mlp.down_proj(u * F.gelu(g)) + h)
class GTE(nn.Module):
    def __init__(s, cfg, S, c=1.0):
        super().__init__(); s.S = S; H, nh = cfg["hidden_size"], cfg["num_attention_heads"]; eps = cfg.get("layer_norm_eps", 1e-12)
        s.embeddings = nn.Module(); tv = cfg.get("type_vocab_size", 0)
        s.embeddings.token_type_embeddings = nn.Embedding(tv, H) if tv > 0 else None
        s.embeddings.LayerNorm = LN(H, eps, c=c); s.encoder = nn.Module()
        s.encoder.layer = nn.ModuleList([GLayer(H, cfg["intermediate_size"], eps, nh, c) for _ in range(cfg["num_hidden_layers"])])
        hd = H // nh; base = cfg.get("rope_theta", 10000.0); rs = cfg.get("rope_scaling") or {}
        if rs.get("type") == "ntk" and rs.get("factor"):
            f = rs["factor"]; corr = f ** (2.0 / hd)
            freqs = torch.tensor([(base * f) ** (2.0 * i / hd) * corr for i in range(hd // 2)])
            inv = 1.0 / freqs
        else:
            inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
        cos, sin = rope_tables(S, hd, inv); s.register_buffer("cos", cos); s.register_buffer("sin", sin)
    def forward(s, embeds, mask):
        e = s.embeddings; x = embeds
        if e.token_type_embeddings is not None:
            x = x + e.token_type_embeddings.weight[0][None, None]
        x = e.LayerNorm(x); bias = pad_bias(mask)
        for l in s.encoder.layer:
            x = l(x, bias, s.cos, s.sin, s.S)
        return x


# ---------------------------------------------------------------- jina-bert-v2 (ALiBi, GLU, post-norm)
def alibi_slopes(n):
    def p2(m):
        st = 2 ** (-(2 ** -(math.log2(m) - 3))); return [st * st ** i for i in range(m)]
    if math.log2(n).is_integer():
        return p2(n)
    cl = 2 ** math.floor(math.log2(n)); out = p2(cl); ex = alibi_slopes(2 * cl); i = 0
    while len(out) < n:
        out.append(ex[i]); i += 2
    return out
class JGlu(nn.Module):
    def __init__(s, H, I, eps, c):
        super().__init__(); s.gated_layers = Lin(H, 2 * I, bias=False); s.wo = Lin(I, H); s.layernorm = LN(H, eps, c=c)
class JLayer(nn.Module):
    def __init__(s, H, I, eps, nh, c):
        super().__init__(); s.attention = BertAtt(H, eps, c); s.mlp = JGlu(H, I, eps, c); s.nh = nh
    def forward(s, x, bias, S):
        a = getattr(s.attention, "self")
        q, k, v = heads(a.query(x), s.nh, S), heads(a.key(x), s.nh, S), heads(a.value(x), s.nh, S)
        h = s.attention.output.LayerNorm(s.attention.output.dense(merge(sdpa(q, k, v, bias, 1.0 / math.sqrt(q.shape[-1])), S)) + x)
        g, u = s.mlp.gated_layers(h).chunk(2, -1)
        return s.mlp.layernorm(s.mlp.wo(F.gelu(g) * u) + h)
class Jina(nn.Module):
    def __init__(s, cfg, S, c=1.0):
        super().__init__(); s.S = S; H, nh = cfg["hidden_size"], cfg["num_attention_heads"]; eps = cfg.get("layer_norm_eps", 1e-12)
        s.embeddings = nn.Module(); s.embeddings.token_type_embeddings = nn.Embedding(max(cfg.get("type_vocab_size", 2), 1), H)
        s.embeddings.LayerNorm = LN(H, eps, c=c); s.encoder = nn.Module()
        s.encoder.layer = nn.ModuleList([JLayer(H, cfg["intermediate_size"], eps, nh, c) for _ in range(cfg["num_hidden_layers"])])
        r = torch.arange(S).float(); dist = (r[:, None] - r[None]).abs()
        sl = torch.tensor(alibi_slopes(nh)).view(nh, 1, 1)
        s.register_buffer("alibi", (-sl * dist[None])[None])
    def forward(s, embeds, mask):
        e = s.embeddings
        x = e.LayerNorm(embeds + e.token_type_embeddings.weight[0][None, None])
        bias = pad_bias(mask) + s.alibi
        for l in s.encoder.layer:
            x = l(x, bias, s.S)
        return x


# ---------------------------------------------------------------- ModernBERT (pre-norm, RoPE, local/global attention, GeGLU)
class MBAtt(nn.Module):
    def __init__(s, H):
        super().__init__(); s.Wqkv = Lin(H, 3 * H, bias=False); s.Wo = Lin(H, H, bias=False)
class MBMlp(nn.Module):
    def __init__(s, H, I):
        super().__init__(); s.Wi = Lin(H, 2 * I, bias=False); s.Wo = Lin(I, H, bias=False)
class MBLayer(nn.Module):
    def __init__(s, cfg, i, c):
        super().__init__(); H, eps = cfg["hidden_size"], cfg.get("norm_eps", 1e-5)
        s.attn_norm = LN(H, eps, bias=False, c=c) if i > 0 else None
        s.attn = MBAtt(H); s.mlp_norm = LN(H, eps, bias=False, c=c); s.mlp = MBMlp(H, cfg["intermediate_size"]); s.nh = cfg["num_attention_heads"]
    def forward(s, x, bias, cos, sin, S):
        n = s.attn_norm(x) if s.attn_norm is not None else x
        q, k, v = (heads(t, s.nh, S) for t in s.attn.Wqkv(n).chunk(3, -1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        x = x + s.attn.Wo(merge(sdpa(q, k, v, bias, 1.0 / math.sqrt(q.shape[-1])), S))
        a, g = s.mlp.Wi(s.mlp_norm(x)).chunk(2, -1)
        return x + s.mlp.Wo(F.gelu(a) * g)
class ModernBert(nn.Module):
    def __init__(s, cfg, S, c=1.0):
        super().__init__(); s.S = S; H, nh = cfg["hidden_size"], cfg["num_attention_heads"]; eps = cfg.get("norm_eps", 1e-5)
        s.embeddings = nn.Module(); s.embeddings.norm = LN(H, eps, bias=False, c=c)
        s.layers = nn.ModuleList([MBLayer(cfg, i, c) for i in range(cfg["num_hidden_layers"])])
        s.final_norm = LN(H, eps, bias=False, c=c)
        hd = H // nh; every = cfg.get("global_attn_every_n_layers", 3); win = cfg.get("local_attention", 128) // 2
        r = torch.arange(S); local = ((r[:, None] - r[None]).abs() <= win).float()
        s.register_buffer("local_bias", ((1 - local) * NEG)[None, None]); s.glob = [i % every == 0 for i in range(len(s.layers))]
        for i in range(len(s.layers)):
            th = cfg.get("global_rope_theta", 160000.0) if s.glob[i] else cfg.get("local_rope_theta", 10000.0)
            cos, sin = rope_tables(S, hd, 1.0 / (th ** (torch.arange(0, hd, 2).float() / hd)))
            s.register_buffer(f"cos{i}", cos); s.register_buffer(f"sin{i}", sin)
    def forward(s, embeds, mask):
        x = s.embeddings.norm(embeds); pb = pad_bias(mask)
        for i, l in enumerate(s.layers):
            x = l(x, pb if s.glob[i] else pb + s.local_bias, getattr(s, f"cos{i}"), getattr(s, f"sin{i}"), s.S)
        return s.final_norm(x)


# ---------------------------------------------------------------- Qwen2 decoder (gte-Qwen2): causal, GQA, RoPE, RMSNorm, SwiGLU
class QAtt(nn.Module):
    def __init__(s, H, nh, nkv, hd):
        super().__init__(); s.q_proj, s.k_proj, s.v_proj = Lin(H, nh * hd), Lin(H, nkv * hd), Lin(H, nkv * hd); s.o_proj = Lin(nh * hd, H, bias=False)
class QMlp(nn.Module):
    def __init__(s, H, I):
        super().__init__(); s.gate_proj, s.up_proj = Lin(H, I, bias=False), Lin(H, I, bias=False); s.down_proj = Lin(I, H, bias=False)
class QLayer(nn.Module):
    def __init__(s, cfg, c):
        super().__init__(); H, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]; hd = H // nh; eps = cfg["rms_norm_eps"]
        s.self_attn = QAtt(H, nh, nkv, hd); s.mlp = QMlp(H, cfg["intermediate_size"])
        s.input_layernorm = RMS(H, eps, c); s.post_attention_layernorm = RMS(H, eps, c); s.nh, s.nkv, s.hd = nh, nkv, hd
    def forward(s, x, bias, cos, sin, S):
        a = s.self_attn; h = s.input_layernorm(x)
        q, k, v = heads(a.q_proj(h), s.nh, S), heads(a.k_proj(h), s.nkv, S), heads(a.v_proj(h), s.nkv, S)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        g = s.nh // s.nkv
        if g > 1:
            k = torch.cat([k[:, i:i + 1].expand(1, g, S, s.hd) for i in range(s.nkv)], 1)
            v = torch.cat([v[:, i:i + 1].expand(1, g, S, s.hd) for i in range(s.nkv)], 1)
        x = x + a.o_proj(merge(sdpa(q, k, v, bias, 1.0 / math.sqrt(s.hd)), S))
        h = s.post_attention_layernorm(x); m = s.mlp
        return x + m.down_proj(F.silu(m.gate_proj(h)) * m.up_proj(h))
class Qwen2(nn.Module):
    def __init__(s, cfg, S, c=1 / 64.):
        super().__init__(); s.S = S
        s.layers = nn.ModuleList([QLayer(cfg, c) for _ in range(cfg["num_hidden_layers"])]); s.norm = RMS(cfg["hidden_size"], cfg["rms_norm_eps"], c)
        hd = cfg["hidden_size"] // cfg["num_attention_heads"]
        cos, sin = rope_tables(S, hd, 1.0 / (cfg.get("rope_theta", 1e6) ** (torch.arange(0, hd, 2).float() / hd)))
        s.register_buffer("cos", cos); s.register_buffer("sin", sin)
        r = torch.arange(S); s.register_buffer("causal", ((r[None, :] > r[:, None]).float() * NEG)[None, None])
    def forward(s, embeds, mask):
        bias = pad_bias(mask) + s.causal; x = embeds
        for l in s.layers:
            x = l(x, bias, s.cos, s.sin, s.S)
        return s.norm(x)


# ---------------------------------------------------------------- nomic-embed-text-v2-moe (every 2nd layer: top-2 of 8 experts)
class V2Att(nn.Module):
    def __init__(s, H):
        super().__init__(); s.Wqkv = Lin(H, 3 * H); s.out_proj = Lin(H, H)
class V2Router(nn.Module):
    def __init__(s, H, E):
        super().__init__(); s.layer = Lin(H, E, bias=False)
class V2ExpertW(nn.Module):
    def __init__(s, E, F_, H):
        super().__init__(); s.w1 = nn.Parameter(torch.zeros(E * F_, H)); s.w2 = nn.Parameter(torch.zeros(E * F_, H))
class V2Experts(nn.Module):
    def __init__(s, E, F_, H):
        super().__init__(); s.mlp = V2ExpertW(E, F_, H); s.bias = nn.Parameter(torch.zeros(H))
class V2Mlp(nn.Module):
    def __init__(s, cfg, moe):
        super().__init__(); H, F_ = cfg["n_embd"], cfg["n_inner"]; s.moe = moe; s.E = cfg.get("num_experts", 0); s.F = F_; s.k = cfg.get("moe_top_k", 2)
        if moe:
            s.router = V2Router(H, s.E); s.experts = V2Experts(s.E, F_, H)
        else:
            s.fc1 = Lin(H, F_); s.fc2 = Lin(F_, H)
    def forward(s, x):
        if not s.moe:
            return s.fc2(F.gelu(s.fc1(x)))
        probs = torch.softmax(s.router.layer(x), -1)
        # top-2 weights without sort/topk (Krill keeps every probability >= the k-th largest): two max-reductions
        m1 = probs.max(-1, keepdim=True)[0]
        rest = probs * (probs < m1).to(probs.dtype)
        thresh = rest.max(-1, keepdim=True)[0] if s.k == 2 else m1
        w = probs * (probs >= thresh).to(probs.dtype)
        w1 = s.experts.mlp.w1.view(s.E, s.F, -1); w2 = s.experts.mlp.w2.view(s.E, s.F, -1)
        out = 0
        for e in range(s.E):
            out = out + w[..., e:e + 1] * F.linear(F.gelu(F.linear(x, w1[e])), w2[e].t())
        return out + s.experts.bias
class V2Layer(nn.Module):
    def __init__(s, cfg, moe, c):
        super().__init__(); H = cfg["n_embd"]; eps = cfg.get("layer_norm_epsilon", 1e-5)
        s.attn = V2Att(H); s.mlp = V2Mlp(cfg, moe); s.norm1 = LN(H, eps, c=c); s.norm2 = LN(H, eps, c=c); s.nh = cfg["n_head"]
    def forward(s, x, bias, cos, sin, S):
        q, k, v = (heads(t, s.nh, S) for t in s.attn.Wqkv(x).chunk(3, -1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        h = s.norm1(s.attn.out_proj(merge(sdpa(q, k, v, bias, 1.0 / math.sqrt(q.shape[-1])), S)) + x)
        return s.norm2(s.mlp(h) + h)
class NomicMoE(nn.Module):
    def __init__(s, cfg, S, c=1.0):
        super().__init__(); s.S = S; H = cfg["n_embd"]; nh = cfg["n_head"]; eps = cfg.get("layer_norm_epsilon", 1e-5)
        s.embeddings = nn.Module(); s.embeddings.token_type_embeddings = nn.Embedding(1, H)
        s.emb_ln = LN(H, eps, c=c); s.encoder = nn.Module(); ev = cfg.get("moe_every_n_layers", 2)
        s.encoder.layers = nn.ModuleList([V2Layer(cfg, ev > 0 and i % ev == 1, c) for i in range(cfg["n_layer"])])
        hd = H // nh; rd = int(hd * cfg.get("rotary_emb_fraction", 1.0)); base = cfg.get("rotary_emb_base", 10000)
        cos, sin = rope_tables(S, rd, 1.0 / (base ** (torch.arange(0, rd, 2).float() / rd)))
        s.register_buffer("cos", cos); s.register_buffer("sin", sin)
    def forward(s, embeds, mask):
        x = s.emb_ln(embeds + s.embeddings.token_type_embeddings.weight[0][None, None]); bias = pad_bias(mask)
        for l in s.encoder.layers:
            x = l(x, bias, s.cos, s.sin, s.S)
        return x


# ---------------------------------------------------------------- pooling wrapper + loading
class Pooled(nn.Module):
    """core(embeds, mask) -> hidden [1,S,H]; pooling mean|cls|last_token|classifier; embedders output the L2-normalised vector."""
    def __init__(s, core, pooling):
        super().__init__(); s.core, s.pooling = core, pooling
    def forward(s, embeds, mask):
        h = s.core(embeds, mask)
        if s.pooling == "classifier":
            return s.core.head(h)
        if s.pooling == "cls":
            p = h[:, 0]
        elif s.pooling == "mean":
            m = mask[:, :, None]; p = (h * m).sum(1) / m.sum(1)
        else:   # last_token: one-hot of the final real position
            nxt = torch.cat([mask[:, 1:], torch.zeros_like(mask[:, :1])], 1)
            p = (h * (mask - nxt)[:, :, None]).sum(1)
        return p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def strip_prefix(sd):
    for pre in ("bert.", "roberta.", "model.", "xlm_roberta."):
        if any(k.startswith(pre) for k in sd):
            sd = {(k[len(pre):] if k.startswith(pre) else k): v for k, v in sd.items()}
    return sd


def read_weights(model_dir):
    import glob
    from safetensors import safe_open
    sd = {}
    for f in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                sd[k] = fh.get_tensor(k)
    return strip_prefix(sd)


def build(kind, cfg, S, c=1.0, cross_encoder=False):
    if kind == "bert": return Bert(cfg, S, c, cross_encoder)
    if kind == "mpnet": return MPNet(cfg, S, c)
    if kind == "nomic_bert": return Nomic(cfg, S, c)
    if kind == "nomic_moe": return NomicMoE(cfg, S, c)
    if kind == "new": return GTE(cfg, S, c)
    if kind == "jina": return Jina(cfg, S, c)
    if kind == "modernbert": return ModernBert(cfg, S, c)
    if kind == "qwen2": return Qwen2(cfg, S)
    raise SystemExit(f"no Neural Engine port for model_type '{kind}'")


def word_table(kind, sd):
    for k in ("embeddings.word_embeddings.weight", "embeddings.tok_embeddings.weight", "embed_tokens.weight"):
        if k in sd:
            return sd[k]
    raise SystemExit("no word-embedding table in the checkpoint")


def load_core(core, sd):
    """Strict load of everything except the host-side word table and buffers/pooler."""
    drop = ("embeddings.word_embeddings.weight", "embeddings.tok_embeddings.weight", "embed_tokens.weight", "embeddings.position_ids")
    own = core.state_dict()
    use = {k: v.float() for k, v in sd.items() if k in own and k not in drop}
    missing = [k for k in own if k not in use and not any(b in k for b in ("pos_ids", "buckets", "cos", "sin", "alibi", "local_bias", "causal"))]
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:3]}")
    core.load_state_dict(use, strict=False)
