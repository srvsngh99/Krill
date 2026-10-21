"""Parity gate for build_ane_package.py: serve the new package with `krill serve` and compare it with the GPU build of
the same model through the same server (request `device` field), so the Swift runtime path is what is tested.

Embedders: cosine of the ANE vector against the GPU vector for 200 queries (one per request) and 200 passages (batches of 8);
the gate is the minimum >= --min-cosine. Rerankers: 40 queries x 25 documents through /v1/rerank; the gate is the Spearman
rank correlation of the pooled logits >= --min-spearman (the maximum absolute logit difference is reported)."""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.request
import numpy as np


def _post(port, path, body, timeout=600):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read()), resp.headers.get("x-krill-embed-device")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"gate: {path} {body.get('device')} -> HTTP {e.code}: {e.read().decode()[:300]}")


def _vecs(port, model, texts, bs, device):
    out, devs = [], []
    for i in range(0, len(texts), bs):
        d, dev = _post(port, "/v1/embeddings", dict(model=model, input=texts[i:i + bs], device=device))
        out += [x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])]
        devs.append(dev)
    return np.array(out, dtype=np.float32), devs


def _cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def _rank(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def _hf_vectors(src, spec, texts):
    """HF fp32 reference vectors (--reference hf) for the texts that fit the ANE buckets (<= 512 tokens)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, m = AutoTokenizer.from_pretrained(src), AutoModel.from_pretrained(src).eval()
    keep, out = [], []
    for t in texts:
        enc = tok(t, truncation=True, max_length=spec["ctx"], return_tensors="pt")
        if enc["input_ids"].shape[1] > 512:
            continue
        with torch.no_grad():
            h = m(**enc).last_hidden_state[0]
        p = h[0] if spec["pooling"] == "cls" else h.mean(0) if spec["pooling"] == "mean" else h[-1]
        out.append((p / p.norm()).numpy()); keep.append(t)
    return keep, np.array(out, dtype=np.float32)


def stats(c):
    return dict(n=len(c), min=float(c.min()), mean=float(c.mean()), p01=float(np.percentile(c, 1)))


def run(a, store, name, pkgdir, spec, family):
    if not a.corpus or not os.path.exists(a.corpus):
        raise SystemExit("the parity gate needs --corpus FILE (JSON {\"queries\": [...], \"passages\": [...]}) or ANE_CORPUS")
    corpus = json.load(open(a.corpus))
    Q, P = corpus["queries"][:200], corpus["passages"][:200]
    krill = os.path.abspath(a.krill)
    if not os.path.exists(krill):
        raise SystemExit(f"krill binary not found: {krill} (build it: make release, or pass --krill)")
    gate = tempfile.mkdtemp(prefix="ane-gate-", dir=os.environ.get("ANE_GATE_TMP") or None)
    for d in ("blobs", "manifests"):
        os.makedirs(os.path.join(gate, d))
    # a real directory of linked files: the loaders do not follow a symlinked model directory
    def _link(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            os.symlink(os.path.abspath(src), dst)
    shutil.copytree(os.path.join(store, "blobs", a.model), os.path.join(gate, "blobs", a.model), copy_function=_link)
    os.symlink(os.path.join(store, "manifests", a.model + ".json"), os.path.join(gate, "manifests", a.model + ".json"))
    os.symlink(os.path.abspath(pkgdir), os.path.join(gate, "blobs", name))
    json.dump({"name": name, "family": family, "params": "", "quant": f"ane-{a.variant}", "source": "gate", "context": 512,
               "chatTemplate": "bert", "files": [], "sizeBytes": 0, "pulledAt": "2026-01-01T00:00:00Z"},
              open(os.path.join(gate, "manifests", name + ".json"), "w"))
    env = dict(os.environ, KRILL_MODELS_DIR=gate, KRILL_API_KEY="", KRILL_EMBED_DEVICE="ane", KRILL_KEEP_ALIVE="30m")
    log = open(os.path.join(gate, "serve.log"), "w")
    p = subprocess.Popen([krill, "serve", "--host", "127.0.0.1", "--port", str(a.port)], env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while time.time() - t0 < 90:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/healthz", timeout=1); break
            except Exception:
                time.sleep(0.2)
        else:
            raise SystemExit("krill serve did not start; see " + log.name)
        res = {"model": a.model, "variant": a.variant}
        if spec["cross"]:
            # Graded-relevance pairs: the query is the first 12 words of a passage; its 25 documents are 15 windows of the
            # same passage (different offsets and lengths, so scores spread) and 10 windows of other passages.
            rng = np.random.RandomState(0); cases = []
            for qi in range(40):
                w = P[qi].split(); q = " ".join(w[:12]); docs = []
                for k in range(15):
                    o = (k * 5) % max(len(w) - 40, 1); docs.append(" ".join(w[o:o + 25 + 3 * k]))
                for k in range(10):
                    o2 = P[(qi + 3 + k) % len(P)].split(); docs.append(" ".join(o2[:40]))
                cases.append((q, docs))
            logits = {}
            for dev in ("ane", "gpu"):
                allv = []
                for q, docs in cases:
                    d, hdr = _post(a.port, "/v1/rerank", dict(model=a.model, query=q, documents=docs, device=dev))
                    if dev == "ane" and hdr != "ane":
                        print(f"gate: rerank was served on '{hdr}', not the ANE package"); return False
                    allv += [x["logit"] for x in sorted(d["results"], key=lambda x: x["index"])]
                logits[dev] = np.array(allv)
            g, n_ = logits["gpu"], logits["ane"]
            live = g > g.min() + 0.3          # pairs the model actually separates; its floor is a plateau of ties
            rho_all = float(np.corrcoef(_rank(n_), _rank(g))[0, 1])
            rho = float(np.corrcoef(_rank(n_[live]), _rank(g[live]))[0, 1])
            mad = float(np.abs(n_ - g).max())
            res.update(spearman=rho, spearman_all_pairs=rho_all, max_abs_diff=mad, n=len(g), n_scored=int(live.sum()))
            ok = rho >= a.min_spearman
            print(f"parity {name}: Spearman {rho:.5f} on {int(live.sum())} scored pairs ({rho_all:.5f} on all {len(g)} incl. the floor plateau), "
                  f"max abs logit diff {mad:.4f} (need Spearman >= {a.min_spearman})")
        else:
            ok = True
            for label, texts, bs in (("queries", Q, 1), ("passages", P, 8)):
                if a.reference == "hf":
                    texts, vg = _hf_vectors(os.path.join(store, "blobs", a.model), spec, texts)
                    va, da = _vecs(a.port, a.model, texts, bs, "ane")
                else:
                    va, da = _vecs(a.port, a.model, texts, bs, "ane")
                    vg, _ = _vecs(a.port, a.model, texts, bs, "gpu")
                if not set(da) <= {"ane", "mixed"}:
                    print(f"gate: {label} were served on {sorted(set(da))}, not the ANE package"); return False
                s = stats(_cos(va, vg)); res[label] = s
                ok &= s["min"] >= a.min_cosine
                print(f"parity {name} {label}: cosine min {s['min']:.5f} mean {s['mean']:.5f} p01 {s['p01']:.5f} (n={s['n']}, ref={a.reference}, need min >= {a.min_cosine})")
        res["pass"] = bool(ok)
        json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(pkgdir)), f"parity-{name}.json"), "w"), indent=2)
        return bool(ok)
    finally:
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
