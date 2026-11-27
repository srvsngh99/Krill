#!/usr/bin/env python3
import hashlib
import importlib
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import tempfile


def run_krill(binary, env, *args, stdin=None):
    return subprocess.run(
        [str(binary), *args], env=env, input=stdin, text=True, capture_output=True, check=True,
    ).stdout


def register_base(models, bundle):
    manifests, blobs = models / "manifests", models / "blobs"
    manifests.mkdir(parents=True)
    blobs.mkdir()
    shutil.copytree(bundle, blobs / "base", copy_function=os.link, ignore=shutil.ignore_patterns("__pycache__"))
    config = json.loads((bundle / "config.json").read_text())
    (manifests / "base.json").write_text(json.dumps({
        "name": "base", "family": "decision", "params": "unknown", "quant": config.get("dtype", "fp16"),
        "source": "local", "context": config["state_max_tokens"], "files": [],
        "chatTemplate": "decision", "sizeBytes": 0, "pulledAt": "2000-01-01T00:00:00Z",
    }))


def krill_input(row):
    if "answer" in row:
        return {"text": row["text"], "yes_no": row["question"]}
    result = {"text": row["text"], "options": row["options"]}
    if row.get("question"):
        result["question"] = row["question"]
    return result


def check_accuracy(label, a, b):
    if not math.isclose(a, b, rel_tol=0, abs_tol=0.01 + 1e-12):
        raise ValueError(f"{label} differs by more than one point ({a:.2%} vs {b:.2%})")


def cross_check(load, bundle, binary, env, rows):
    stdin = "".join(json.dumps(krill_input(row)) + "\n" for row in rows)
    output = run_krill(binary, env, "decide", "--model", str(bundle), "--json-in", "-", stdin=stdin)
    scores = [json.loads(line) for line in output.splitlines()]
    model = load(bundle, device="cpu")
    worst = 0.0
    reference_hits = krill_hits = 0
    for row, actual in zip(rows, scores, strict=True):
        if "answer" in row:
            reference = model.yes_no(row["text"], row["question"])
            expected, observed = [reference.p_yes], [actual["p_yes"]]
            target, python_prediction, krill_prediction = row["answer"], reference.answer, actual["answer"]
        else:
            reference = model.choose(row["text"], row["options"], row.get("question"))
            expected = [*reference.scores, reference.p_none]
            observed = [*actual["scores"], actual["p_none"]]
            target, python_prediction, krill_prediction = row["label"], reference.label, actual["label"]
        reference_hits += python_prediction == target
        krill_hits += krill_prediction == target
        for a, b in zip(expected, observed, strict=True):
            if not math.isclose(a, b, rel_tol=0, abs_tol=1e-3):
                raise ValueError(f"{bundle.name}: score difference exceeds 1e-3 ({a} vs {b})")
            worst = max(worst, abs(a - b))
    python_accuracy, krill_accuracy = reference_hits / len(rows), krill_hits / len(rows)
    check_accuracy(f"{bundle.name} cross-runtime accuracy", python_accuracy, krill_accuracy)
    return worst, python_accuracy, krill_accuracy


def main():
    bundle = Path(os.environ["KRILL_TUNE_BASE"]).resolve(strict=True)
    data = Path(os.environ["KRILL_TUNE_DATA"]).resolve(strict=True)
    load = importlib.import_module(os.environ["KRILL_TUNE_REFERENCE"]).load
    default_binary = Path(__file__).resolve().parents[1] / ".build/release/krill"
    binary = Path(os.environ.get("KRILL_BIN", default_binary)).resolve(strict=True)
    with data.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    if len(rows) < 20:
        raise ValueError("held-out parity requires at least 20 examples")
    order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(f"0:0:{i}".encode("ascii")).digest())
    held_out = [rows[i] for i in order[:int(len(rows) * 0.2)]]

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        models = root / "models"
        register_base(models, bundle)
        env = {**os.environ, "KRILL_MODELS_DIR": str(models)}
        model = load(bundle, device="cpu")
        python_report = model.tune(rows, force=True)
        python_bundle = model.save(root / "python-tuned")
        del model
        krill_bundle = models / "blobs" / "krill-tuned"
        output = run_krill(binary, env, "tune", "--model", "base", "--name", krill_bundle.name,
                           "--data", str(data), "--force")
        krill_report, _ = json.JSONDecoder().raw_decode(output.lstrip())
        for report in (python_report, krill_report):
            if not report["accepted"] or report["holdout_examples"] != len(held_out):
                raise ValueError(f"unexpected split or rejected tune: {report}")

        a, b = python_report["after"]["accuracy"], krill_report["after"]["accuracy"]
        check_accuracy("held-out report accuracy", a, b)
        print(f"held-out report accuracy: Python {a:.2%}, Krill {b:.2%}")
        python_gap, python_accuracy, _ = cross_check(load, python_bundle, binary, env, held_out)
        krill_gap, krill_in_python, krill_accuracy = cross_check(load, krill_bundle, binary, env, held_out)
        check_accuracy("exported bundle accuracy in Python", python_accuracy, krill_in_python)
        check_accuracy("exported bundle accuracy", python_accuracy, krill_accuracy)
        print(f"exported bundle accuracy: Python {python_accuracy:.2%}, Krill {krill_accuracy:.2%}")
        print(f"largest cross-runtime score difference: {max(python_gap, krill_gap):.6g}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.stderr or error.stdout or str(error)) from error
