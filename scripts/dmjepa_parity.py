#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import tempfile

from amber_parity import read_rows


def render(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2) if value else ""
    if value is None:
        return "None"
    return str(value)


def main():
    inputs = read_rows("KRILL_DMJEPA_INPUTS")
    results = read_rows("KRILL_DMJEPA_RESULTS")
    by_id = {row["run_id"]: row for row in results}
    ids = {row["_evaluation"]["run_id"] for row in inputs}
    if not (len(inputs) == len(results) == len(by_id) == len(ids) == 200 and ids == by_id.keys()):
        raise ValueError("expected 200 matching input and result IDs")

    model = Path(os.environ["KRILL_DMJEPA_MODEL_PATH"])
    tokenizer = Path(os.environ.get("KRILL_DMJEPA_TOKENIZER_PATH", model))
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        for source, names in (
            (model, ("config.json", "model.safetensors")),
            (tokenizer, ("tokenizer.json", "tokenizer_config.json")),
        ):
            for name in names:
                (bundle / name).symlink_to((source / name).resolve(strict=True))

        path = bundle / "rows.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in inputs:
                reference = by_id[row["_evaluation"]["run_id"]]
                payload = {key: row[key] for key in ("state", "questions")}
                if reference["status"] != "ok" or reference["payload"] != payload or len(row["questions"]) != 1:
                    raise ValueError("expected a successful one-question result for each input")
                name, question = next(iter(row["questions"].items()))
                answer = reference["response"]["answers"][name]
                criteria = question["criteria"]
                if question["type"] != "choice" or answer["probabilities"].keys() != criteria.keys():
                    raise ValueError("expected probabilities for every choice criterion")
                labels = list(criteria)
                case = {
                    "state": render(row["state"]),
                    "instructions": render(question.get("instructions") or ""),
                    "labels": labels,
                    "options": [str(value) if value else "" for value in criteria.values()],
                    "choice": labels.index(answer["choice"]),
                    "probabilities": [answer["probabilities"][label] for label in labels],
                }
                file.write(json.dumps(case, ensure_ascii=False) + "\n")
        return subprocess.call(
            ["swift", "test", "-c", "release", "--skip-build", "--filter", "DMJEPADecisionModelTests/testReferenceParity"],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "KRILL_DMJEPA_MODEL_PATH": str(bundle), "KRILL_DMJEPA_PARITY_ROWS": str(path)},
        )


if __name__ == "__main__":
    raise SystemExit(main())
