#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import tempfile


def read_rows(variable):
    with Path(os.environ[variable]).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def render_state(value):
    if value in (None, "", {}, []):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(
            f"{key}: {item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)}"
            for key, item in value.items()
        )
    return json.dumps(value, ensure_ascii=False)


def matched_choices(model):
    inputs = read_rows(f"KRILL_{model}_INPUTS")
    results = read_rows(f"KRILL_{model}_RESULTS")
    by_id = {row["run_id"]: row for row in results}
    ids = {row["_evaluation"]["run_id"] for row in inputs}
    if not (len(inputs) == len(results) == len(by_id) == len(ids) == 200 and ids == by_id.keys()):
        raise ValueError("expected 200 matching input and result IDs")
    for row in inputs:
        reference = by_id[row["_evaluation"]["run_id"]]
        payload = {key: row[key] for key in ("state", "questions")}
        if reference["status"] != "ok" or reference["payload"] != payload or len(row["questions"]) != 1:
            raise ValueError("expected a successful one-question result for each input")
        name, question = next(iter(row["questions"].items()))
        response = reference["response"]
        answer = response["answers"][name]
        criteria = question["criteria"]
        if question["type"] != "choice" or answer["probabilities"].keys() != criteria.keys():
            raise ValueError("expected probabilities for every choice criterion")
        yield row["state"], question, answer, response.get("usage", {})


def run_parity(model, test, cases):
    model_path = f"KRILL_{model}_MODEL_PATH"
    bundle = Path(os.environ[model_path]).resolve(strict=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for case in cases:
                file.write(json.dumps(case, ensure_ascii=False) + "\n")
        return subprocess.call(
            ["swift", "test", "-c", "release", "--skip-build", "--filter", test],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, model_path: str(bundle), f"KRILL_{model}_PARITY_ROWS": str(path)},
        )


def amber_cases():
    for state, question, answer, _ in matched_choices("AMBER"):
        criteria = question["criteria"]
        yield {
            "state": render_state(state),
            "instructions": render_state(question.get("instructions")),
            "options": [render_state(value) or key for key, value in criteria.items()],
            "choice": list(criteria).index(answer["choice"]),
            "probabilities": [answer["probabilities"][key] for key in criteria],
        }


if __name__ == "__main__":
    raise SystemExit(run_parity("AMBER", "AmberDecisionModelTests/testReferenceParity", amber_cases()))
