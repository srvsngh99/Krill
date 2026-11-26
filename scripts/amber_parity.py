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


def main():
    inputs = read_rows("KRILL_AMBER_INPUTS")
    results = read_rows("KRILL_AMBER_RESULTS")
    by_id = {row["run_id"]: row for row in results}
    ids = {row["_evaluation"]["run_id"] for row in inputs}
    if not (len(inputs) == len(results) == len(by_id) == len(ids) == 200 and ids == by_id.keys()):
        raise ValueError("expected 200 matching input and result IDs")
    os.environ["KRILL_AMBER_MODEL_PATH"] = str(Path(os.environ["KRILL_AMBER_MODEL_PATH"]).resolve(strict=True))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.jsonl"
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
                case = {
                    "state": render_state(row["state"]),
                    "instructions": render_state(question.get("instructions")),
                    "options": [render_state(value) or key for key, value in criteria.items()],
                    "choice": list(criteria).index(answer["choice"]),
                    "probabilities": [answer["probabilities"][key] for key in criteria],
                }
                file.write(json.dumps(case, ensure_ascii=False) + "\n")
        return subprocess.call(
            ["swift", "test", "-c", "release", "--skip-build", "--filter", "AmberDecisionModelTests/testReferenceParity"],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "KRILL_AMBER_PARITY_ROWS": str(path)},
        )


if __name__ == "__main__":
    raise SystemExit(main())
