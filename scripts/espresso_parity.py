#!/usr/bin/env python3
import json

from amber_parity import matched_choices, run_parity


def espresso_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def espresso_cases():
    for state, question, answer, _ in matched_choices("ESPRESSO"):
        criteria = question["criteria"]
        yield {
            "state": "" if state is None else espresso_text(state),
            "instructions": espresso_text(question.get("instructions", "")),
            "options": ["" if value is None else espresso_text(value) for value in criteria.values()],
            "choiceKeys": list(criteria),
            "choice": list(criteria).index(answer["choice"]),
            "probabilities": [answer["probabilities"][key] for key in criteria],
        }


if __name__ == "__main__":
    raise SystemExit(run_parity("ESPRESSO", "EspressoDecisionModelTests/testReferenceParity", espresso_cases()))
