#!/usr/bin/env python3
import json

from amber_parity import matched_choices, render_state, run_parity


def json_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def glide_cases():
    for state, question, answer, usage in matched_choices("GLIDE"):
        criteria = question["criteria"]
        options = []
        for key, value in criteria.items():
            description = key if value in (None, "") else json_text(value)
            options.append(key if key == description else f"{key}: {description}")
        instructions = question.get("instructions", "")
        yield {
            "state": json_text(state),
            "instructions": render_state(instructions) if isinstance(instructions, dict) else json_text(instructions),
            "options": options,
            "choice": list(criteria).index(answer["choice"]),
            "probabilities": [answer["probabilities"][key] for key in criteria],
            "inputTokens": usage["input_tokens"],
        }


if __name__ == "__main__":
    raise SystemExit(run_parity("GLIDE", "GlideDecisionModelTests/testReferenceParity", glide_cases()))
