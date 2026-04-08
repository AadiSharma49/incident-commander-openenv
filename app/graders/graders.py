from __future__ import annotations

from collections.abc import Iterable

from app.tasks.tasks import TASKS

MIN_SCORE = 0.01
MAX_SCORE = 0.99


def _fraction(found: Iterable[str], required: Iterable[str]) -> float:
    required_items = tuple(required)
    if not required_items:
        return 0.99
    found_set = set(found)
    return len(found_set.intersection(required_items)) / len(required_items)


def grade_task(state: dict) -> float:
    task = TASKS[state["task_id"]]
    facts_score = _fraction(state.get("known_facts", []), task.required_facts)
    mitigation_score = _fraction(state.get("fixes_applied", []), task.mitigation_targets)
    diagnosis_score = 0.99 if state.get("diagnosis") == task.diagnosis_target else 0.0
    validation_score = 0.99 if task.validation_target in state.get("validations_passed", []) else 0.0
    communication_score = 0.99 if task.communication_target in state.get("communications_sent", []) else 0.0
    penalties = min(
        0.25,
        state.get("wrong_actions", 0) * 0.1 + state.get("repeated_useless_actions", 0) * 0.05,
    )

    score = (
        0.2 * facts_score
        + 0.3 * diagnosis_score
        + 0.3 * mitigation_score
        + 0.1 * validation_score
        + 0.1 * communication_score
        - penalties
    )
    return round(max(MIN_SCORE, min(MAX_SCORE, score)), 4)
