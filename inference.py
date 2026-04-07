#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from statistics import mean

from openai import OpenAI

from app.env.env import IncidentCommanderEnvironment
from app.models import Action
from app.tasks.tasks import TASKS

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = (
    os.getenv("MODEL_NAME")
    or os.getenv("OPENAI_MODEL")
)
API_KEY = os.getenv("API_KEY")
HAS_PROXY_ENV = "API_BASE_URL" in os.environ or "API_KEY" in os.environ


def build_openai_client() -> OpenAI | None:
    if HAS_PROXY_ENV and (not API_KEY or "API_BASE_URL" not in os.environ):
        raise RuntimeError("Submission requires both API_BASE_URL and API_KEY environment variables.")
    if API_KEY and "API_BASE_URL" in os.environ:
        return OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
    return None


def resolve_model_name(client: OpenAI | None) -> str | None:
    if client is None:
        return None
    if MODEL_NAME:
        return MODEL_NAME
    try:
        models = client.models.list()
        for model in models.data:
            if getattr(model, "id", None):
                return str(model.id)
    except Exception:
        return None
    return None


def maybe_request_plan(client: OpenAI | None, model_name: str | None, task_id: str) -> str:
    if client is None:
        return "baseline-plan"
    if not model_name:
        raise RuntimeError("Proxy client was initialized, but no model name could be resolved.")
    task = TASKS[task_id]
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Return a short deterministic incident response plan label."},
            {"role": "user", "content": f"Task: {task.title}. Return a concise plan label only."},
        ],
        temperature=0.0,
        max_tokens=16,
    )
    return (response.choices[0].message.content or "").strip() or "baseline-plan"


def emit_block(tag: str, payload: dict[str, object]) -> None:
    print(f"[{tag}] {json.dumps(payload, sort_keys=True)}")


def run_task(env: IncidentCommanderEnvironment, task_id: str, plan_label: str) -> dict[str, object]:
    task = TASKS[task_id]
    observation = env.reset(task_id)
    emit_block(
        "START",
        {
            "task_id": task_id,
            "difficulty": task.difficulty,
            "plan": plan_label,
            "system_status": observation.system_status,
        },
    )

    done = False
    last_reward = 0.0
    steps_taken = 0
    for action_type, target in task.baseline_plan:
        steps_taken += 1
        observation, reward, done, info = env.step(Action(action_type=action_type, target=target))
        last_reward = reward.score
        emit_block(
            "STEP",
            {
                "task_id": task_id,
                "step": steps_taken,
                "action_type": action_type,
                "target": target,
                "reward": reward.score,
                "done": done,
                "score": info["score"],
                "reason": reward.reason,
            },
        )
        if done:
            break
    state = env.state()
    result = {
        "task_id": task_id,
        "difficulty": task.difficulty,
        "done": done,
        "score": state["score"],
        "reward_total": state["reward_total"],
        "last_reward": last_reward,
        "turns_used": state["turn"],
        "final_status": observation.system_status,
    }
    emit_block("END", result)
    return result


def main() -> None:
    client = build_openai_client()
    model_name = resolve_model_name(client)
    if HAS_PROXY_ENV and client is None:
        raise RuntimeError("Proxy environment detected, but OpenAI client could not be initialized.")
    env = IncidentCommanderEnvironment()
    scores: list[float] = []

    for task_id in TASKS:
        plan_label = maybe_request_plan(client, model_name, task_id)
        result = run_task(env, task_id, plan_label)
        scores.append(result["score"])

    emit_block("END", {"average_score": round(mean(scores), 4), "tasks_completed": len(scores)})


if __name__ == "__main__":
    main()
