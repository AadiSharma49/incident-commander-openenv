#!/usr/bin/env python3
from __future__ import annotations

import os
from statistics import mean

from openai import OpenAI

from app.env.env import IncidentCommanderEnvironment
from app.models import Action
from app.tasks.tasks import TASKS

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")


def build_openai_client() -> OpenAI | None:
    if not MODEL_NAME or not HF_TOKEN:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def maybe_request_plan(client: OpenAI | None, task_id: str) -> str:
    if client is None:
        return "baseline-plan"
    task = TASKS[task_id]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return a short deterministic incident response plan label."},
            {"role": "user", "content": f"Task: {task.title}. Return a concise plan label only."},
        ],
        temperature=0.0,
        max_tokens=16,
    )
    return (response.choices[0].message.content or "").strip() or "baseline-plan"


def run_task(env: IncidentCommanderEnvironment, task_id: str) -> dict[str, object]:
    task = TASKS[task_id]
    observation = env.reset(task_id)
    done = False
    last_reward = 0.0
    for action_type, target in task.baseline_plan:
        observation, reward, done, _ = env.step(Action(action_type=action_type, target=target))
        last_reward = reward.score
        if done:
            break
    state = env.state()
    return {
        "task_id": task_id,
        "difficulty": task.difficulty,
        "done": done,
        "score": state["score"],
        "reward_total": state["reward_total"],
        "last_reward": last_reward,
        "turns_used": state["turn"],
        "final_status": observation.system_status,
    }


def main() -> None:
    client = build_openai_client()
    env = IncidentCommanderEnvironment()
    scores: list[float] = []

    for task_id in TASKS:
        plan_label = maybe_request_plan(client, task_id)
        result = run_task(env, task_id)
        scores.append(result["score"])
        print(
            f"{task_id}: score={result['score']:.2f} reward_total={result['reward_total']:.2f} "
            f"done={result['done']} plan={plan_label}"
        )

    print(f"average_score: {mean(scores):.2f}")


if __name__ == "__main__":
    main()
