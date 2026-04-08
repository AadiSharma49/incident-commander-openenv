#!/usr/bin/env python3
from __future__ import annotations

import os

from openai import OpenAI

from app.env.env import IncidentCommanderEnvironment
from app.graders.graders import MIN_SCORE
from app.models import Action
from app.tasks.tasks import TASKS

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK_NAME = "incident-commander-openenv"


def format_bool(value: bool) -> str:
    return "true" if value else "false"


def format_reward(value: float) -> str:
    return f"{value:.2f}"


def format_action(action_type: str, target: str) -> str:
    return f"{action_type}('{target}')"


def sanitize_error(value: str | None) -> str:
    if not value:
        return "null"
    return value.replace("\n", " ").replace("\r", " ").strip() or "null"


def build_openai_client() -> OpenAI:
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def maybe_request_plan_label(client: OpenAI, task_id: str) -> str:
    task = TASKS[task_id]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return a short deterministic plan label only."},
                {"role": "user", "content": f"Task: {task.title}"},
            ],
            temperature=0.0,
            max_tokens=16,
        )
        content = response.choices[0].message.content or ""
        if isinstance(content, str):
            return content.strip() or "baseline-plan"
    except Exception:
        pass
    return "baseline-plan"


def run_episode(env: IncidentCommanderEnvironment, client: OpenAI, task_id: str) -> None:
    task = TASKS[task_id]
    _ = maybe_request_plan_label(client, task_id)

    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={MODEL_NAME}")

    rewards: list[str] = []
    steps_taken = 0
    success = False

    try:
        env.reset(task_id)
        for action_type, target in task.baseline_plan:
            steps_taken += 1
            action_text = format_action(action_type, target)
            error_text = "null"
            reward_value = MIN_SCORE
            done = False

            try:
                _, reward, done, _ = env.step(Action(action_type=action_type, target=target))
                reward_value = reward.score
                success = done
            except Exception as exc:
                done = True
                error_text = sanitize_error(str(exc))

            rewards.append(format_reward(reward_value))
            print(
                f"[STEP] step={steps_taken} action={action_text} reward={format_reward(reward_value)} "
                f"done={format_bool(done)} error={error_text}"
            )

            if done:
                break
    finally:
        print(
            f"[END] success={format_bool(success)} steps={steps_taken} "
            f"rewards={','.join(rewards)}"
        )


def main() -> None:
    client = build_openai_client()
    env = IncidentCommanderEnvironment()
    for task_id in TASKS:
        run_episode(env, client, task_id)


if __name__ == "__main__":
    main()
