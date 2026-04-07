#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from statistics import mean
from typing import Any

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
    try:
        if API_KEY and "API_BASE_URL" in os.environ:
            return OpenAI(
                base_url=os.environ["API_BASE_URL"],
                api_key=os.environ["API_KEY"],
            )
    except Exception:
        return None
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


def extract_message_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except Exception:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(part.strip() for part in parts if part).strip()
    return ""


def candidate_models(client: OpenAI | None, resolved_model_name: str | None) -> list[str]:
    candidates: list[str] = []
    for value in (
        resolved_model_name,
        os.getenv("MODEL"),
        os.getenv("LITELLM_MODEL"),
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4.1",
    ):
        if value and value not in candidates:
            candidates.append(value)
    if client is None:
        return candidates
    try:
        models = client.models.list()
        for model in models.data:
            model_id = getattr(model, "id", None)
            if isinstance(model_id, str) and model_id not in candidates:
                candidates.append(model_id)
    except Exception:
        pass
    return candidates


def maybe_request_plan(client: OpenAI | None, model_name: str | None, task_id: str) -> str:
    if client is None:
        return "baseline-plan"
    task = TASKS[task_id]
    for candidate in candidate_models(client, model_name):
        try:
            response = client.chat.completions.create(
                model=candidate,
                messages=[
                    {"role": "system", "content": "Return a short deterministic incident response plan label."},
                    {"role": "user", "content": f"Task: {task.title}. Return a concise plan label only."},
                ],
                temperature=0.0,
                max_tokens=16,
            )
            text = extract_message_text(response)
            if text:
                return text
        except Exception:
            continue
    return "baseline-plan"


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
        try:
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
        except Exception as exc:
            emit_block(
                "STEP",
                {
                    "task_id": task_id,
                    "step": steps_taken,
                    "action_type": action_type,
                    "target": target,
                    "reward": 0.0,
                    "done": True,
                    "score": 0.0,
                    "reason": f"step_failed:{type(exc).__name__}",
                },
            )
            done = True
            break
    try:
        state = env.state()
    except Exception:
        state = {
            "score": 0.0,
            "reward_total": 0.0,
            "turn": steps_taken,
        }
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
    scores: list[float] = []
    try:
        client = build_openai_client()
        model_name = resolve_model_name(client)
        env = IncidentCommanderEnvironment()

        for task_id in TASKS:
            try:
                plan_label = maybe_request_plan(client, model_name, task_id)
            except Exception:
                plan_label = "baseline-plan"
            result = run_task(env, task_id, plan_label)
            scores.append(float(result["score"]))
    except Exception as exc:
        emit_block("END", {"error": type(exc).__name__, "tasks_completed": len(scores)})
    finally:
        average_score = round(mean(scores), 4) if scores else 0.0
        emit_block("END", {"average_score": average_score, "tasks_completed": len(scores)})


if __name__ == "__main__":
    main()
