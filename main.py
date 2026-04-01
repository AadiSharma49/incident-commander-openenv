from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from app.env.env import IncidentCommanderEnvironment
from app.models import Action, Observation, ResetRequest, StateSnapshot, StepResponse
from app.tasks.tasks import TASKS

app = FastAPI(
    title="incident-commander-openenv",
    description="Deterministic OpenEnv incident response simulator.",
    version="1.0.0",
)

env = IncidentCommanderEnvironment()


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "incident-commander-openenv",
        "entrypoint": "main.py",
        "routes": ["POST /reset", "POST /step", "GET /state"],
        "tasks": list(TASKS),
    }


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest | None = None) -> Observation:
    payload = request or ResetRequest()
    try:
        return env.reset(task_id=payload.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    try:
        observation, reward, done, info = env.step(action)
        return StepResponse(observation=observation, reward=reward, done=done, info=info)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=StateSnapshot)
def state() -> StateSnapshot:
    try:
        return StateSnapshot.model_validate(env.state())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
