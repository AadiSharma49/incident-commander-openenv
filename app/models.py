from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Difficulty = Literal["easy", "medium", "hard"]
ActionType = Literal["inspect", "diagnose", "mitigate", "validate", "communicate"]


class Observation(BaseModel):
    task_id: str
    difficulty: Difficulty
    system_status: str
    alerts: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    available_actions: list[ActionType] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)
    completed_actions: list[str] = Field(default_factory=list)
    turns_remaining: int = Field(ge=0)


class Action(BaseModel):
    action_type: ActionType
    target: str
    note: str | None = None


class Reward(BaseModel):
    score: float
    reason: str


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: str | None = None


class StateSnapshot(BaseModel):
    task_id: str
    difficulty: Difficulty
    turn: int = Field(ge=0)
    max_turns: int = Field(ge=1)
    done: bool
    score: float = Field(ge=0.0, le=1.0)
    reward_total: float
    diagnosis: str | None = None
    fixes_applied: list[str] = Field(default_factory=list)
    validations_passed: list[str] = Field(default_factory=list)
    communications_sent: list[str] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)
    action_history: list[str] = Field(default_factory=list)
    repeated_useless_actions: int = Field(ge=0)
    wrong_actions: int = Field(ge=0)
