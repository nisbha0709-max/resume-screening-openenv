"""
models.py — Pydantic models for the Resume Screening OpenEnv.
All reward scores are strictly between 0.0 and 1.0 (exclusive).
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Action(BaseModel):
    decision: Literal["accept", "reject", "shortlist"] = Field(
        ..., description="Hiring decision."
    )
    reasoning: str = Field(..., min_length=10)


class HistoryEntry(BaseModel):
    step: int
    action: Action
    reward: float
    feedback: str


class Observation(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    job_description: str
    resume: str
    history: List[HistoryEntry] = Field(default_factory=list)
    step_count: int = 0


class RewardBreakdown(BaseModel):
    skill_match_score: float = Field(..., gt=0.0, lt=1.0)
    decision_correctness: float = Field(..., gt=0.0, lt=1.0)
    reasoning_quality: float = Field(..., gt=0.0, lt=1.0)
    partial_credit: float = Field(..., gt=0.0, lt=1.0)
    penalty: float = Field(..., gt=0.0, lt=1.0)


class Reward(BaseModel):
    total: float = Field(..., gt=0.0, lt=1.0)
    breakdown: RewardBreakdown
    feedback: str


class EnvState(BaseModel):
    task_id: str
    difficulty: str
    current_step: int
    done: bool
    cumulative_reward: float
    last_action: Optional[Action] = None
    last_reward: Optional[Reward] = None
