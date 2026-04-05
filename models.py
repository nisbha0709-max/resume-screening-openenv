"""
models.py — Pydantic models for the Resume Screening OpenEnv.
Defines typed structures for Observation, Action, and Reward.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ─── Action ───────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """Agent's hiring decision and reasoning for a given resume."""
    decision: Literal["accept", "reject", "shortlist"] = Field(
        ..., description="Hiring decision: accept, reject, or shortlist the candidate."
    )
    reasoning: str = Field(
        ..., min_length=10, description="Agent's explanation for the decision."
    )


# ─── Observation ──────────────────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    """A single past step in the episode."""
    step: int
    action: Action
    reward: float
    feedback: str  # human-readable grader feedback


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    job_description: str
    resume: str
    history: List[HistoryEntry] = Field(default_factory=list)
    step_count: int = 0


# ─── Reward ───────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """Fine-grained scoring components."""
    skill_match_score: float = Field(..., ge=0.0, le=1.0)
    decision_correctness: float = Field(..., ge=0.0, le=1.0)
    reasoning_quality: float = Field(..., ge=0.0, le=1.0)
    partial_credit: float = Field(..., ge=0.0, le=1.0)
    penalty: float = Field(..., ge=0.0, le=1.0)  # how much was deducted


class Reward(BaseModel):
    """Composite reward with breakdown."""
    total: float = Field(..., ge=0.0, le=1.0)
    breakdown: RewardBreakdown
    feedback: str  # narrative explanation


# ─── Environment State ────────────────────────────────────────────────────────

class EnvState(BaseModel):
    """Internal state of the environment."""
    task_id: str
    difficulty: str
    current_step: int
    done: bool
    cumulative_reward: float
    last_action: Optional[Action] = None
    last_reward: Optional[Reward] = None
