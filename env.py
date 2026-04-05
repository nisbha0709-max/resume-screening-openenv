"""
env.py — Core environment logic for the Resume Screening OpenEnv.

Implements the OpenEnv interface:
  - reset(task_id)  → Observation
  - step(action)    → (Observation, Reward, done, info)
  - state()         → EnvState
"""

import logging
from typing import Optional, Tuple, Dict, Any

from models import Action, Observation, Reward, EnvState, HistoryEntry
from tasks import get_task, list_tasks
from grader import grade

logger = logging.getLogger(__name__)


class ResumeScreeningEnv:
    """
    OpenEnv-compliant environment for AI-driven resume screening.

    One episode = one task (one job description + one resume).
    The agent takes a single action (decision + reasoning).
    The episode terminates after that action.
    """

    MAX_STEPS = 1  # Each episode is a single decision

    def __init__(self):
        self._task: Optional[Dict[str, Any]] = None
        self._obs: Optional[Observation] = None
        self._state: Optional[EnvState] = None

    # ─── reset ────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "task_easy") -> Observation:
        """
        Reset the environment to the beginning of a new episode.

        Args:
            task_id: One of 'task_easy', 'task_medium', 'task_hard'.

        Returns:
            Initial Observation (job description + resume, empty history).
        """
        self._task = get_task(task_id)

        self._obs = Observation(
            task_id=self._task["task_id"],
            difficulty=self._task["difficulty"],
            job_description=self._task["job_description"],
            resume=self._task["resume"],
            history=[],
            step_count=0,
        )

        self._state = EnvState(
            task_id=self._task["task_id"],
            difficulty=self._task["difficulty"],
            current_step=0,
            done=False,
            cumulative_reward=0.0,
        )

        logger.info(f"[ENV] Reset to task '{task_id}' (difficulty={self._task['difficulty']})")
        return self._obs

    # ─── step ─────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one agent action.

        Args:
            action: Agent's decision + reasoning.

        Returns:
            (observation, reward, done, info)
        """
        if self._task is None or self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._state.done:
            raise RuntimeError("Episode already done. Call reset() to start a new episode.")

        # ── Grade the action ──────────────────────────────────────────────────
        reward = grade(action, self._task)

        # ── Update step counter ───────────────────────────────────────────────
        self._state.current_step += 1
        self._state.cumulative_reward += reward.total
        self._state.last_action = action
        self._state.last_reward = reward

        # ── Episode ends after one decision ───────────────────────────────────
        done = True
        self._state.done = done

        # ── Append to observation history ─────────────────────────────────────
        history_entry = HistoryEntry(
            step=self._state.current_step,
            action=action,
            reward=reward.total,
            feedback=reward.feedback,
        )
        self._obs.history.append(history_entry)
        self._obs.step_count = self._state.current_step

        logger.info(
            f"[ENV] Step {self._state.current_step}: "
            f"decision={action.decision}, reward={reward.total:.4f}, done={done}"
        )

        info = {
            "task_id": self._task["task_id"],
            "difficulty": self._task["difficulty"],
            "expected_decision": self._task["expected_decision"],
            "reward_breakdown": reward.breakdown.model_dump(),
            "cumulative_reward": self._state.cumulative_reward,
        }

        return self._obs, reward, done, info

    # ─── state ────────────────────────────────────────────────────────────────

    def state(self) -> EnvState:
        """Return the current internal state of the environment."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    # ─── utility ──────────────────────────────────────────────────────────────

    def available_tasks(self):
        """Return a list of available task summaries."""
        return list_tasks()
