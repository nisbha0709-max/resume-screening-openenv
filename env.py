"""
env.py — Core environment logic for the Resume Screening OpenEnv.
Each task has a grader. Scores are always strictly between 0 and 1.
"""

import logging
from typing import Optional, Tuple, Dict, Any

from models import Action, Observation, Reward, EnvState, HistoryEntry
from tasks import get_task, list_tasks, TASKS
from grader import grade

logger = logging.getLogger(__name__)


class ResumeScreeningEnv:
    """OpenEnv-compliant resume screening environment."""

    MAX_STEPS = 1

    def __init__(self):
        self._task: Optional[Dict[str, Any]] = None
        self._obs: Optional[Observation] = None
        self._state: Optional[EnvState] = None

        # Register graders for all tasks (required by OpenEnv validator)
        self._graders = {task_id: grade for task_id in TASKS.keys()}
        logger.info(f"[ENV] Registered graders for tasks: {list(self._graders.keys())}")

    def reset(self, task_id: str = "task_easy") -> Observation:
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

        logger.info(f"[ENV] Reset to task '{task_id}'")
        return self._obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._task is None or self._state is None:
            raise RuntimeError("Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode done. Call reset().")

        # Use registered grader for this task
        grader_fn = self._graders.get(self._task["task_id"], grade)
        reward = grader_fn(action, self._task)

        self._state.current_step += 1
        self._state.cumulative_reward += reward.total
        self._state.last_action = action
        self._state.last_reward = reward
        self._state.done = True

        history_entry = HistoryEntry(
            step=self._state.current_step,
            action=action,
            reward=reward.total,
            feedback=reward.feedback,
        )
        self._obs.history.append(history_entry)
        self._obs.step_count = self._state.current_step

        logger.info(f"[ENV] Step: decision={action.decision}, reward={reward.total:.4f}")

        info = {
            "task_id": self._task["task_id"],
            "difficulty": self._task["difficulty"],
            "expected_decision": self._task["expected_decision"],
            "reward_breakdown": reward.breakdown.model_dump(),
            "cumulative_reward": self._state.cumulative_reward,
            "grader": "deterministic_keyword_grader",
        }

        return self._obs, reward, True, info

    def state(self) -> EnvState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def available_tasks(self):
        return list_tasks()

    def get_graders(self):
        """Return registered grader info for all tasks."""
        return {
            task_id: {"grader": "deterministic_keyword_grader", "task_id": task_id}
            for task_id in self._graders.keys()
        }
