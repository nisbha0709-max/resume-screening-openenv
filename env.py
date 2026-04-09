"""
env.py - OpenEnv environment. All 3 tasks have registered graders.
"""

import logging
from typing import Optional, Tuple, Dict, Any

from models import Action, Observation, Reward, EnvState, HistoryEntry
from tasks import get_task, list_tasks, TASKS
from grader import grade

logger = logging.getLogger(__name__)


class ResumeScreeningEnv:

    def __init__(self):
        self._task: Optional[Dict[str, Any]] = None
        self._obs: Optional[Observation] = None
        self._state: Optional[EnvState] = None
        # Explicitly register grader for every task
        self._graders = {task_id: grade for task_id in TASKS.keys()}
        logger.info(f"[ENV] Graders registered: {list(self._graders.keys())}")

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
        logger.info(f"[ENV] Reset: task_id={task_id}")
        return self._obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._task is None or self._state is None:
            raise RuntimeError("Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode done. Call reset().")

        grader_fn = self._graders[self._task["task_id"]]
        reward = grader_fn(action, self._task)

        # Validate score is strictly in range
        assert 0.0 < reward.total < 1.0, f"Score out of range: {reward.total}"

        self._state.current_step += 1
        self._state.cumulative_reward += reward.total
        self._state.last_action = action
        self._state.last_reward = reward
        self._state.done = True

        self._obs.history.append(HistoryEntry(
            step=self._state.current_step,
            action=action,
            reward=reward.total,
            feedback=reward.feedback,
        ))
        self._obs.step_count = self._state.current_step

        info = {
            "task_id": self._task["task_id"],
            "difficulty": self._task["difficulty"],
            "expected_decision": self._task["expected_decision"],
            "has_grader": True,
            "score_in_range": 0.0 < reward.total < 1.0,
            "reward_breakdown": reward.breakdown.model_dump(),
            "cumulative_reward": self._state.cumulative_reward,
        }

        logger.info(f"[ENV] Step: decision={action.decision} reward={reward.total:.4f} in_range={info['score_in_range']}")
        return self._obs, reward, True, info

    def state(self) -> EnvState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def available_tasks(self):
        return list_tasks()

    def get_graders(self):
        return {tid: {"has_grader": True, "type": "deterministic"} for tid in self._graders}

