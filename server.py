"""
server.py - FastAPI server for Resume Screening OpenEnv.
Endpoints: /reset, /step, /state, /tasks, /grade, /graders
"""

import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from models import Action, Observation, Reward, EnvState
from env import ResumeScreeningEnv
from tasks import TASKS
from grader import grade

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume Screening OpenEnv",
    version="1.0.0",
    description="OpenEnv-compliant resume screening environment with 3 graded tasks.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = ResumeScreeningEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class GradeRequest(BaseModel):
    task_id: str
    action: Action


@app.get("/")
async def health():
    return {
        "status": "ok",
        "environment": "Resume Screening OpenEnv",
        "version": "1.0.0",
        "num_tasks": len(TASKS),
        "tasks": [t for t in TASKS.keys()],
    }


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "task_id": t["task_id"],
                "difficulty": t["difficulty"],
                "expected_decision": t["expected_decision"],
                "has_grader": True,
            }
            for t in TASKS.values()
        ]
    }


@app.get("/graders")
async def get_graders():
    """Expose grader info for all tasks — required by OpenEnv validator."""
    return {
        task_id: {
            "task_id": task_id,
            "has_grader": True,
            "grader_type": "deterministic_keyword_grader",
            "score_range": {"min": 0.01, "max": 0.99},
            "expected_decision": TASKS[task_id]["expected_decision"],
        }
        for task_id in TASKS.keys()
    }


@app.post("/reset", response_model=Observation)
async def reset(body: Optional[ResetRequest] = None):
    try:
        task_id = body.task_id if body and body.task_id else "task_easy"
        obs = env.reset(task_id=task_id)
        logger.info(f"/reset task_id={task_id}")
        return obs
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(body: StepRequest):
    try:
        if env._state is None or env._state.done:
            env.reset(task_id="task_easy")
        obs, reward, done, info = env.step(body.action)
        logger.info(f"/step decision={body.action.decision} reward={reward.total:.4f}")
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade")
async def grade_action(body: GradeRequest):
    """
    Grade an action for a specific task directly.
    Used by the OpenEnv validator to check each task's grader.
    """
    try:
        if body.task_id not in TASKS:
            raise HTTPException(status_code=404, detail=f"Task '{body.task_id}' not found.")
        task = TASKS[body.task_id]
        reward = grade(body.action, task)
        logger.info(f"/grade task={body.task_id} decision={body.action.decision} score={reward.total:.4f}")
        return {
            "task_id": body.task_id,
            "score": reward.total,
            "reward": reward.model_dump(),
            "in_range": 0.0 < reward.total < 1.0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=EnvState)
async def state():
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
