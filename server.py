"""
server.py — FastAPI server for the Resume Screening OpenEnv.

Exposes:
  GET  /             → health check + available tasks
  POST /reset        → reset environment, returns initial observation
  POST /step         → execute an action, returns obs + reward + done + info
  GET  /state        → current environment state
  GET  /tasks        → list all available tasks
"""

import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import Action, Observation, Reward, EnvState
from env import ResumeScreeningEnv

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Resume Screening OpenEnv",
    description=(
        "An OpenEnv-compliant environment where an AI agent performs resume screening "
        "and hiring decisions against job descriptions."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global environment instance ─────────────────────────────────────────────
env = ResumeScreeningEnv()


# ─── Request/Response schemas ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    """Health check and environment info."""
    return {
        "status": "ok",
        "environment": "Resume Screening OpenEnv",
        "version": "1.0.0",
        "available_tasks": env.available_tasks(),
    }


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with difficulty and expected decisions."""
    return {"tasks": env.available_tasks()}


@app.post("/reset", response_model=Observation)
async def reset(body: Optional[ResetRequest] = None):
    """
    Reset the environment to a new episode for the given task.
    Body is optional — defaults to task_easy if not provided.

    Args:
        task_id: 'task_easy' | 'task_medium' | 'task_hard'

    Returns:
        Initial observation (job description + resume).
    """
    try:
        task_id = body.task_id if body else "task_easy"
        obs = env.reset(task_id=task_id)
        logger.info(f"[API] /reset called with task_id='{task_id}'")
        return obs
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in /reset")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(body: StepRequest):
    """
    Execute one agent action.

    Args:
        action: { decision: 'accept'|'reject'|'shortlist', reasoning: str }

    Returns:
        observation, reward (with breakdown), done flag, info dict.
    """
    try:
        # Auto-reset to task_easy if environment not yet initialized
        if env._state is None or env._state.done:
            env.reset(task_id="task_easy")
        obs, reward, done, info = env.step(body.action)
        logger.info(
            f"[API] /step: decision={body.action.decision}, "
            f"reward={reward.total:.4f}, done={done}"
        )
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in /step")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=EnvState)
async def state():
    """Return the current internal state of the environment."""
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)


@app.get("/graders")
async def graders():
    """Return registered graders for all tasks."""
    return env.get_graders()
