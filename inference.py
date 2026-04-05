"""
inference.py — Baseline agent for the Resume Screening OpenEnv.

Reads environment from:
  API_BASE_URL     (default: http://localhost:7860)
  MODEL_NAME       (default: gpt-4o-mini)
  OPENAI_API_KEY   (or HF_TOKEN for HuggingFace-hosted models)

Runs all 3 tasks, logs each step, and reports a reproducible score.

Logging format:
  [START]
  [STEP]  task=... decision=... reward=...
  [END]   total_score=...
"""

import os
import json
import logging
import requests
from openai import OpenAI

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY      = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")

TASK_IDS = ["task_easy", "task_medium", "task_hard"]

# ─── Prompt template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior HR professional and talent acquisition specialist.
You will be given a job description and a candidate's resume.
Your task is to evaluate the candidate and make a hiring decision.

You MUST respond with a JSON object in this exact format:
{
  "decision": "<accept|reject|shortlist>",
  "reasoning": "<detailed explanation of your decision>"
}

Guidelines:
- "accept": The candidate is a strong match for the role.
- "reject": The candidate clearly does not meet the requirements.
- "shortlist": The candidate partially matches and warrants further evaluation.

Your reasoning should be specific, referencing skills, experience, and qualifications.
Respond ONLY with the JSON object, no other text."""

USER_PROMPT_TEMPLATE = """Job Description:
{job_description}

---

Candidate Resume:
{resume}

---

Please evaluate this candidate for the role and provide your hiring decision."""


# ─── OpenAI client ────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    return OpenAI(
        api_key=API_KEY or "not-needed",
        base_url=None,  # Use default OpenAI endpoint; override for HF inference
    )


# ─── Environment API helpers ──────────────────────────────────────────────────

def env_reset(task_id: str) -> dict:
    """Call /reset on the environment server."""
    response = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def env_step(decision: str, reasoning: str) -> dict:
    """Call /step on the environment server."""
    response = requests.post(
        f"{API_BASE_URL}/step",
        json={"action": {"decision": decision, "reasoning": reasoning}},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# ─── Agent inference ─────────────────────────────────────────────────────────

def agent_decide(client: OpenAI, observation: dict) -> tuple[str, str]:
    """
    Call the LLM to make a hiring decision based on the observation.
    Returns (decision, reasoning).
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        job_description=observation["job_description"],
        resume=observation["resume"],
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,   # Deterministic output
        max_tokens=512,
    )

    raw = completion.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)
    return parsed["decision"], parsed["reasoning"]


# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    client = get_client()
    scores = []

    logger.info("[START]")
    logger.info(f"  model={MODEL_NAME}")
    logger.info(f"  api_base={API_BASE_URL}")
    logger.info(f"  tasks={TASK_IDS}")
    logger.info("")

    for task_id in TASK_IDS:
        logger.info(f"[STEP] task={task_id}")

        # 1. Reset environment
        try:
            obs = env_reset(task_id)
        except Exception as e:
            logger.error(f"  [ERROR] Failed to reset task '{task_id}': {e}")
            scores.append(0.0)
            continue

        logger.info(f"  difficulty={obs['difficulty']}")

        # 2. Agent makes a decision
        try:
            decision, reasoning = agent_decide(client, obs)
        except Exception as e:
            logger.error(f"  [ERROR] Agent failed on task '{task_id}': {e}")
            # Fallback: shortlist everything to get partial credit
            decision, reasoning = "shortlist", "Unable to generate reasoning due to an error."

        logger.info(f"  decision={decision}")
        logger.info(f"  reasoning_preview={reasoning[:120].replace(chr(10), ' ')}...")

        # 3. Step the environment
        try:
            result = env_step(decision, reasoning)
        except Exception as e:
            logger.error(f"  [ERROR] Failed to step environment: {e}")
            scores.append(0.0)
            continue

        reward_total = result["reward"]["total"]
        feedback     = result["reward"]["feedback"]
        breakdown    = result["reward"]["breakdown"]

        logger.info(f"  reward={reward_total:.4f}")
        logger.info(f"  skill_match={breakdown['skill_match_score']:.2f} "
                    f"decision={breakdown['decision_correctness']:.2f} "
                    f"reasoning={breakdown['reasoning_quality']:.2f} "
                    f"partial={breakdown['partial_credit']:.2f} "
                    f"penalty={breakdown['penalty']:.2f}")
        logger.info(f"  feedback={feedback}")
        logger.info("")

        scores.append(reward_total)

    # ─── Final score ──────────────────────────────────────────────────────────
    total_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    logger.info("[END]")
    logger.info(f"  scores_per_task={scores}")
    logger.info(f"  total_score={total_score:.4f}")
    logger.info(f"  max_possible=1.0000")

    return total_score


if __name__ == "__main__":
    main()
