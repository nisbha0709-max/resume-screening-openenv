"""
inference.py — Baseline agent for the Resume Screening OpenEnv.

Reads environment from:
  API_BASE_URL     (default: http://localhost:7860)
  MODEL_NAME       (default: gpt-4o-mini)
  OPENAI_API_KEY   (or HF_TOKEN for HuggingFace-hosted models)

Outputs STRICT structured logs to stdout:
  [START] task=...
  [STEP] step=... reward=...
  [END] task=... score=... steps=...
"""

import os
import json
import requests
from openai import OpenAI

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
Respond ONLY with the JSON object, no other text.
"""

USER_PROMPT_TEMPLATE = """Job Description:
{job_description}

---

Candidate Resume:
{resume}

---

Please evaluate this candidate for the role and provide your hiring decision.
"""

# ─── OpenAI client ────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    return OpenAI(
        api_key=API_KEY or "not-needed",
        base_url=None,
    )

# ─── Environment API helpers ──────────────────────────────────────────────────
def env_reset(task_id: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()

def env_step(decision: str, reasoning: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/step",
        json={"action": {"decision": decision, "reasoning": reasoning}},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()

# ─── Agent inference ─────────────────────────────────────────────────────────
def agent_decide(client: OpenAI, observation: dict) -> tuple[str, str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        job_description=observation["job_description"],
        resume=observation["resume"],
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    raw = completion.choices[0].message.content.strip()

    # Remove markdown formatting if present
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
    task_name = "resume_screening"

    # ✅ START block
    print(f"[START] task={task_name}", flush=True)

    step_count = 0

    for task_id in TASK_IDS:
        step_count += 1

        # Reset environment
        try:
            obs = env_reset(task_id)
        except Exception:
            reward_total = 0.0
            scores.append(reward_total)
            print(f"[STEP] step={step_count} reward={reward_total}", flush=True)
            continue

        # Agent decision
        try:
            decision, reasoning = agent_decide(client, obs)
        except Exception:
            decision, reasoning = "shortlist", "Fallback due to error"

        # Step environment
        try:
            result = env_step(decision, reasoning)
            reward_total = result["reward"]["total"]
        except Exception:
            reward_total = 0.0

        scores.append(reward_total)

        # ✅ STEP block (STRICT FORMAT)
        print(f"[STEP] step={step_count} reward={reward_total}", flush=True)

    # Final score
    total_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    # ✅ END block
    print(f"[END] task={task_name} score={total_score} steps={step_count}", flush=True)

    return total_score

# ─── Entry ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
