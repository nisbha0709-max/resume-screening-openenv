"""
inference.py — FINAL (Phase 2 Compliant)

✅ Uses API_BASE_URL + API_KEY (MANDATORY)
✅ Uses OpenAI client
✅ Sends calls through LiteLLM proxy
✅ Emits strict structured stdout logs
"""

import os
import json
import requests
from openai import OpenAI

# ─── REQUIRED CONFIG ──────────────────────────────────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]   # MUST use
API_KEY      = os.environ["API_KEY"]        # MUST use
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASK_IDS = ["task_easy", "task_medium", "task_hard"]

# ─── PROMPTS ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior HR professional.

Respond ONLY in JSON:
{
  "decision": "accept|reject|shortlist",
  "reasoning": "brief explanation"
}
"""

USER_PROMPT_TEMPLATE = """Job Description:
{job_description}

Candidate Resume:
{resume}

Evaluate the candidate.
"""

# ─── CLIENT (CRITICAL) ────────────────────────────────────────────────────────
def get_client():
    return OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL  # ensures proxy usage
    )

# ─── ENV API ──────────────────────────────────────────────────────────────────
def env_reset(task_id):
    r = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(decision, reasoning):
    r = requests.post(
        f"{API_BASE_URL}/step",
        json={"action": {"decision": decision, "reasoning": reasoning}},
        timeout=30
    )
    r.raise_for_status()
    return r.json()

# ─── AGENT ────────────────────────────────────────────────────────────────────
def agent_decide(client, obs):
    prompt = USER_PROMPT_TEMPLATE.format(
        job_description=obs["job_description"],
        resume=obs["resume"],
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)
    return parsed["decision"], parsed["reasoning"]

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    client = get_client()
    scores = []
    task_name = "resume_screening"

    # START
    print(f"[START] task={task_name}", flush=True)

    step_count = 0

    for task_id in TASK_IDS:
        step_count += 1

        # Reset
        try:
            obs = env_reset(task_id)
        except Exception:
            reward = 0.0
            scores.append(reward)
            print(f"[STEP] step={step_count} reward={reward}", flush=True)
            continue

        # Decide
        try:
            decision, reasoning = agent_decide(client, obs)
        except Exception:
            decision, reasoning = "shortlist", "fallback"

        # Step
        try:
            result = env_step(decision, reasoning)
            reward = result["reward"]["total"]
        except Exception:
            reward = 0.0

        scores.append(reward)

        # STEP
        print(f"[STEP] step={step_count} reward={reward}", flush=True)

    final_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    # END
    print(f"[END] task={task_name} score={final_score} steps={step_count}", flush=True)

    return final_score


if __name__ == "__main__":
    main()
