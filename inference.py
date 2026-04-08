"""
inference.py — Resume Screening OpenEnv (Validator-Compliant)

Requirements satisfied:
✅ Uses LiteLLM proxy (API_BASE_URL + API_KEY)
✅ Prints structured output
"""

import os
import json
import requests
from openai import OpenAI

# ─── Config (STRICT — DO NOT CHANGE) ──────────────────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]   
API_KEY      = os.environ["API_KEY"]        
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASK_IDS = ["task_easy", "task_medium", "task_hard"]

# ─── Prompts ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior HR professional and talent acquisition specialist.
You MUST respond with JSON:
{
  "decision": "<accept|reject|shortlist>",
  "reasoning": "<clear explanation>"
}
"""

USER_PROMPT_TEMPLATE = """Job Description:
{job_description}

Candidate Resume:
{resume}

Evaluate the candidate.
"""

# ─── OpenAI Client (FIXED) ────────────────────────────────────────────────────
def get_client() -> OpenAI:
    return OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,  
    )

# ─── Environment API ──────────────────────────────────────────────────────────
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

# ─── Agent ────────────────────────────────────────────────────────────────────
def agent_decide(client: OpenAI, observation: dict):
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
        max_tokens=300,
    )

    raw = completion.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)
    return parsed["decision"], parsed["reasoning"]

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    client = get_client()
    scores = []
    task_name = "resume_screening"

    # ✅ START
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

        
        print(f"[STEP] step={step_count} reward={reward}", flush=True)

    # Final score
    final_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    
    print(f"[END] task={task_name} score={final_score} steps={step_count}", flush=True)

    return final_score


if __name__ == "__main__":
    main()
