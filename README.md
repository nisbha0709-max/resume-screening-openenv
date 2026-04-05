# Resume Screening AI Environment (OpenEnv)

A production-ready **OpenEnv-compliant simulation environment** where an AI agent
performs resume screening against job descriptions and makes hiring decisions.

---

## Problem Description

Hiring decisions are high-stakes, time-consuming, and prone to inconsistency.
This environment simulates a real-world HR screening task where an LLM agent
is given a job description and a candidate resume, and must decide:

- **Accept** — forward the candidate to interviews
- **Reject** — decline the application
- **Shortlist** — flag for further human review

The environment grades these decisions across multiple dimensions: skill alignment,
decision correctness, and quality of reasoning.

---

## Real-World Motivation

Companies process thousands of applications per open role. Automating the first
screening pass reduces time-to-hire and allows recruiters to focus on final
evaluation. This environment helps evaluate and improve AI agents for this task,
with a reward function that penalizes wrong decisions and rewards nuanced reasoning.

---

## Action Space

Each action must include:

| Field      | Type                              | Description                        |
|------------|-----------------------------------|------------------------------------|
| `decision` | `"accept"` / `"reject"` / `"shortlist"` | Hiring decision for the candidate  |
| `reasoning`| `string` (min 10 chars)           | Explanation justifying the decision|

```json
{
  "decision": "accept",
  "reasoning": "The candidate has 7 years of ML engineering experience, strong Python and TensorFlow skills, and prior production deployment experience — an excellent match for this senior role."
}
```

---

## Observation Space

Each observation contains:

| Field             | Type                        | Description                          |
|-------------------|-----------------------------|--------------------------------------|
| `task_id`         | `string`                    | Unique task identifier               |
| `difficulty`      | `easy` / `medium` / `hard`  | Task difficulty level                |
| `job_description` | `string`                    | Full job description text            |
| `resume`          | `string`                    | Full candidate resume text           |
| `history`         | `list[HistoryEntry]`        | Past actions and rewards in episode  |
| `step_count`      | `int`                       | Current step number                  |

---

## Reward Design

Rewards are **partial** (not binary) and always in range `[0.0, 1.0]`.

### Components

| Component            | Weight | Description                                              |
|----------------------|--------|----------------------------------------------------------|
| `skill_match_score`  | 0.25   | Keyword overlap: required skills vs. resume text         |
| `decision_correctness`| 0.40  | How correct the decision is vs. expected answer          |
| `reasoning_quality`  | 0.20   | Presence of quality indicator keywords in reasoning      |
| `partial_credit`     | 0.15   | Credit for near-correct decisions (e.g. shortlist vs accept) |

### Penalty

| Condition                         | Deduction |
|-----------------------------------|-----------|
| Wrong decision (opposite expected)| −0.30     |
| Off-by-one decision               | −0.10     |

### Example Scoring

```
Perfect decision + strong reasoning:  ~0.85–1.0
Shortlist instead of accept/reject:   ~0.45–0.60
Completely wrong decision:            ~0.00–0.25
```

---

## Task Descriptions

### Task 1 — Easy (`task_easy`)

- **Role**: Senior ML Engineer
- **Candidate**: 7-year ML engineer with TensorFlow, Python, SQL, scikit-learn, production deployments
- **Expected decision**: `accept`
- **Why**: Strong, unambiguous match across all required skills and experience level

### Task 2 — Medium (`task_medium`)

- **Role**: Senior DevOps / Platform Engineer (Kubernetes, Terraform, AWS, CI/CD)
- **Candidate**: Junior web developer with 2 years experience, basic HTML/CSS/JS, no K8s/Terraform
- **Expected decision**: `reject`
- **Why**: Clear skill mismatch — candidate is in a completely different domain

### Task 3 — Hard (`task_hard`)

- **Role**: Senior Product Manager, B2B SaaS (5+ years PM experience required)
- **Candidate**: Software engineer with 1 year informal hybrid PM experience, relevant domain (B2B SaaS), strong analytical skills but not a dedicated PM
- **Expected decision**: `shortlist`
- **Why**: Ambiguous — transferable skills and domain familiarity, but lacks formal PM tenure

---

## Project Structure

```
/
├── server.py          FastAPI server (/reset, /step, /state, /tasks)
├── env.py             OpenEnv environment logic
├── models.py          Pydantic models (Observation, Action, Reward, EnvState)
├── tasks.py           Task definitions (easy / medium / hard)
├── grader.py          Deterministic multi-factor reward grader
├── inference.py       Baseline agent (OpenAI client)
├── openenv.yaml       OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup Instructions

### Local (Python)

```bash
# 1. Clone / download the project
cd resume-screening-openenv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python server.py
# Server runs at http://localhost:7860

# 4. Test the API
curl http://localhost:7860/
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'
```

---

## Docker Instructions

```bash
# Build
docker build -t resume-screening-openenv .

# Run
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  -e MODEL_NAME=gpt-4o-mini \
  resume-screening-openenv

# Server is available at http://localhost:7860
```

---

## Running inference.py

```bash
# Set environment variables
export API_BASE_URL=http://localhost:7860
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-your-key-here

# Run baseline agent
python inference.py
```

### Expected output format

```
[START]
  model=gpt-4o-mini
  api_base=http://localhost:7860
  tasks=['task_easy', 'task_medium', 'task_hard']

[STEP] task=task_easy
  difficulty=easy
  decision=accept
  reasoning_preview=The candidate has 7 years of ML engineering experience...
  reward=0.8750
  skill_match=0.86 decision=1.00 reasoning=0.88 partial=0.00 penalty=0.00
  feedback=Expected: 'accept' | Given: 'accept'. Skill match: 86% ...

[STEP] task=task_medium
  ...

[STEP] task=task_hard
  ...

[END]
  scores_per_task=[0.875, 0.85, 0.72]
  total_score=0.8150
  max_possible=1.0000
```

---

## Expected Baseline Score

Using GPT-4o-mini with temperature=0.0:

| Task        | Expected Score |
|-------------|---------------|
| task_easy   | ~0.85–0.95    |
| task_medium | ~0.80–0.90    |
| task_hard   | ~0.65–0.80    |
| **Average** | **~0.77–0.88**|

Scores may vary slightly based on model behavior, but grading is deterministic
given a fixed decision and fixed reasoning text.

---

## HuggingFace Spaces Deployment

This project is designed to deploy directly to HuggingFace Spaces using the Docker SDK.

1. Create a new Space with **Docker** SDK
2. Upload all project files (or push via git)
3. Set the following Space secrets:
   - `OPENAI_API_KEY` — your OpenAI key (for running inference.py)
   - `MODEL_NAME` — e.g. `gpt-4o-mini`
4. The Space will auto-build and expose the environment at your Space URL

---

## License

MIT
