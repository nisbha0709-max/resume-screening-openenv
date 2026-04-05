"""
grader.py — Deterministic, multi-factor reward grader for the Resume Screening OpenEnv.

Scoring components (all normalized to 0.0–1.0):
  1. skill_match_score     — keyword overlap between resume and required skills
  2. decision_correctness  — how correct the decision is vs. expected answer
  3. reasoning_quality     — presence of good reasoning keywords in agent output
  4. partial_credit        — awarded when decision is close but not perfect
  5. penalty               — deducted for clearly wrong decisions

Final reward = weighted combination, clamped to [0.0, 1.0]
"""

from models import Action, Reward, RewardBreakdown
from typing import Dict, Any


# ─── Weights ──────────────────────────────────────────────────────────────────
WEIGHT_SKILL       = 0.25
WEIGHT_DECISION    = 0.40
WEIGHT_REASONING   = 0.20
WEIGHT_PARTIAL     = 0.15

# Penalty multiplier for bad decisions (deducted from total)
PENALTY_WRONG_HARD  = 0.30   # e.g., accept when should reject
PENALTY_WRONG_SOFT  = 0.10   # e.g., shortlist when should accept/reject clearly


# ─── Decision Correctness Matrix ─────────────────────────────────────────────
# Maps (expected, given) → (correctness_score, penalty, partial_credit)
DECISION_MATRIX = {
    # Perfect matches
    ("accept",    "accept"):    (1.0, 0.0,  0.0),
    ("reject",    "reject"):    (1.0, 0.0,  0.0),
    ("shortlist", "shortlist"): (1.0, 0.0,  0.0),

    # Near misses — shortlist is often a reasonable middle ground
    ("accept",    "shortlist"): (0.4, 0.10, 0.5),
    ("reject",    "shortlist"): (0.4, 0.10, 0.5),
    ("shortlist", "accept"):    (0.3, 0.10, 0.3),
    ("shortlist", "reject"):    (0.3, 0.10, 0.3),

    # Hard errors — opposite of expected
    ("accept",    "reject"):    (0.0, 0.30, 0.0),
    ("reject",    "accept"):    (0.0, 0.30, 0.0),
}


def _skill_match_score(resume: str, required_skills: list) -> float:
    """
    Compute what fraction of required skills appear in the resume text.
    Case-insensitive keyword search.
    """
    if not required_skills:
        return 0.5  # neutral if no skills defined

    resume_lower = resume.lower()
    matched = sum(1 for skill in required_skills if skill.lower() in resume_lower)
    return round(matched / len(required_skills), 4)


def _reasoning_quality_score(reasoning: str, good_keywords: list) -> float:
    """
    Check if the agent's reasoning contains quality indicator keywords.
    Returns fraction of good keywords found (capped at 1.0).
    """
    if not good_keywords:
        return 0.5

    reasoning_lower = reasoning.lower()
    matched = sum(1 for kw in good_keywords if kw.lower() in reasoning_lower)
    # Even matching 1 keyword gives partial credit; 3+ gives full credit
    raw = matched / max(len(good_keywords), 1)
    # Scale so that matching ~30% of keywords = 0.5, 60%+ = 1.0
    return min(round(raw * 1.6, 4), 1.0)


def grade(action: Action, task: Dict[str, Any]) -> Reward:
    """
    Grade an agent's action against a task definition.
    Returns a fully populated Reward object.
    """
    expected   = task["expected_decision"]
    given      = action.decision
    resume     = task["resume"]
    req_skills = task.get("required_skills", [])
    good_kws   = task.get("good_reasoning_keywords", [])

    # ── Component 1: Skill match (resume vs. required skills) ────────────────
    skill_score = _skill_match_score(resume, req_skills)

    # ── Component 2: Decision correctness + penalty + partial credit ─────────
    correctness, penalty, partial_credit = DECISION_MATRIX.get(
        (expected, given), (0.0, PENALTY_WRONG_HARD, 0.0)
    )

    # ── Component 3: Reasoning quality ───────────────────────────────────────
    reasoning_score = _reasoning_quality_score(action.reasoning, good_kws)

    # ── Weighted total (before penalty) ──────────────────────────────────────
    weighted = (
        skill_score     * WEIGHT_SKILL    +
        correctness     * WEIGHT_DECISION +
        reasoning_score * WEIGHT_REASONING +
        partial_credit  * WEIGHT_PARTIAL
    )

    # ── Apply penalty ─────────────────────────────────────────────────────────
    total = max(0.0, min(1.0, round(weighted - penalty, 4)))

    # ── Build feedback string ─────────────────────────────────────────────────
    feedback_parts = [
        f"Expected: '{expected}' | Given: '{given}'.",
        f"Skill match: {skill_score:.0%} of required skills found in resume.",
        f"Decision correctness: {correctness:.0%}.",
        f"Reasoning quality: {reasoning_score:.0%} (keyword hits).",
    ]
    if partial_credit > 0:
        feedback_parts.append(f"Partial credit applied: +{partial_credit:.2f}.")
    if penalty > 0:
        feedback_parts.append(f"Penalty applied: -{penalty:.2f} (wrong decision).")
    feedback_parts.append(f"Final reward: {total:.4f}.")

    return Reward(
        total=total,
        breakdown=RewardBreakdown(
            skill_match_score=skill_score,
            decision_correctness=correctness,
            reasoning_quality=reasoning_score,
            partial_credit=partial_credit,
            penalty=penalty,
        ),
        feedback=" ".join(feedback_parts),
    )
