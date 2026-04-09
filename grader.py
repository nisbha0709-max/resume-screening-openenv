"""
grader.py — Deterministic, multi-factor reward grader for the Resume Screening OpenEnv.

IMPORTANT: All scores are strictly between 0.0 and 1.0 (exclusive).
Min score: 0.01, Max score: 0.99
"""

from models import Action, Reward, RewardBreakdown
from typing import Dict, Any


# ─── Weights ──────────────────────────────────────────────────────────────────
WEIGHT_SKILL       = 0.25
WEIGHT_DECISION    = 0.40
WEIGHT_REASONING   = 0.20
WEIGHT_PARTIAL     = 0.15

# ─── Score bounds (strictly between 0 and 1) ─────────────────────────────────
SCORE_MIN = 0.01
SCORE_MAX = 0.99

# ─── Decision Correctness Matrix ─────────────────────────────────────────────
# Maps (expected, given) → (correctness_score, penalty, partial_credit)
DECISION_MATRIX = {
    ("accept",    "accept"):    (0.95, 0.0,  0.0),
    ("reject",    "reject"):    (0.95, 0.0,  0.0),
    ("shortlist", "shortlist"): (0.95, 0.0,  0.0),

    ("accept",    "shortlist"): (0.40, 0.10, 0.45),
    ("reject",    "shortlist"): (0.40, 0.10, 0.45),
    ("shortlist", "accept"):    (0.30, 0.10, 0.30),
    ("shortlist", "reject"):    (0.30, 0.10, 0.30),

    ("accept",    "reject"):    (0.05, 0.25, 0.0),
    ("reject",    "accept"):    (0.05, 0.25, 0.0),
}


def _clamp(value: float) -> float:
    """Clamp to strictly (0.0, 1.0) — never exactly 0 or 1."""
    return round(max(SCORE_MIN, min(SCORE_MAX, value)), 4)


def _skill_match_score(resume: str, required_skills: list) -> float:
    """Keyword overlap between resume and required skills. Always strictly (0,1)."""
    if not required_skills:
        return 0.50

    resume_lower = resume.lower()
    matched = sum(1 for skill in required_skills if skill.lower() in resume_lower)
    raw = matched / len(required_skills)

    # Scale so it never hits exactly 0.0 or 1.0
    scaled = 0.05 + (raw * 0.90)
    return _clamp(scaled)


def _reasoning_quality_score(reasoning: str, good_keywords: list) -> float:
    """Keyword hits in agent reasoning. Always strictly (0,1)."""
    if not good_keywords:
        return 0.50

    reasoning_lower = reasoning.lower()
    matched = sum(1 for kw in good_keywords if kw.lower() in reasoning_lower)
    raw = matched / max(len(good_keywords), 1)

    # Scale: minimum 0.05 even with zero hits
    scaled = 0.05 + (min(raw * 1.6, 1.0) * 0.90)
    return _clamp(scaled)


def grade(action: Action, task: Dict[str, Any]) -> Reward:
    """
    Grade an agent's action against a task definition.
    Always returns a reward strictly between 0.0 and 1.0.
    """
    expected   = task["expected_decision"]
    given      = action.decision
    resume     = task["resume"]
    req_skills = task.get("required_skills", [])
    good_kws   = task.get("good_reasoning_keywords", [])

    # ── Component scores ─────────────────────────────────────────────────────
    skill_score = _skill_match_score(resume, req_skills)
    correctness, penalty, partial_credit = DECISION_MATRIX.get(
        (expected, given), (0.05, 0.25, 0.0)
    )
    reasoning_score = _reasoning_quality_score(action.reasoning, good_kws)

    # ── Weighted total ────────────────────────────────────────────────────────
    weighted = (
        skill_score     * WEIGHT_SKILL    +
        correctness     * WEIGHT_DECISION +
        reasoning_score * WEIGHT_REASONING +
        partial_credit  * WEIGHT_PARTIAL
    )

    # ── Apply penalty then clamp strictly between 0 and 1 ────────────────────
    total = _clamp(weighted - penalty)

    # ── Feedback ──────────────────────────────────────────────────────────────
    feedback_parts = [
        f"Expected: '{expected}' | Given: '{given}'.",
        f"Skill match: {skill_score:.2f}.",
        f"Decision correctness: {correctness:.2f}.",
        f"Reasoning quality: {reasoning_score:.2f}.",
    ]
    if partial_credit > 0:
        feedback_parts.append(f"Partial credit: +{partial_credit:.2f}.")
    if penalty > 0:
        feedback_parts.append(f"Penalty: -{penalty:.2f}.")
    feedback_parts.append(f"Final reward: {total:.4f}.")

    return Reward(
        total=total,
        breakdown=RewardBreakdown(
            skill_match_score=skill_score,
            decision_correctness=_clamp(correctness),
            reasoning_quality=reasoning_score,
            partial_credit=_clamp(partial_credit) if partial_credit > 0 else SCORE_MIN,
            penalty=_clamp(penalty) if penalty > 0 else SCORE_MIN,
        ),
        feedback=" ".join(feedback_parts),
    )
