"""
grader.py - Deterministic grader. All scores strictly between 0.0 and 1.0.
"""

from models import Action, Reward, RewardBreakdown
from typing import Dict, Any


def _clamp(value: float) -> float:
    """Force value to be strictly between 0.0 and 1.0."""
    return round(min(0.99, max(0.01, float(value))), 4)


def _skill_match(resume: str, skills: list) -> float:
    if not skills:
        return 0.50
    resume_lower = resume.lower()
    matched = sum(1 for s in skills if s.lower() in resume_lower)
    ratio = matched / len(skills)
    # Scale to 0.10 - 0.90 to avoid 0.0 or 1.0
    return _clamp(0.10 + ratio * 0.80)


def _reasoning_score(reasoning: str, keywords: list) -> float:
    if not keywords:
        return 0.50
    r = reasoning.lower()
    matched = sum(1 for k in keywords if k.lower() in r)
    ratio = matched / len(keywords)
    return _clamp(0.10 + min(ratio * 1.5, 1.0) * 0.80)


# Maps (expected, given) -> (base_score, penalty, partial)
MATRIX = {
    ("accept",    "accept"):    (0.90, 0.00, 0.00),
    ("reject",    "reject"):    (0.90, 0.00, 0.00),
    ("shortlist", "shortlist"): (0.90, 0.00, 0.00),
    ("accept",    "shortlist"): (0.40, 0.05, 0.40),
    ("reject",    "shortlist"): (0.40, 0.05, 0.40),
    ("shortlist", "accept"):    (0.35, 0.05, 0.25),
    ("shortlist", "reject"):    (0.35, 0.05, 0.25),
    ("accept",    "reject"):    (0.08, 0.20, 0.00),
    ("reject",    "accept"):    (0.08, 0.20, 0.00),
}


def grade(action: Action, task: Dict[str, Any]) -> Reward:
    expected = task["expected_decision"]
    given = action.decision

    skill  = _skill_match(task["resume"], task.get("required_skills", []))
    reason = _reasoning_score(action.reasoning, task.get("good_reasoning_keywords", []))
    base, penalty, partial = MATRIX.get((expected, given), (0.08, 0.20, 0.00))

    raw = (skill * 0.25) + (base * 0.40) + (reason * 0.20) + (partial * 0.15) - penalty
    total = _clamp(raw)

    return Reward(
        total=total,
        breakdown=RewardBreakdown(
            skill_match_score=skill,
            decision_correctness=_clamp(base),
            reasoning_quality=reason,
            partial_credit=_clamp(partial) if partial > 0 else 0.01,
            penalty=_clamp(penalty) if penalty > 0 else 0.01,
        ),
        feedback=(
            f"Expected '{expected}', got '{given}'. "
            f"skill={skill:.2f} decision={base:.2f} "
            f"reasoning={reason:.2f} total={total:.4f}"
        ),
    )
