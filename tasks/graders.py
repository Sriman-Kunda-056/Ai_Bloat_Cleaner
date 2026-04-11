"""Reward functions for AI bloat detection tasks. All rewards in (0.0, 1.0)."""

from __future__ import annotations

from typing import Dict

from .definitions import TASKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use a generous epsilon so floating-point rounding can never produce 0.0 or 1.0
_EPS = 0.01
_MAX = 1.0 - _EPS   # 0.99  — never reach 1.0
_MIN = _EPS          # 0.01  — never reach 0.0


def _clamp_open_interval(score: float) -> float:
    """Clamp score to the strict open interval (0, 1).

    The interval is (_EPS, 1-_EPS) = (0.01, 0.99), which guarantees the
    score can never equal the boundary values 0.0 or 1.0 even after
    floating-point rounding.
    """
    return round(min(max(float(score), _MIN), _MAX), 4)


def _compute_ai_strength(signals: Dict) -> float:
    """
    Compute composite bloat signal strength.
    Combines ai_probability with binary signal indicators.
    Returns a value clamped to [0.05, 0.95] so downstream formulas
    can never push a score to exactly 0 or 1.
    """
    strength = float(signals.get("ai_probability", 0.05) or 0.05)

    signal_bonuses = [
        ("hidden_artifact_dir", 0.15),
        ("dependency_bloat", 0.15),
        ("build_output", 0.10),
        ("type_mismatch", 0.15),
        ("bytecode_artifact", 0.15),
        ("virtualenv_internal", 0.15),
        ("batch_creation", 0.08),
        ("duplicate_content", 0.08),
        ("ai_scaffold_name", 0.08),
        ("temp_draft", 0.08),
    ]

    for signal_name, bonus in signal_bonuses:
        if signals.get(signal_name):
            strength += bonus

    if signals.get("file_kind") == "directory":
        strength += 0.04

    # Clamp to [0.05, 0.95] — wide enough to be meaningful, narrow enough
    # so that *any* downstream linear formula stays away from 0 and 1.
    return min(0.95, max(0.05, strength))


def _grade_precision(action: str, signals: Dict) -> float:
    """
    Precision task: minimise false positives by deleting only obvious bloat.
    Correct action: DELETE for high-confidence bloat, SKIP for human files.
    """
    strength = _compute_ai_strength(signals)

    if action == "delete":
        # Max possible: 0.60 + 0.35*0.95 = 0.9325  →  well below 1.0
        raw = 0.60 + 0.35 * strength
    elif action == "flag":
        # Range: [0.41, 0.59]
        raw = 0.40 + 0.20 * strength
    else:  # skip
        # SKIP is correct for low-confidence; penalise for high-confidence bloat
        penalty = strength * 0.80
        raw = max(0.12, 0.40 - penalty)

    return _clamp_open_interval(raw)


def _grade_recall(action: str, signals: Dict) -> float:
    """
    Recall task: minimise false negatives by catching all bloat.
    Correct action: DELETE for any bloat-like signals.
    """
    strength = _compute_ai_strength(signals)

    if action == "delete":
        # Max possible: 0.50 + 0.44*0.95 = 0.918  →  below 1.0
        raw = 0.50 + 0.44 * strength
    elif action == "flag":
        # Range: [0.37, 0.65]
        raw = 0.35 + 0.30 * strength
    else:  # skip
        penalty = strength * 0.90
        raw = max(0.06, 0.20 - penalty)

    return _clamp_open_interval(raw)


def _grade_f1_score(action: str, signals: Dict) -> float:
    """
    F1 task: balance precision and recall for robust overall triage.
    Correct action: DELETE for obvious bloat, FLAG for marginal, SKIP for human.
    """
    strength = _compute_ai_strength(signals)

    if action == "delete":
        # Max possible: 0.55 + 0.38*0.95 = 0.911  →  below 1.0
        raw = 0.55 + 0.38 * strength
    elif action == "flag":
        # Range: [0.46, 0.69]
        raw = 0.45 + 0.25 * strength
    else:  # skip
        penalty = strength * 0.85
        raw = max(0.11, 0.35 - penalty)

    return _clamp_open_interval(raw)


def _grade_efficiency(action: str, signals: Dict) -> float:
    """
    Efficiency task: maximise bytes freed per decision under a limited budget.
    Correct action: prioritise DELETE for large, high-confidence bloat first.
    """
    strength = _compute_ai_strength(signals)
    size_bytes = float(signals.get("size_bytes", 0) or 0)

    # Size bonus: large files freed = more bytes saved.
    # Normalise to ~100MB; cap at 0.20 (reduced from 0.25) so the
    # combined formula can never reach 1.0.
    size_bonus = min(size_bytes / 100_000_000.0, 0.20)

    if action == "delete":
        # Max possible: 0.55 + 0.22*0.95 + 0.20 = 0.959  →  below 1.0
        raw = 0.55 + 0.22 * strength + size_bonus
    elif action == "flag":
        raw = 0.35 + 0.18 * strength + size_bonus * 0.40
    else:  # skip
        penalty = size_bonus * 0.70
        raw = max(0.06, 0.20 - penalty)

    return _clamp_open_interval(raw)


def grade_action(task_id: str, action: str, signals: dict) -> float:
    """
    Score a single action for a given bloat-detection task and signal state.
    Returns a float in (0.0, 1.0).
    """
    action = (action or "").lower().strip()
    # Normalise any variant spellings the LLM might emit
    if action not in ("delete", "flag", "skip"):
        # Try partial match
        for a in ("delete", "flag", "skip"):
            if a in action:
                action = a
                break
        else:
            return 0.1  # unrecognised action — small non-zero to avoid 0-reward crashes

    signals = signals or {}

    raw_score = 0.5
    if task_id == "precision":
        raw_score = _grade_precision(action, signals)
    elif task_id == "recall":
        raw_score = _grade_recall(action, signals)
    elif task_id == "f1_score":
        raw_score = _grade_f1_score(action, signals)
    elif task_id == "efficiency":
        raw_score = _grade_efficiency(action, signals)

    # OpenEnv requirement: scores must be strictly in (0.0, 1.0)
    return _clamp_open_interval(raw_score)


def grader_precision(base_url: str = "http://localhost:8000") -> float:
    """Precision grader: oracle performance on precision task."""
    task = TASKS["precision"]
    scores = []

    for step in task["steps"]:
        signals = step.get("signals", {})
        ideal_action = signals.get("ideal_action", task.get("ideal_action", "skip"))
        score = grade_action("precision", ideal_action, signals)
        scores.append(score)

    return _clamp_open_interval(sum(scores) / len(scores) if scores else 0.5)


def grader_recall(base_url: str = "http://localhost:8000") -> float:
    """Recall grader: oracle performance on recall task."""
    task = TASKS["recall"]
    scores = []

    for step in task["steps"]:
        signals = step.get("signals", {})
        ideal_action = signals.get("ideal_action", task.get("ideal_action", "skip"))
        score = grade_action("recall", ideal_action, signals)
        scores.append(score)

    return _clamp_open_interval(sum(scores) / len(scores) if scores else 0.5)


def grader_f1_score(base_url: str = "http://localhost:8000") -> float:
    """F1 grader: oracle performance on f1_score task."""
    task = TASKS["f1_score"]
    scores = []

    for step in task["steps"]:
        signals = step.get("signals", {})
        ideal_action = signals.get("ideal_action", task.get("ideal_action", "skip"))
        score = grade_action("f1_score", ideal_action, signals)
        scores.append(score)

    return _clamp_open_interval(sum(scores) / len(scores) if scores else 0.5)


def grader_efficiency(base_url: str = "http://localhost:8000") -> float:
    """Efficiency grader: oracle performance on efficiency task."""
    task = TASKS["efficiency"]
    scores = []

    for step in task["steps"]:
        signals = step.get("signals", {})
        ideal_action = signals.get("ideal_action", task.get("ideal_action", "skip"))
        score = grade_action("efficiency", ideal_action, signals)
        scores.append(score)

    return _clamp_open_interval(sum(scores) / len(scores) if scores else 0.5)


def run_all_graders(base_url: str = "http://localhost:8000") -> Dict[str, float]:
    return {
        "precision": grader_precision(base_url),
        "recall": grader_recall(base_url),
        "f1_score": grader_f1_score(base_url),
        "efficiency": grader_efficiency(base_url),
    }