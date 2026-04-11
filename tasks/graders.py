"""Reward functions for AI bloat detection tasks. All rewards in (0.0, 1.0)."""

from __future__ import annotations

from typing import Dict

from .definitions import TASKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_open_interval(score: float, eps: float = 0.001) -> float:
    """Clamp score to the strict open interval (0, 1) with stable rounding."""
    return round(min(max(float(score), eps), 1.0 - eps), 3)


def _compute_ai_strength(signals: Dict) -> float:
    """
    Compute composite bloat signal strength.
    Combines ai_probability with binary signal indicators.
    Returns [0.0, 1.0].
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

    return min(0.99, strength)


def _grade_precision(action: str, signals: Dict) -> float:
    """
    Precision task: minimise false positives by deleting only obvious bloat.
    Correct action: DELETE for high-confidence bloat, SKIP for human files.
    """
    strength = _compute_ai_strength(signals)

    if action == "delete":
        # Reward DELETE, but only if confidence is high
        return round(min(0.65 + 0.35 * strength, 1.0), 3)
    elif action == "flag":
        # Moderate credit for uncertain cases
        return round(0.40 + 0.20 * strength, 3)
    else:  # skip
        # SKIP is correct for low-confidence files (avoid false positives)
        penalty = strength * 0.80  # penalise skipping high-confidence bloat
        return round(max(0.10, 0.40 - penalty), 3)


def _grade_recall(action: str, signals: Dict) -> float:
    """
    Recall task: minimise false negatives by catching all bloat.
    Correct action: DELETE for any bloat-like signals.
    """
    strength = _compute_ai_strength(signals)

    if action == "delete":
        # Strong reward for deleting anything with bloat signals
        return round(min(0.55 + 0.45 * strength, 1.0), 3)
    elif action == "flag":
        # Partial credit for marking uncertain cases
        return round(0.35 + 0.30 * strength, 3)
    else:  # skip
        # Penalise skipping potential bloat
        penalty = strength * 0.90
        return round(max(0.05, 0.20 - penalty), 3)


def _grade_f1_score(action: str, signals: Dict) -> float:
    """
    F1 task: balance precision and recall for robust overall triage.
    Correct action: DELETE for obvious bloat, FLAG for marginal, SKIP for human.
    """
    strength = _compute_ai_strength(signals)

    if action == "delete":
        # Reward DELETE with strength-dependent scoring
        return round(min(0.60 + 0.40 * strength, 1.0), 3)
    elif action == "flag":
        # FLAG gets moderate rewards across all certainty levels
        return round(0.45 + 0.25 * strength, 3)
    else:  # skip
        # SKIP is OK for low-confidence, but penalise for high-confidence bloat
        penalty = strength * 0.85
        return round(max(0.10, 0.35 - penalty), 3)


def _grade_efficiency(action: str, signals: Dict) -> float:
    """
    Efficiency task: maximise bytes freed per decision under a limited budget.
    Correct action: prioritise DELETE for large, high-confidence bloat first.
    """
    strength = _compute_ai_strength(signals)
    size_bytes = float(signals.get("size_bytes", 0) or 0)

    # Size bonus: large files freed = more bytes saved
    # Normalise to ~100MB as typical large bloat (node_modules, dist, venv)
    size_bonus = min(size_bytes / 100_000_000.0, 0.25)

    if action == "delete":
        # Strong reward for DELETE, especially large high-confidence bloat
        return round(min(0.60 + 0.25 * strength + size_bonus, 1.0), 3)
    elif action == "flag":
        # Partial credit for flagging (less bytes freed than delete)
        return round(0.35 + 0.20 * strength + size_bonus * 0.40, 3)
    else:  # skip
        # Penalise skipping large bloat files
        penalty = size_bonus * 0.70
        return round(max(0.05, 0.20 - penalty), 3)


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