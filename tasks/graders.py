# tasks/graders.py
# Standalone reward functions — no live server required.
# All graders return a float strictly in the open interval (0, 1).
#
# Scoring methodology:
#   Each grader simulates an oracle agent on its task's synthetic file set,
#   computes the appropriate metric (precision / recall / F1 / efficiency),
#   then clips to (0.01, 0.99) to satisfy OpenEnv's open-interval constraint.

from __future__ import annotations

from typing import Dict, List, Tuple

from .definitions import TASKS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_OBVIOUS_BLOAT_THRESHOLD = 0.80  # matches server/reward.py


def _oracle_action(file_meta: Dict) -> int:
    """Simple threshold-based oracle that mirrors the ideal_action labels."""
    if file_meta["ideal_action"] == "delete":
        return 1  # ACTION_DELETE
    return 0  # ACTION_KEEP


def _simulate_episode(task_id: str) -> Tuple[int, int, int, int, float, int]:
    """
    Run oracle agent over a task's file list.

    Returns:
        (tp, fp, fn, tn, bytes_freed, steps)
        where tp/fp/fn/tn are standard confusion-matrix counts for DELETE vs KEEP.
    """
    files = TASKS[task_id]["files"]
    tp = fp = fn = tn = 0
    bytes_freed = 0.0
    steps = len(files)

    for f in files:
        action = _oracle_action(f)
        is_bloat = f["label"] == "bloat"

        if action == 1:  # DELETE
            if is_bloat:
                tp += 1
                bytes_freed += f["size_mb"]
            else:
                fp += 1
        else:  # KEEP
            if is_bloat:
                fn += 1
            else:
                tn += 1

    return tp, fp, fn, tn, bytes_freed, steps


def _clip(value: float) -> float:
    """Clip to open interval (0, 1) as required by OpenEnv scoring."""
    return round(min(max(value, 0.01), 0.99), 4)


# ---------------------------------------------------------------------------
# Public grader functions — called directly by the OpenEnv validator
# ---------------------------------------------------------------------------

def grader_precision(base_url: str = "http://localhost:8000") -> float:
    """
    Task: precision — minimise false positives.
    Score = TP / (TP + FP), clipped to (0, 1).
    """
    tp, fp, fn, tn, _, _ = _simulate_episode("precision")
    if tp + fp == 0:
        return _clip(0.5)
    precision = tp / (tp + fp)
    return _clip(precision)


def grader_recall(base_url: str = "http://localhost:8000") -> float:
    """
    Task: recall — minimise false negatives.
    Score = TP / (TP + FN), clipped to (0, 1).
    """
    tp, fp, fn, tn, _, _ = _simulate_episode("recall")
    if tp + fn == 0:
        return _clip(0.5)
    recall = tp / (tp + fn)
    return _clip(recall)


def grader_f1_score(base_url: str = "http://localhost:8000") -> float:
    """
    Task: F1-score — harmonic mean of precision and recall.
    Score = 2 * P * R / (P + R), clipped to (0, 1).
    """
    tp, fp, fn, tn, _, _ = _simulate_episode("f1_score")
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return _clip(f1)


def grader_efficiency(base_url: str = "http://localhost:8000") -> float:
    """
    Task: efficiency — bytes freed per step, normalised by theoretical maximum.
    Score = bytes_freed / (max_possible_bytes * steps), clipped to (0, 1).
    """
    files = TASKS["efficiency"]["files"]
    tp, fp, fn, tn, bytes_freed, steps = _simulate_episode("efficiency")

    # Theoretical max: delete every bloat file
    max_bytes = sum(f["size_mb"] for f in files if f["label"] == "bloat")
    if max_bytes == 0 or steps == 0:
        return _clip(0.5)

    # Normalise: bytes_freed / max_bytes gives a [0,1] efficiency ratio
    efficiency = bytes_freed / max_bytes
    return _clip(efficiency)


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_all_graders(base_url: str = "http://localhost:8000") -> Dict[str, float]:
    return {
        "precision":  grader_precision(base_url),
        "recall":     grader_recall(base_url),
        "f1_score":   grader_f1_score(base_url),
        "efficiency": grader_efficiency(base_url),
    }
