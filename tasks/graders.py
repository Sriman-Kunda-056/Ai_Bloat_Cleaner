"""Grader functions for AI Bloat Detector tasks.

Each public grader function returns a float **strictly** in the open
interval (0, 1).  The OpenEnv validator rejects any value that equals
exactly 0.0 or 1.0, so every code path is guarded by _safe_score().

This file is intentionally self-contained — it defines its own task
data and helper constants so it never fails due to a missing import.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Hard bounds: every score is forced into this interval before returning
# ---------------------------------------------------------------------------
_LO: float = 0.05   # minimum score — safely above 0.0
_HI: float = 0.95   # maximum score — safely below 1.0


def _safe_score(value: float) -> float:
    """Return value clamped to (_LO, _HI), rounded to 4 decimal places.

    This is the single chokepoint through which every grader result
    passes, making it impossible to return 0.0 or 1.0.
    """
    clamped = min(_HI, max(_LO, float(value)))
    return round(clamped, 4)


# ---------------------------------------------------------------------------
# Inline task definitions (avoids any dependency on definitions.py)
# ---------------------------------------------------------------------------
# Each entry: (task_id, ideal_action, ai_probability, bonus_signals, size_bytes)
_TASK_STEPS: Dict[str, List[Dict]] = {
    "precision": [
        {"ideal": "delete", "ai_prob": 0.98, "bonuses": ["hidden_artifact_dir"], "is_dir": True,  "size": 4096},
        {"ideal": "skip",   "ai_prob": 0.09, "bonuses": [],                        "is_dir": False, "size": 2432},
        {"ideal": "delete", "ai_prob": 0.94, "bonuses": ["type_mismatch"],          "is_dir": False, "size": 8192},
    ],
    "recall": [
        {"ideal": "delete", "ai_prob": 0.97, "bonuses": ["virtualenv_internal"],    "is_dir": False, "size": 101},
        {"ideal": "delete", "ai_prob": 0.99, "bonuses": ["bytecode_artifact"],      "is_dir": False, "size": 2048},
        {"ideal": "delete", "ai_prob": 0.88, "bonuses": ["dependency_bloat"],       "is_dir": False, "size": 512000},
    ],
    "f1_score": [
        {"ideal": "delete", "ai_prob": 0.86, "bonuses": ["ai_scaffold_name", "batch_creation"], "is_dir": False, "size": 6144},
        {"ideal": "skip",   "ai_prob": 0.18, "bonuses": [],                                      "is_dir": False, "size": 2048},
        {"ideal": "delete", "ai_prob": 0.83, "bonuses": ["temp_draft", "duplicate_content"],     "is_dir": False, "size": 1280},
    ],
    "efficiency": [
        {"ideal": "delete", "ai_prob": 0.95, "bonuses": ["dependency_bloat"], "is_dir": True,  "size": 78643200},
        {"ideal": "delete", "ai_prob": 0.89, "bonuses": ["build_output"],     "is_dir": True,  "size": 31457280},
        {"ideal": "skip",   "ai_prob": 0.14, "bonuses": [],                   "is_dir": False, "size": 512},
    ],
}

# Bonus weights for each signal type
_BONUS_WEIGHTS: Dict[str, float] = {
    "hidden_artifact_dir":  0.10,
    "dependency_bloat":     0.10,
    "build_output":         0.08,
    "type_mismatch":        0.10,
    "bytecode_artifact":    0.10,
    "virtualenv_internal":  0.10,
    "batch_creation":       0.06,
    "duplicate_content":    0.06,
    "ai_scaffold_name":     0.06,
    "temp_draft":           0.06,
}


# ---------------------------------------------------------------------------
# Strength helper
# ---------------------------------------------------------------------------

def _strength(ai_prob: float, bonuses: List[str], is_dir: bool) -> float:
    """Compute [0.05, 0.90] confidence score.

    Capped at 0.90 (not 0.95 or 0.99) so that the most generous
    downstream formula (base + coef * strength) stays below 1.0.
    """
    s = float(ai_prob)
    for b in bonuses:
        s += _BONUS_WEIGHTS.get(b, 0.0)
    if is_dir:
        s += 0.03
    return min(0.90, max(0.05, s))


# ---------------------------------------------------------------------------
# Per-task scoring formulas
# Worst-case maximums are annotated so they are easy to audit:
#   delete max = base + coef * 0.90
# All maxima are strictly below _HI = 0.95.
# ---------------------------------------------------------------------------

def _score_precision(action: str, ai_prob: float, bonuses: List[str], is_dir: bool) -> float:
    st = _strength(ai_prob, bonuses, is_dir)
    if action == "delete":
        raw = 0.58 + 0.34 * st      # max: 0.58 + 0.306 = 0.886
    elif action == "flag":
        raw = 0.38 + 0.18 * st      # max: 0.38 + 0.162 = 0.542
    else:                            # skip
        raw = max(0.10, 0.42 - 0.75 * st)
    return _safe_score(raw)


def _score_recall(action: str, ai_prob: float, bonuses: List[str], is_dir: bool) -> float:
    st = _strength(ai_prob, bonuses, is_dir)
    if action == "delete":
        raw = 0.52 + 0.40 * st      # max: 0.52 + 0.360 = 0.880
    elif action == "flag":
        raw = 0.33 + 0.28 * st      # max: 0.33 + 0.252 = 0.582
    else:                            # skip
        raw = max(0.07, 0.22 - 0.85 * st)
    return _safe_score(raw)


def _score_f1(action: str, ai_prob: float, bonuses: List[str], is_dir: bool) -> float:
    st = _strength(ai_prob, bonuses, is_dir)
    if action == "delete":
        raw = 0.54 + 0.36 * st      # max: 0.54 + 0.324 = 0.864
    elif action == "flag":
        raw = 0.43 + 0.24 * st      # max: 0.43 + 0.216 = 0.646
    else:                            # skip
        raw = max(0.10, 0.36 - 0.80 * st)
    return _safe_score(raw)


def _score_efficiency(action: str, ai_prob: float, bonuses: List[str], is_dir: bool, size: int) -> float:
    st = _strength(ai_prob, bonuses, is_dir)
    # size_bonus is capped at 0.15 so delete max = 0.54 + 0.22*0.90 + 0.15 = 0.888
    size_bonus = min(float(size) / 100_000_000.0, 0.15)
    if action == "delete":
        raw = 0.54 + 0.22 * st + size_bonus
    elif action == "flag":
        raw = 0.33 + 0.16 * st + size_bonus * 0.30
    else:                            # skip
        raw = max(0.07, 0.22 - size_bonus * 0.60)
    return _safe_score(raw)


# ---------------------------------------------------------------------------
# Public API: grade_action  (called by grader_* and by inference.py)
# ---------------------------------------------------------------------------

def grade_action(task_id: str, action: str, signals: dict) -> float:
    """Score a single agent action.  Always returns float in (0.05, 0.95)."""
    action = (action or "").lower().strip()
    for a in ("delete", "flag", "skip"):
        if a in action:
            action = a
            break
    else:
        action = "skip"   # safe default for unrecognised strings

    signals = signals or {}
    ai_prob  = float(signals.get("ai_probability", 0.10) or 0.10)
    is_dir   = signals.get("file_kind") == "directory"
    size     = int(signals.get("size_bytes", 0) or 0)

    # Rebuild bonus list from signal dict keys
    bonuses = [k for k in _BONUS_WEIGHTS if signals.get(k)]

    if task_id == "precision":
        return _score_precision(action, ai_prob, bonuses, is_dir)
    if task_id == "recall":
        return _score_recall(action, ai_prob, bonuses, is_dir)
    if task_id == "f1_score":
        return _score_f1(action, ai_prob, bonuses, is_dir)
    if task_id == "efficiency":
        return _score_efficiency(action, ai_prob, bonuses, is_dir, size)

    return _safe_score(0.50)   # unknown task → neutral score


# ---------------------------------------------------------------------------
# Public API: per-task grader functions
# ---------------------------------------------------------------------------

def _run_task(task_id: str, base_url: str = "http://localhost:8000") -> float:
    """Compute oracle score for *task_id* using its ideal actions."""
    steps = _TASK_STEPS.get(task_id, [])
    if not steps:
        return _safe_score(0.50)

    total = 0.0
    for step in steps:
        score = grade_action(
            task_id,
            step["ideal"],
            {
                "ai_probability":   step["ai_prob"],
                "file_kind":        "directory" if step["is_dir"] else "file",
                "size_bytes":       step["size"],
                **{b: True for b in step["bonuses"]},
            },
        )
        total += score

    return _safe_score(total / len(steps))


def grader_precision(base_url: str = "http://localhost:8000") -> float:
    """Oracle precision score.  Strictly in (0, 1)."""
    return _run_task("precision", base_url)


def grader_recall(base_url: str = "http://localhost:8000") -> float:
    """Oracle recall score.  Strictly in (0, 1)."""
    return _run_task("recall", base_url)


def grader_f1_score(base_url: str = "http://localhost:8000") -> float:
    """Oracle F1 score.  Strictly in (0, 1)."""
    return _run_task("f1_score", base_url)


def grader_efficiency(base_url: str = "http://localhost:8000") -> float:
    """Oracle efficiency score.  Strictly in (0, 1)."""
    return _run_task("efficiency", base_url)


def run_all_graders(base_url: str = "http://localhost:8000") -> Dict[str, float]:
    """Return all four task scores as a dict."""
    return {
        "precision":  grader_precision(base_url),
        "recall":     grader_recall(base_url),
        "f1_score":   grader_f1_score(base_url),
        "efficiency": grader_efficiency(base_url),
    }


# ---------------------------------------------------------------------------
# Self-test — run with:  python tasks/graders.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    results = run_all_graders()
    ok = True
    for name, score in results.items():
        in_range = 0.0 < score < 1.0
        status = "PASS" if in_range else "FAIL"
        print(f"  {status}  {name}: {score}")
        if not in_range:
            ok = False

    # Also test every combination exhaustively
    all_actions = ["delete", "flag", "skip"]
    fail_count = 0
    for tid, steps in _TASK_STEPS.items():
        for i, step in enumerate(steps):
            for act in all_actions:
                signals = {
                    "ai_probability": step["ai_prob"],
                    "file_kind": "directory" if step["is_dir"] else "file",
                    "size_bytes": step["size"],
                    **{b: True for b in step["bonuses"]},
                }
                s = grade_action(tid, act, signals)
                if not (0.0 < s < 1.0):
                    print(f"  FAIL  grade_action({tid!r},{act!r}) = {s!r}")
                    fail_count += 1

    if fail_count == 0 and ok:
        print(f"\n  All scores verified strictly in (0, 1). Ready to submit.")
        sys.exit(0)
    else:
        print(f"\n  {fail_count} score(s) out of range!")
        sys.exit(1)
