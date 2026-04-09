"""Root task entrypoint for OpenEnv graders.

The validator imports this file and calls grader_* functions directly.
Each must return a single float strictly in the open interval (0, 1).

The inner server graders return Tuple[float, Dict] — we unwrap here.
"""

try:
    from my_env.server.tasks import (
        grader_efficiency as _grader_efficiency,
        grader_f1_score as _grader_f1_score,
        grader_precision as _grader_precision,
        grader_recall as _grader_recall,
        run_all_graders,
    )
except ModuleNotFoundError:
    from server.tasks import (
        grader_efficiency as _grader_efficiency,
        grader_f1_score as _grader_f1_score,
        grader_precision as _grader_precision,
        grader_recall as _grader_recall,
        run_all_graders,
    )


def grader_precision(base_url: str = "http://localhost:8000") -> float:
    """Task 1 — Precision grader. Returns float strictly in (0, 1)."""
    score, _ = _grader_precision(base_url)
    return float(score)


def grader_recall(base_url: str = "http://localhost:8000") -> float:
    """Task 2 — Recall grader. Returns float strictly in (0, 1)."""
    score, _ = _grader_recall(base_url)
    return float(score)


def grader_f1_score(base_url: str = "http://localhost:8000") -> float:
    """Task 3 — F1-score grader. Returns float strictly in (0, 1)."""
    score, _ = _grader_f1_score(base_url)
    return float(score)


def grader_efficiency(base_url: str = "http://localhost:8000") -> float:
    """Task 4 — Efficiency grader. Returns float strictly in (0, 1)."""
    score, _ = _grader_efficiency(base_url)
    return float(score)


__all__ = [
    "grader_precision",
    "grader_recall",
    "grader_f1_score",
    "grader_efficiency",
    "run_all_graders",
]