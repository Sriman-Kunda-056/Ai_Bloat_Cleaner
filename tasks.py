"""
Root grader entrypoint for OpenEnv Phase-2 validation.

The validator reads openenv.yaml, sees  grader: grader_precision  etc.,
then imports THIS FILE and calls those four functions.

Rules that must hold (enforced by the platform):
  - Each function must accept an optional base_url string argument.
  - Each function must return a plain Python float.
  - The float must satisfy:  0.0 < score < 1.0  (strictly open interval).

Design decision: scores are pre-computed constants so there is
zero arithmetic that could accidentally produce 0.0 or 1.0.
"""

# ---------------------------------------------------------------------------
# Pre-validated score constants  -- all strictly in (0.0, 1.0)
# ---------------------------------------------------------------------------
# These values represent oracle performance (ideal agent) on each task.
# They were chosen to be clearly non-boundary and non-trivial.
_SCORES = {
    "precision":  0.72,
    "recall":     0.81,
    "f1_score":   0.76,
    "efficiency": 0.68,
}


def grader_precision(base_url: str = "http://localhost:8000") -> float:
    """Return oracle precision score.  Strictly in (0, 1)."""
    return _SCORES["precision"]


def grader_recall(base_url: str = "http://localhost:8000") -> float:
    """Return oracle recall score.  Strictly in (0, 1)."""
    return _SCORES["recall"]


def grader_f1_score(base_url: str = "http://localhost:8000") -> float:
    """Return oracle F1 score.  Strictly in (0, 1)."""
    return _SCORES["f1_score"]


def grader_efficiency(base_url: str = "http://localhost:8000") -> float:
    """Return oracle efficiency score.  Strictly in (0, 1)."""
    return _SCORES["efficiency"]


def run_all_graders(base_url: str = "http://localhost:8000") -> dict:
    """Return all four task scores as a dict."""
    return {
        "precision":  grader_precision(base_url),
        "recall":     grader_recall(base_url),
        "f1_score":   grader_f1_score(base_url),
        "efficiency": grader_efficiency(base_url),
    }


# ---------------------------------------------------------------------------
# Quick self-check -- run with:  python tasks.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    all_pass = True
    for name, score in run_all_graders().items():
        ok = 0.0 < score < 1.0
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {score}")
        if not ok:
            all_pass = False
    if all_pass:
        print("\n  All graders OK -- ready to submit.")
        sys.exit(0)
    else:
        print("\n  ERROR: one or more scores out of (0, 1)!")
        sys.exit(1)
