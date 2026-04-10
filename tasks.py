"""Root task entrypoint for OpenEnv graders.

The validator imports this file and calls grader_* functions directly.
Each must return a single float strictly in the open interval (0, 1).
"""

try:
    from tasks.graders import (
        grader_efficiency,
        grader_f1_score,
        grader_precision,
        grader_recall,
        run_all_graders,
    )
except ModuleNotFoundError:
    from tasks.graders import (
        grader_efficiency,
        grader_f1_score,
        grader_precision,
        grader_recall,
        run_all_graders,
    )


__all__ = [
    "grader_precision",
    "grader_recall",
    "grader_f1_score",
    "grader_efficiency",
    "run_all_graders",
]