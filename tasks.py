"""Root task entrypoint for OpenEnv graders.

The validator imports this file and calls grader_* functions directly.
Each must return a single float strictly in the open interval (0, 1).
"""

import importlib
import os
import sys


def _import_graders():
    """Try multiple import strategies to find the graders module."""
    # Strategy 1: absolute import (works when package is installed)
    try:
        return importlib.import_module("tasks.graders")
    except (ModuleNotFoundError, ImportError):
        pass

    # Strategy 2: add project root to path then retry
    _root = os.path.dirname(os.path.abspath(__file__))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        return importlib.import_module("tasks.graders")
    except (ModuleNotFoundError, ImportError):
        pass

    raise ImportError("Could not import tasks.graders from any path strategy.")


_g = _import_graders()

grader_precision = _g.grader_precision
grader_recall = _g.grader_recall
grader_f1_score = _g.grader_f1_score
grader_efficiency = _g.grader_efficiency
run_all_graders = _g.run_all_graders


__all__ = [
    "grader_precision",
    "grader_recall",
    "grader_f1_score",
    "grader_efficiency",
    "run_all_graders",
]