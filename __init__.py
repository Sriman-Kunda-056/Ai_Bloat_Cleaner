# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AI Digital Bloat Detector Environment."""

from .client import AiBloatDetector
from .models import AISignal, BloatAction, BloatObservation, FileFingerprint

try:
    from .tasks import (
        grader_efficiency,
        grader_f1_score,
        grader_precision,
        grader_recall,
        run_all_graders,
    )
except Exception:  # pragma: no cover
    grader_efficiency = None
    grader_f1_score = None
    grader_precision = None
    grader_recall = None
    run_all_graders = None

__all__ = [
    "AiBloatDetector",
    "BloatAction",
    "BloatObservation",
    "FileFingerprint",
    "AISignal",
    "grader_precision",
    "grader_recall",
    "grader_f1_score",
    "grader_efficiency",
    "run_all_graders",
]
