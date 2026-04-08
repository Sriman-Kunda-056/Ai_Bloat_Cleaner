# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task Graders for AI Digital Bloat Detector.

Defines 3+ evaluation tasks with grader functions that assess agent performance.
Each grader returns a score in [0.0, 1.0] range.

Key improvement over naive threshold approach:
  - composite_score() combines ai_probability + signal evidence + metadata heuristics
  - Each task tunes its own delete/flag/skip thresholds on composite_score
  - Human-file guard prevents false positives on files with old mtimes or tiny sizes
"""

import asyncio
import time
from typing import Dict, Optional, Tuple

try:
    from my_env import AiBloatDetector, BloatAction
    from my_env.models import FileFingerprint
except ModuleNotFoundError:
    from client import AiBloatDetector
    from models import BloatAction, FileFingerprint


# ---------------------------------------------------------------------------
# Signal weight table — tuned to the synthetic workspace described in README
# ---------------------------------------------------------------------------

# High-confidence bloat signals: presence alone is near-decisive
HIGH_WEIGHT_SIGNALS = {
    "HIDDEN_ARTIFACT_DIR",   # .cursorrules, .claude/, .cursor/
    "DEPENDENCY_BLOAT",      # node_modules/, venv/
    "BUILD_CACHE",           # __pycache__/, .pytest_cache/
    "BYTECODE_ARTIFACT",     # .pyc
    "VIRTUALENV_INTERNAL",   # pyvenv.cfg
}

# Medium-confidence: meaningful but not sufficient on their own
MEDIUM_WEIGHT_SIGNALS = {
    "BATCH_CREATION",        # same mtime cluster
    "TEMP_DRAFT",            # temp_, draft_, _copy, .bak
    "DUPLICATE_CONTENT",     # identical SHA-256
    "TYPE_MISMATCH",         # extension vs magic mismatch
    "AI_SCAFFOLD_NAME",      # utils.py, services.py, helpers.py …
}

SIGNAL_WEIGHTS: Dict[str, float] = {
    "HIDDEN_ARTIFACT_DIR":        0.90,
    "DEPENDENCY_BLOAT":           0.90,
    "BUILD_CACHE":                0.85,
    "BYTECODE_ARTIFACT":          0.80,
    "VIRTUALENV_INTERNAL":        0.80,
    "BATCH_CREATION":             0.55,
    "TEMP_DRAFT":                 0.60,
    "DUPLICATE_CONTENT":          0.65,
    "TYPE_MISMATCH":              0.60,
    "AI_SCAFFOLD_NAME":           0.45,
}

# Files modified less than this many seconds ago are suspicious (AI agents
# batch-create files in short bursts).
RECENT_MTIME_WINDOW_SECS: float = 300.0   # 5 minutes

# Files older than this many days are very likely human-authored.
OLD_FILE_AGE_DAYS: float = 14.0

# Directories: raise delete threshold — one wrong delete can cascade.
DIRECTORY_DELETE_THRESHOLD_BUMP: float = 0.08

# Competition requires each task score to be strictly between 0 and 1.
STRICT_SCORE_EPS: float = 1e-3


def _strict_score(value: float) -> float:
    """Clamp scores to the open interval (0, 1)."""
    return float(min(1.0 - STRICT_SCORE_EPS, max(STRICT_SCORE_EPS, value)))


# ---------------------------------------------------------------------------
# Core scoring function — shared across all tasks
# ---------------------------------------------------------------------------


def composite_score(item: FileFingerprint) -> float:
    """
    Compute a blended bloat-probability score that combines:
      1. The environment's own ai_probability
      2. Weighted evidence from individual ai_signals
      3. Metadata heuristics (age, mtime recency, type mismatch)

    Returns a float in [0.0, 1.0].  Higher = more confident it is AI bloat.
    """
    base = item.ai_probability

    # ── Signal evidence ───────────────────────────────────────────────────
    # Walk each signal; accumulate a weighted vote.
    signal_boost = 0.0
    high_signal_count = 0
    for sig in item.ai_signals:
        w = SIGNAL_WEIGHTS.get(sig.signal_type, 0.30)
        # Weight further by the signal's own reported confidence
        effective = w * sig.confidence
        signal_boost += effective
        if sig.signal_type in HIGH_WEIGHT_SIGNALS and sig.confidence > 0.7:
            high_signal_count += 1

    # Normalise: first signal has full effect, diminishing returns after that
    n = max(len(item.ai_signals), 1)
    signal_contribution = signal_boost / (1 + 0.5 * (n - 1))

    # ── Metadata heuristics ──────────────────────────────────────────────
    meta_boost = 0.0

    # Type mismatch is a strong independent indicator
    if item.type_mismatch:
        meta_boost += 0.15

    # Recently created or modified files smell like AI batch output
    now = time.time()
    if item.mtime and (now - item.mtime) < RECENT_MTIME_WINDOW_SECS:
        meta_boost += 0.10
    if item.ctime and (now - item.ctime) < RECENT_MTIME_WINDOW_SECS:
        meta_boost += 0.05

    # Very old files are almost certainly human — apply a penalty
    if item.mtime:
        age_days = (now - item.mtime) / 86400.0
        if age_days > OLD_FILE_AGE_DAYS:
            meta_boost -= 0.20

    # Empty files / zero-byte artefacts are suspicious
    if item.size_bytes == 0 and not item.is_directory:
        meta_boost += 0.05

    # ── Combine ──────────────────────────────────────────────────────────
    # Weighted blend: base 50 %, signal evidence 35 %, metadata 15 %
    score = 0.50 * base + 0.35 * signal_contribution + 0.15 * meta_boost

    # If multiple high-confidence signals fire, escalate
    if high_signal_count >= 2:
        score = min(1.0, score + 0.10)

    return float(max(0.0, min(1.0, score)))


def decide(
    item: FileFingerprint,
    delete_thresh: float,
    flag_thresh: float,
) -> str:
    """
    Convert a composite_score into an action string.

    Args:
        item:          Current FileFingerprint.
        delete_thresh: Minimum composite score to choose "delete".
        flag_thresh:   Minimum composite score to choose "flag".

    Returns:
        "delete" | "flag" | "skip"
    """
    score = composite_score(item)

    # Directories: be more conservative — bump threshold slightly
    effective_delete = delete_thresh
    if item.is_directory:
        effective_delete += DIRECTORY_DELETE_THRESHOLD_BUMP

    if score >= effective_delete:
        return "delete"
    elif score >= flag_thresh:
        return "flag"
    else:
        return "skip"


# ---------------------------------------------------------------------------
# Task 1: Maximize Precision (Minimize False Positives)
# ---------------------------------------------------------------------------


def grader_precision(
    base_url: str = "http://localhost:8000",
) -> Tuple[float, Dict]:
    return asyncio.run(_grader_precision(base_url))


async def _grader_precision(
    base_url: str = "http://localhost:8000",
) -> Tuple[float, Dict]:
    """
    Task 1: Precision-focused grader.

    Strategy: only delete when composite evidence is overwhelming.
    High delete threshold + human-age guard avoids false positives.

    Score: precision in [0, 1].
    """
    env = AiBloatDetector(base_url=base_url)
    try:
        result = await env.reset()
        tp = fp = steps = 0

        while not result.observation.done and steps < 100:
            current = result.observation.current_item
            if current is None:
                break

            # Precision task: very conservative delete threshold
            action_type = decide(current, delete_thresh=0.82, flag_thresh=0.60)

            # Extra human-file guard: never delete old files even if model says so
            now = time.time()
            if action_type == "delete" and current.mtime:
                age_days = (now - current.mtime) / 86400.0
                if age_days > OLD_FILE_AGE_DAYS:
                    action_type = "flag"  # downgrade — too old to be AI batch output

            action = BloatAction(action_type=action_type)
            result = await env.step(action)
            steps += 1

            if action_type == "delete":
                if result.observation.reward > 0.5:
                    tp += 1
                else:
                    fp += 1

        summary = result.observation.episode_summary or {}
        precision = summary.get("precision", 0.0)

        score = _strict_score(precision)
        return (
            score,
            {
                "precision": precision,
                "score": score,
                "tp": tp,
                "fp": fp,
                "steps": steps,
                "f1": summary.get("f1_score", 0.0),
            },
        )
    finally:
        await env.close()


# ---------------------------------------------------------------------------
# Task 2: Maximize Recall (Minimize False Negatives)
# ---------------------------------------------------------------------------


def grader_recall(
    base_url: str = "http://localhost:8000",
) -> Tuple[float, Dict]:
    return asyncio.run(_grader_recall(base_url))


async def _grader_recall(
    base_url: str = "http://localhost:8000",
) -> Tuple[float, Dict]:
    """
    Task 2: Recall-focused grader.

    Strategy: lower thresholds to catch more AI bloat, accept higher FP rate.
    Flags anything above a low composite score so human reviewers catch the rest.

    Score: recall in [0, 1].
    """
    env = AiBloatDetector(base_url=base_url)
    try:
        result = await env.reset()
        tp = fn = steps = 0

        while not result.observation.done and steps < 100:
            current = result.observation.current_item
            if current is None:
                break

            # Recall task: aggressive — lower thresholds, flag freely
            action_type = decide(current, delete_thresh=0.55, flag_thresh=0.30)

            action = BloatAction(action_type=action_type)
            result = await env.step(action)
            steps += 1

            if action_type in ("delete", "flag"):
                if result.observation.reward > 0.2:
                    tp += 1
                else:
                    fn += 1

        summary = result.observation.episode_summary or {}
        recall = summary.get("recall", 0.0)

        score = _strict_score(recall)
        return (
            score,
            {
                "recall": recall,
                "score": score,
                "tp": tp,
                "fn": fn,
                "steps": steps,
                "f1": summary.get("f1_score", 0.0),
            },
        )
    finally:
        await env.close()


# ---------------------------------------------------------------------------
# Task 3: Maximize F1-Score (Balance Precision & Recall)
# ---------------------------------------------------------------------------


def grader_f1_score(
    base_url: str = "http://localhost:8000",
) -> Tuple[float, Dict]:
    return asyncio.run(_grader_f1_score(base_url))


async def _grader_f1_score(
    base_url: str = "http://localhost:8000",
) -> Tuple[float, Dict]:
    """
    Task 3: F1-score grader (primary evaluation metric).

    Strategy: balanced thresholds on composite_score.  The agent also
    dynamically tightens its delete threshold if running precision drops
    below 0.80, preventing a precision collapse mid-episode.

    Score: F1-score in [0, 1].
    """
    env = AiBloatDetector(base_url=base_url)
    try:
        result = await env.reset()
        steps = 0

        # Dynamic threshold adjustment
        running_tp = 0
        running_fp = 0
        delete_thresh = 0.68
        flag_thresh   = 0.38

        while not result.observation.done and steps < 100:
            current = result.observation.current_item
            if current is None:
                break

            # Adaptive threshold: tighten if precision is slipping
            if running_tp + running_fp > 5:
                running_precision = running_tp / (running_tp + running_fp)
                if running_precision < 0.75:
                    delete_thresh = min(0.82, delete_thresh + 0.02)
                elif running_precision > 0.92:
                    delete_thresh = max(0.60, delete_thresh - 0.01)

            action_type = decide(current, delete_thresh=delete_thresh, flag_thresh=flag_thresh)
            action = BloatAction(action_type=action_type)
            result = await env.step(action)
            steps += 1

            if action_type == "delete":
                if result.observation.reward > 0.5:
                    running_tp += 1
                else:
                    running_fp += 1

        summary = result.observation.episode_summary or {}
        f1 = summary.get("f1_score", 0.0)

        score = _strict_score(f1)
        return (
            score,
            {
                "f1_score": f1,
                "score": score,
                "precision": summary.get("precision", 0.0),
                "recall": summary.get("recall", 0.0),
                "steps": steps,
                "bytes_freed": summary.get("bytes_freed", 0),
                "reward_total": summary.get("reward_total", 0.0),
                "final_delete_thresh": round(delete_thresh, 3),
            },
        )
    finally:
        await env.close()


# ---------------------------------------------------------------------------
# Task 4: Efficiency (Bytes Freed / Steps)
# ---------------------------------------------------------------------------


def grader_efficiency(
    base_url: str = "http://localhost:8000",
) -> Tuple[float, Dict]:
    return asyncio.run(_grader_efficiency(base_url))


async def _grader_efficiency(
    base_url: str = "http://localhost:8000",
) -> Tuple[float, Dict]:
    """
    Task 4: Efficiency grader.

    Strategy: prioritise large high-confidence items first.
    For efficiency, we skip low-probability small files immediately
    instead of wasting a flag action on them, reducing total steps.

    Score = bytes_freed / total_AI_bytes  minus a step-budget penalty.
    """
    MAX_STEPS = 50
    env = AiBloatDetector(base_url=base_url)
    try:
        result = await env.reset()
        steps = 0
        total_ai_bytes = 0
        total_bytes_seen = 0

        while not result.observation.done and steps < MAX_STEPS:
            current = result.observation.current_item
            if current is None:
                break

            item_bytes = current.size_bytes or 0
            total_bytes_seen += item_bytes
            score = composite_score(current)

            if score > 0.60:
                total_ai_bytes += item_bytes
                action_type = "delete"
            else:
                # Efficiency task: skip directly — don't waste a flag step
                action_type = "skip"

            action = BloatAction(action_type=action_type)
            result = await env.step(action)
            steps += 1

        summary = result.observation.episode_summary or {}
        bytes_freed = summary.get("bytes_freed", 0)

        denominator = total_ai_bytes if total_ai_bytes > 0 else max(total_bytes_seen, 1)
        recall_score = min(bytes_freed / denominator, 1.0)

        half_budget = MAX_STEPS / 2
        step_penalty = max(0.0, (steps - half_budget) / half_budget) * 0.10
        efficiency_score = max(0.0, recall_score - step_penalty)

        score = _strict_score(efficiency_score)
        return (
            score,
            {
                "bytes_freed": bytes_freed,
                "total_ai_bytes": total_ai_bytes,
                "total_bytes_seen": total_bytes_seen,
                "steps": steps,
                "bytes_per_step": bytes_freed / max(steps, 1),
                "recall_score": round(recall_score, 4),
                "step_penalty": round(step_penalty, 4),
                "efficiency_score": efficiency_score,
                "score": score,
            },
        )
    finally:
        await env.close()


# ---------------------------------------------------------------------------
# Evaluation Harness
# ---------------------------------------------------------------------------


def run_all_graders(base_url: str = "http://localhost:8000") -> Dict:
    return asyncio.run(_run_all_graders(base_url))


async def _run_all_graders(base_url: str = "http://localhost:8000") -> Dict:
    """
    Run all graders sequentially and return aggregated scores.

    Returns:
        Dict with individual task results and overall_score (mean).
    """
    tasks = [
        ("task_1_precision",  _grader_precision),
        ("task_2_recall",     _grader_recall),
        ("task_3_f1",         _grader_f1_score),
        ("task_4_efficiency", _grader_efficiency),
    ]

    results: Dict = {}
    for name, grader in tasks:
        try:
            score, details = await grader(base_url)
            results[name] = {"score": _strict_score(score), "details": details}
        except Exception as e:
            # Keep score in open interval to satisfy validator constraints.
            results[name] = {"score": STRICT_SCORE_EPS, "error": str(e)}

    scores = [
        v["score"] for v in results.values()
        if isinstance(v, dict) and "score" in v
    ]
    overall = sum(scores) / len(scores) if scores else STRICT_SCORE_EPS
    results["overall_score"] = _strict_score(overall)

    return results


if __name__ == "__main__":
    import json

    print("Running all graders...")
    results = asyncio.run(_run_all_graders())
    print(json.dumps(results, indent=2))
