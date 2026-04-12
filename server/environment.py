"""
AI Bloat Detector -- RL Environment.

The agent receives FileFingerprint observations and must decide:
  delete  -> remove AI-generated bloat
  flag    -> mark uncertain file for human review
  skip    -> preserve human-authored file
  done    -> end episode early

Reward mapping (all values strictly in 0.0 < r < 1.0):
  delete  + bloat  -> 0.90   correct removal
  delete  + human  -> 0.05   false positive (heavy penalty)
  flag    + bloat  -> 0.60   partial credit
  flag    + human  -> 0.30   uncertain but safe
  skip    + human  -> 0.80   correct preservation
  skip    + bloat  -> 0.10   missed bloat (penalty)
"""

from __future__ import annotations

import random
import uuid
from typing import List, Tuple

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:
    # Fallback base classes so the file is importable without openenv installed
    class Environment:  # type: ignore[no-redef]
        pass
    class State(dict):  # type: ignore[no-redef]
        pass

try:
    from ..models import BloatAction, BloatObservation, BloatState, FileFingerprint, FileSignal
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from models import BloatAction, BloatObservation, BloatState, FileFingerprint, FileSignal


# ---------------------------------------------------------------------------
# Reward table -- every value is strictly in (0.0, 1.0)
# ---------------------------------------------------------------------------
REWARD = {
    ("delete", True):  0.90,  # correctly deleted bloat
    ("delete", False): 0.05,  # deleted a human file (bad)
    ("flag",   True):  0.60,  # flagged bloat for review
    ("flag",   False): 0.30,  # flagged human file (cautious but ok)
    ("skip",   False): 0.80,  # correctly skipped human file
    ("skip",   True):  0.10,  # missed obvious bloat
}
DEFAULT_REWARD = 0.50          # for 'done' action or unknown


# ---------------------------------------------------------------------------
# Synthetic workspace generator
# ---------------------------------------------------------------------------

# (path_pattern, is_bloat, ai_probability, size_bytes, signals)
_WORKSPACE_TEMPLATES: List[Tuple[str, bool, float, int, List[str]]] = [
    # High-confidence bloat
    (".cursor/rules.json",            True,  0.97, 4096,     ["hidden_artifact_dir"]),
    ("__pycache__/main.cpython-311.pyc", True, 0.99, 2048,   ["bytecode_artifact"]),
    ("node_modules/lodash/lodash.js", True,  0.92, 524288,   ["dependency_bloat"]),
    ("dist/bundle.min.js",            True,  0.88, 204800,   ["build_output"]),
    ("venv/pyvenv.cfg",               True,  0.95, 128,      ["virtualenv_internal"]),
    (".claude/session.json",          True,  0.96, 8192,     ["hidden_artifact_dir"]),
    ("temp_draft_v2_copy.py",         True,  0.84, 1024,     ["temp_draft", "duplicate_content"]),
    ("src/helpers_generated.py",      True,  0.82, 6144,     ["ai_scaffold_name", "batch_creation"]),
    ("assets/icon_hidden.png",        True,  0.91, 16384,    ["type_mismatch"]),
    ("node_modules/",                 True,  0.95, 78643200, ["dependency_bloat"]),

    # Human files -- should be kept
    ("README.md",                     False, 0.08, 3200,  []),
    ("src/main.py",                   False, 0.12, 8192,  []),
    ("tests/test_core.py",            False, 0.09, 5120,  []),
    ("requirements.txt",              False, 0.11, 512,   []),
    ("docs/architecture.md",          False, 0.07, 12288, []),
    (".github/workflows/ci.yml",      False, 0.15, 2048,  []),
    ("pyproject.toml",                False, 0.10, 1024,  []),
    ("LICENSE",                       False, 0.06, 1068,  []),

    # Marginal / uncertain (should flag)
    ("config/settings.json",          False, 0.45, 2048, []),
    ("scripts/setup.sh",              False, 0.38, 1536, []),
]


def _build_fingerprint(template: Tuple) -> FileFingerprint:
    path, is_bloat, ai_prob, size, signal_names = template
    signals = [
        FileSignal(
            signal_type=s.upper(),
            description=f"Detected: {s.replace('_', ' ')}",
            confidence=round(ai_prob - 0.05, 2),
        )
        for s in signal_names
    ]
    return FileFingerprint(
        path=path,
        file_kind="directory" if path.endswith("/") else "file",
        size_bytes=size,
        ai_probability=ai_prob,
        signals=signals,
        is_bloat=is_bloat,
    )


def _make_observation_text(fp: FileFingerprint, step: int, remaining: int) -> str:
    signal_str = ", ".join(s.signal_type for s in fp.signals) or "none"
    return (
        f"Step {step} -- {remaining} files remaining\n"
        f"PATH         : {fp.path}\n"
        f"FILE_KIND    : {fp.file_kind}\n"
        f"SIZE_BYTES   : {fp.size_bytes:,}\n"
        f"AI_PROB      : {fp.ai_probability:.2f}\n"
        f"SIGNALS      : {signal_str}\n"
        f"Decide: DELETE (remove bloat), FLAG (uncertain), or SKIP (keep human file)."
    )


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class AiBloatDetectorEnvironment(Environment):
    """OpenEnv-compatible AI bloat detection environment."""

    def __init__(self) -> None:
        super().__init__()
        self._queue: List[FileFingerprint] = []
        self._index: int = 0
        self._step: int = 0
        self._bytes_freed: int = 0
        self._tp = self._fp = self._tn = self._fn = 0
        self._episode_id: str = ""
        self._last_obs: BloatObservation = BloatObservation(done=True)

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> BloatObservation:
        self._episode_id = str(uuid.uuid4())
        self._queue = [_build_fingerprint(t) for t in _WORKSPACE_TEMPLATES]
        random.shuffle(self._queue)
        self._index = 0
        self._step = 0
        self._bytes_freed = 0
        self._tp = self._fp = self._tn = self._fn = 0

        first = self._queue[0]
        obs = BloatObservation(
            current_file=first,
            observation_text=_make_observation_text(first, 0, len(self._queue)),
            step_count=0,
            queue_remaining=len(self._queue),
            reward=0.0,
            done=False,
        )
        self._last_obs = obs
        return obs

    def step(self, action: BloatAction) -> BloatObservation:
        act = (action.action_type or "").lower().strip()

        # Episode done
        if act == "done" or self._index >= len(self._queue):
            return self._terminal()

        fp = self._queue[self._index]
        self._index += 1
        self._step += 1

        # Compute reward -- always in (0.0, 1.0)
        reward = REWARD.get((act, fp.is_bloat), DEFAULT_REWARD)

        # Update metrics
        if act == "delete":
            if fp.is_bloat:
                self._tp += 1
                self._bytes_freed += fp.size_bytes
                result = f"CORRECT DELETE (+{reward}): '{fp.path}' was AI bloat."
            else:
                self._fp += 1
                result = f"FALSE POSITIVE ({reward}): '{fp.path}' was a human file!"
        elif act == "flag":
            if fp.is_bloat:
                self._fn += 1   # identified but not removed
                result = f"FLAGGED ({reward}): '{fp.path}' marked for review."
            else:
                result = f"FLAGGED HUMAN ({reward}): '{fp.path}' -- cautious choice."
        elif act == "skip":
            if not fp.is_bloat:
                self._tn += 1
                result = f"CORRECT SKIP (+{reward}): '{fp.path}' is a human file."
            else:
                self._fn += 1
                result = f"MISSED BLOAT ({reward}): '{fp.path}' was AI bloat!"
        else:
            reward = DEFAULT_REWARD
            result = f"UNKNOWN action '{act}' -- treated as skip."
            if not fp.is_bloat:
                self._tn += 1
            else:
                self._fn += 1

        # Next observation
        if self._index >= len(self._queue):
            return self._terminal(last_reward=reward, last_result=result)

        next_fp = self._queue[self._index]
        remaining = len(self._queue) - self._index
        obs = BloatObservation(
            current_file=next_fp,
            observation_text=_make_observation_text(next_fp, self._step, remaining),
            step_count=self._step,
            queue_remaining=remaining,
            bytes_freed=self._bytes_freed,
            last_action=act,
            last_reward=reward,
            last_result=result,
            reward=reward,
            true_positives=self._tp,
            false_positives=self._fp,
            true_negatives=self._tn,
            false_negatives=self._fn,
            done=False,
        )
        self._last_obs = obs
        return obs

    @property
    def state(self) -> BloatState:
        return BloatState(
            episode_id=self._episode_id,
            step_count=self._step,
            queue_remaining=max(0, len(self._queue) - self._index),
            bytes_freed=self._bytes_freed,
            true_positives=self._tp,
            false_positives=self._fp,
            true_negatives=self._tn,
            false_negatives=self._fn,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precision(self) -> float:
        denom = self._tp + self._fp
        return self._tp / denom if denom else 0.0

    def _recall(self) -> float:
        denom = self._tp + self._fn
        return self._tp / denom if denom else 0.0

    def _f1(self) -> float:
        p, r = self._precision(), self._recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def _terminal(
        self,
        last_reward: float = DEFAULT_REWARD,
        last_result: str = "Episode complete.",
    ) -> BloatObservation:
        summary = {
            "episode_id": self._episode_id,
            "steps": self._step,
            "bytes_freed": self._bytes_freed,
            "true_positives": self._tp,
            "false_positives": self._fp,
            "true_negatives": self._tn,
            "false_negatives": self._fn,
            "precision": round(self._precision(), 4),
            "recall": round(self._recall(), 4),
            "f1_score": round(self._f1(), 4),
        }
        obs = BloatObservation(
            current_file=None,
            observation_text=(
                f"Episode complete after {self._step} steps. "
                f"Bytes freed: {self._bytes_freed:,}. "
                f"F1: {summary['f1_score']:.3f}."
            ),
            step_count=self._step,
            queue_remaining=0,
            bytes_freed=self._bytes_freed,
            last_action=None,
            last_reward=last_reward,
            last_result=last_result,
            reward=last_reward,
            true_positives=self._tp,
            false_positives=self._fp,
            true_negatives=self._tn,
            false_negatives=self._fn,
            done=True,
            episode_summary=summary,
        )
        self._last_obs = obs
        return obs
