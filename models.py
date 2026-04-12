"""Pydantic models for the AI Bloat Detector environment."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BloatAction(BaseModel):
    """Action submitted by the agent at each step."""
    action_type: str = Field(
        default="skip",
        description="One of: 'delete', 'flag', 'skip', or 'done'. Defaults to 'skip'.",
    )


class FileSignal(BaseModel):
    """A single forensic signal attached to a file."""
    signal_type: str
    description: str
    confidence: float = 0.5


class FileFingerprint(BaseModel):
    """Full forensic fingerprint for one file/directory."""
    path: str
    file_kind: str = "file"          # 'file' or 'directory'
    size_bytes: int = 0
    ai_probability: float = 0.5
    signals: List[FileSignal] = Field(default_factory=list)
    is_bloat: bool = False           # ground-truth label (hidden from agent)


class BloatObservation(BaseModel):
    """Observation returned after each step."""
    # Current item to triage (None when episode is done)
    current_file: Optional[FileFingerprint] = None
    observation_text: str = ""

    # Episode bookkeeping
    step_count: int = 0
    queue_remaining: int = 0
    bytes_freed: int = 0

    # Outcome of the last action
    last_action: Optional[str] = None
    last_reward: float = 0.0
    last_result: str = ""

    # Running metrics
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    # Terminal flag
    done: bool = False
    episode_summary: Optional[Dict[str, Any]] = None
