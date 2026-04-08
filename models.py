

"""
Data models for the AI Digital Bloat Detector Environment.

Action Space
------------
BloatAction.action_type:
    "delete" -> Remove this item (high-confidence AI bloat)
    "flag"   -> Mark for human review (uncertain)
    "skip"   -> Keep this item (human-created)
    "done"   -> End the episode early

Observation Space
-----------------
BloatObservation provides a full FileFingerprint for the current queue item,
running precision/recall/F1, cumulative bytes freed, and an episode summary
when done=True.
"""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class BloatAction(Action):
    """Action to classify the current item in the bloat detection queue."""

    action_type: Literal["delete", "flag", "skip", "done"] = Field(
        ...,
        description=(
            "'delete': Permanently remove the item (high-confidence AI bloat). "
            "'flag': Mark for manual review (uncertain). "
            "'skip': Keep the item (human-created). "
            "'done': Terminate the episode early."
        ),
    )


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class AISignal(BaseModel):
    """A single forensic signal indicating AI-generated content."""

    signal_type: str = Field(
        ...,
        description=(
            "One of: HIDDEN_ARTIFACT_DIR, DEPENDENCY_BLOAT, BUILD_CACHE, "
            "BATCH_CREATION, DUPLICATE_CONTENT, TEMP_DRAFT, "
            "BYTECODE_ARTIFACT, VIRTUALENV_INTERNAL, TYPE_MISMATCH, "
            "AI_SCAFFOLD_NAME"
        ),
    )
    description: str = Field(..., description="Human-readable explanation")
    confidence: float = Field(..., ge=0.0, le=1.0)


class FileFingerprint(BaseModel):
    """Complete forensic fingerprint of a file or directory."""

    # Identity
    path: str = Field(..., description="Path relative to workspace root")
    is_directory: bool = Field(default=False)

    # Physical attributes
    size_bytes: int = Field(default=0)
    child_count: int = Field(default=0, description="Direct children (dirs only)")
    extension: str = Field(default="", description="Lowercase extension e.g. '.py'")

    # Timestamps (Unix epoch floats)
    ctime: float = Field(..., description="inode-change timestamp")
    mtime: float = Field(..., description="Last-modified timestamp")
    atime: float = Field(..., description="Last-accessed timestamp")

    # Deep content (files only)
    sha256_hash: str = Field(default="", description="SHA-256 hex digest")
    magic_header: str = Field(default="", description="First 16 bytes in hex")
    declared_type: str = Field(default="", description="Type from extension")
    detected_type: str = Field(default="", description="Type from magic bytes")
    type_mismatch: bool = Field(default=False, description="Extension vs magic mismatch")

    # AI Fingerprints
    ai_signals: List[AISignal] = Field(default_factory=list)
    ai_probability: float = Field(default=0.05, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class BloatObservation(Observation):
    """Observation returned after each environment step."""

    current_item: Optional[FileFingerprint] = Field(
        default=None,
        description="Fingerprint of the item to classify. None when queue is exhausted.",
    )
    queue_size: int = Field(default=0)
    step_count: int = Field(default=0)

    # Cumulative stats
    bytes_freed: int = Field(default=0)
    true_positives: int = Field(default=0)
    false_positives: int = Field(default=0)
    true_negatives: int = Field(default=0)
    false_negatives: int = Field(default=0)

    # Running metrics
    precision: float = Field(default=0.0)
    recall: float = Field(default=0.0)
    f1_score: float = Field(default=0.0)

    last_action_result: str = Field(default="")
    episode_summary: Optional[Dict] = Field(default=None)
