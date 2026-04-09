"""
Isolated reward logic for the AI Digital Bloat Detector environment.

All reward calculations live here — triage_env.py must import from this
module and must NOT contain any raw reward numbers.

Reward constants (ALL_CAPS, never inline magic numbers):
    DELETE on bloat:           +SIZE_MB * DELETE_CORRECT_PER_MB
    DELETE on critical file:   CRITICAL_DELETE_PENALTY  (truncates episode)
    DELETE on human file:      HUMAN_DELETE_PENALTY
    CONSOLIDATE on cluster:    CONSOLIDATE_CLUSTER_BONUS
    CONSOLIDATE premature:     PREMATURE_CONSOLIDATE_PENALTY
    CONSOLIDATE non-duplicate: NON_DUPLICATE_CONSOLIDATE_PENALTY
    COMPRESS on large file:    COMPRESS_BONUS_PER_MB * size_mb
    COMPRESS already small:    COMPRESS_SMALL_PENALTY
    KEEP bloat (obvious):      KEEP_OBVIOUS_BLOAT_PENALTY
    KEEP human file:           KEEP_HUMAN_REWARD
"""

from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Reward constants — modify here, never inside triage_env.py
# ---------------------------------------------------------------------------

DELETE_CORRECT_PER_MB: float     = 0.10   # reward per MB of correctly deleted bloat
CRITICAL_DELETE_PENALTY: float   = -100.0 # integrity violation — also truncates episode
HUMAN_DELETE_PENALTY: float      = -5.0   # deleted a human-authored file

CONSOLIDATE_CLUSTER_BONUS: float      = 3.0   # reward for valid consolidation
PREMATURE_CONSOLIDATE_PENALTY: float  = -5.0  # consolidate before KEEP-ing a cluster member
NON_DUPLICATE_CONSOLIDATE_PENALTY: float = -2.0  # consolidate on a unique-hash file

COMPRESS_BONUS_PER_MB: float  = 0.05  # reward per MB of large file compressed
COMPRESS_SMALL_THRESHOLD_MB: float = 0.1  # files under this are "already small"
COMPRESS_SMALL_PENALTY: float = -1.0  # compressing a tiny file wastes cycles

KEEP_OBVIOUS_BLOAT_PENALTY: float = -0.5  # keep when ai_signature > OBVIOUS_BLOAT_THRESHOLD
KEEP_HUMAN_REWARD: float          = +0.3  # correctly kept a human file
OBVIOUS_BLOAT_THRESHOLD: float    = 0.80  # ai_signature above this = "obvious bloat"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_reward(
    action: int,
    file_meta: Dict,
    kept_hash_groups: set,
) -> Tuple[float, bool, str]:
    """
    Compute the reward for a single agent action on a single file.

    Args:
        action:           Integer action (0=KEEP, 1=DELETE, 2=CONSOLIDATE, 3=COMPRESS).
        file_meta:        File metadata dict from SimulatedFileSystem.files[i].
        kept_hash_groups: Set of content_hash_group strings for all files the
                          agent has previously KEEP-ed. Required to validate
                          CONSOLIDATE eligibility (multi-step reasoning check).

    Returns:
        Tuple of (reward: float, integrity_broken: bool, feedback: str).
        integrity_broken=True means the episode should be truncated.
    """
    # from env.file_system import (  # local import to avoid circular refs
    #     ACTION_KEEP, ACTION_DELETE, ACTION_CONSOLIDATE, ACTION_COMPRESS,
    # )

    ACTION_KEEP = 0
    ACTION_DELETE = 1
    ACTION_CONSOLIDATE = 2
    ACTION_COMPRESS = 3

    label       = file_meta["label"]           # 'bloat' or 'human'
    is_critical = file_meta["is_critical"]
    size_mb     = file_meta["size_mb"]
    ai_sig      = file_meta["ai_signature"]
    hash_group  = file_meta["content_hash_group"]

    # ── DELETE ──────────────────────────────────────────────────────────────
    if action == ACTION_DELETE:
        if is_critical:
            return (
                CRITICAL_DELETE_PENALTY,
                True,
                f"INTEGRITY VIOLATION: deleted critical file '{file_meta['path']}' "
                f"(reward {CRITICAL_DELETE_PENALTY:.0f}, episode truncated)",
            )
        if label == "bloat":
            reward = size_mb * DELETE_CORRECT_PER_MB
            return (
                reward,
                False,
                f"CORRECT DELETE (+{reward:.3f}): '{file_meta['path']}' "
                f"({size_mb:.2f} MB of bloat removed)",
            )
        # human file, non-critical
        return (
            HUMAN_DELETE_PENALTY,
            False,
            f"WRONG DELETE ({HUMAN_DELETE_PENALTY:.1f}): '{file_meta['path']}' "
            "is a human-authored file.",
        )

    # ── CONSOLIDATE ─────────────────────────────────────────────────────────
    if action == ACTION_CONSOLIDATE:
        if hash_group == "unique":
            return (
                NON_DUPLICATE_CONSOLIDATE_PENALTY,
                False,
                f"INVALID CONSOLIDATE ({NON_DUPLICATE_CONSOLIDATE_PENALTY:.1f}): "
                f"'{file_meta['path']}' has no duplicate cluster.",
            )
        # Multi-step reasoning gate: agent must KEEP one cluster member first
        if hash_group not in kept_hash_groups:
            return (
                PREMATURE_CONSOLIDATE_PENALTY,
                False,
                f"PREMATURE CONSOLIDATE ({PREMATURE_CONSOLIDATE_PENALTY:.1f}): "
                f"Must KEEP at least one file from cluster '{hash_group}' first.",
            )
        return (
            CONSOLIDATE_CLUSTER_BONUS,
            False,
            f"CLUSTER CONSOLIDATE (+{CONSOLIDATE_CLUSTER_BONUS:.1f}): "
            f"'{file_meta['path']}' merged into cluster '{hash_group}'.",
        )

    # ── COMPRESS ────────────────────────────────────────────────────────────
    if action == ACTION_COMPRESS:
        if size_mb < COMPRESS_SMALL_THRESHOLD_MB:
            return (
                COMPRESS_SMALL_PENALTY,
                False,
                f"WASTEFUL COMPRESS ({COMPRESS_SMALL_PENALTY:.1f}): "
                f"'{file_meta['path']}' is only {size_mb*1024:.1f} KB — too small.",
            )
        reward = size_mb * COMPRESS_BONUS_PER_MB
        return (
            reward,
            False,
            f"COMPRESS (+{reward:.3f}): '{file_meta['path']}' "
            f"({size_mb:.2f} MB compressed).",
        )

    # ── KEEP (default) ──────────────────────────────────────────────────────
    if action == ACTION_KEEP:
        if label == "human":
            return (
                KEEP_HUMAN_REWARD,
                False,
                f"CORRECT KEEP (+{KEEP_HUMAN_REWARD:.2f}): '{file_meta['path']}' "
                "is a human file — preserved.",
            )
        if ai_sig > OBVIOUS_BLOAT_THRESHOLD:
            return (
                KEEP_OBVIOUS_BLOAT_PENALTY,
                False,
                f"OBVIOUS BLOAT KEPT ({KEEP_OBVIOUS_BLOAT_PENALTY:.2f}): "
                f"'{file_meta['path']}' has ai_signature={ai_sig:.2f} — should be deleted.",
            )
        # Marginal bloat — no penalty, no reward
        return (
            0.0,
            False,
            f"KEEP (0.0): '{file_meta['path']}' — marginal confidence, no action.",
        )

    # Unknown action (should never reach here if action_space is correct)
    return (0.0, False, f"UNKNOWN action {action}")
