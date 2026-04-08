"""
Gymnasium environment for the AI Digital Bloat Detector.

SystemTriageEnv trains an RL agent to scan a simulated developer
workspace and take intelligent cleanup actions on each file.  The
agent earns rewards for correctly deleting bloat and is heavily
penalised for destroying critical human-authored files.

Observation Space
-----------------
spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

    Index 0 – size_mb          : file size, normalised to [0, 1]
    Index 1 – age_days         : days since creation, normalised
    Index 2 – access_days      : days since last access, normalised
    Index 3 – extension        : categorical encoding of file extension
    Index 4 – depth            : directory nesting depth, normalised
    Index 5 – content_hash     : 3-byte hash fingerprint, normalised
    Index 6 – ai_signature     : AI-generation probability score [0, 1]

Action Space
------------
spaces.Discrete(4)

    0 – KEEP        : Preserve the file unchanged.
    1 – DELETE      : Permanently remove the file.
    2 – CONSOLIDATE : Merge duplicates into a single canonical copy.
                      Requires KEEP-ing one cluster member first.
    3 – COMPRESS    : Apply compression to large files.

Reward Function
---------------
Defined in env/reward.py.  Key values:
    Correct DELETE   : +size_mb * 0.10
    Delete critical  : -100  (+ episode truncation)
    CONSOLIDATE dup  : +3.0
    CONSOLIDATE early: -5.0
    KEEP human       : +0.30
    KEEP obvious bloat: -0.50

Termination Conditions
----------------------
    terminated : All files in the queue have been processed.
    truncated  : Agent deletes a critical file (integrity violation).
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.file_system import (
    ACTION_COMPRESS,
    ACTION_CONSOLIDATE,
    ACTION_DELETE,
    ACTION_KEEP,
    OBS_DIM,
    SimulatedFileSystem,
)
from env.reward import compute_reward


class SystemTriageEnv(gym.Env):
    """
    Gymnasium environment that simulates AI-generated filesystem bloat
    and trains an agent to triage files intelligently.

    Args:
        render_mode: Rendering mode.  'human' prints step info to stdout.
                     None (default) disables rendering.
        seed:        Optional fixed seed for reproducible episodes.
                     Overridden by the seed passed to reset().

    Example::

        env = SystemTriageEnv(render_mode='human')
        obs, info = env.reset(seed=42)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        # ── Spaces ──────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)   # KEEP / DELETE / CONSOLIDATE / COMPRESS

        # ── Internal state ───────────────────────────────────────────────────
        self.render_mode = render_mode
        self._fs = SimulatedFileSystem(seed=seed)
        self._queue_index: int = 0
        self._kept_hash_groups: set = set()      # for multi-step CONSOLIDATE gate
        self._integrity_broken: bool = False
        self._actions_taken: list = []           # list of (file_index, action)

        # Episode-level accumulators (exposed in info dict)
        self._total_recovered_mb: float = 0.0
        self._delete_count: int = 0
        self._consolidate_count: int = 0
        self._compress_count: int = 0
        self._keep_count: int = 0
        self._false_positive_count: int = 0      # human files wrongly deleted

    # ── Core Gymnasium API ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment and return the first observation.

        Args:
            seed:    Random seed forwarded to the file system generator.
            options: Ignored; accepted for API compatibility.

        Returns:
            Tuple of (observation: np.ndarray, info: dict).
            Gymnasium ≥ 0.26 requires BOTH values — do not drop the dict.
        """
        super().reset(seed=seed)  # seeds self.np_random for reproducibility

        self._fs.reset(seed=seed)
        self._queue_index = 0
        self._kept_hash_groups = set()
        self._integrity_broken = False
        self._actions_taken = []
        self._total_recovered_mb = 0.0
        self._delete_count = 0
        self._consolidate_count = 0
        self._compress_count = 0
        self._keep_count = 0
        self._false_positive_count = 0

        obs = self._fs.get_obs_vector(self._queue_index)
        info = self._build_info(reward=0.0, feedback="Environment reset.")
        if self.render_mode == "human":
            self._render_step(info)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one agent action on the current file in the queue.

        Args:
            action: Integer in {0, 1, 2, 3} — see action space docstring.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info) per
            Gymnasium ≥ 0.26 convention.  All 5 values must be unpacked.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        file_idx  = self._queue_index
        file_meta = self._fs.files[file_idx]
        self._fs.mark_visited(file_idx)

        # ── Reward calculation (all logic in reward.py) ───────────────────
        reward, integrity_broken, feedback = compute_reward(
            action, file_meta, self._kept_hash_groups
        )

        # ── Update accumulators ───────────────────────────────────────────
        self._actions_taken.append((file_idx, action))
        self._integrity_broken = self._integrity_broken or integrity_broken

        if action == ACTION_DELETE and not integrity_broken:
            self._total_recovered_mb += file_meta["size_mb"]
            self._delete_count += 1
            if file_meta["label"] == "human":
                self._false_positive_count += 1
        elif action == ACTION_KEEP:
            self._keep_count += 1
            self._kept_hash_groups.add(file_meta["content_hash_group"])
        elif action == ACTION_CONSOLIDATE:
            self._consolidate_count += 1
        elif action == ACTION_COMPRESS:
            self._compress_count += 1

        # ── Advance queue ─────────────────────────────────────────────────
        self._queue_index += 1
        terminated = self._queue_index >= len(self._fs.files)
        truncated  = integrity_broken        # critical file deleted

        # ── Next observation ──────────────────────────────────────────────
        if terminated or truncated:
            # Terminal state: return zeros (agent will not act again)
            obs = np.zeros(OBS_DIM, dtype=np.float32)
        else:
            obs = self._fs.get_obs_vector(self._queue_index)

        info = self._build_info(reward=reward, feedback=feedback)
        if self.render_mode == "human":
            self._render_step(info)
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Render the current environment state to stdout.

        Only active when render_mode='human'.  Called automatically by
        step() and reset() when render_mode is set.
        """
        if self.render_mode != "human":
            return
        idx = min(self._queue_index, len(self._fs.files) - 1)
        f = self._fs.files[idx]
        print(
            f"[{idx+1:02d}/{len(self._fs.files)}] "
            f"{f['path']:<45} "
            f"sig={f['ai_signature']:.2f}  "
            f"size={f['size_mb']:.2f}MB  "
            f"label={f['label']}"
        )

    def close(self) -> None:
        """Clean up resources. No external handles to close in this env."""
        pass

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_info(self, reward: float, feedback: str) -> Dict[str, Any]:
        """
        Build the info dict returned from step() and reset().

        The info dict always contains rich episode metadata so that
        judges and evaluators can inspect environment behaviour without
        modifying source code.

        Returns:
            Dict with keys: total_recovered_mb, integrity_ok,
            files_deleted, files_consolidated, files_compressed,
            files_kept, false_positives, queue_remaining,
            step_reward, feedback.
        """
        return {
            "total_recovered_mb":     round(self._total_recovered_mb, 4),
            "integrity_ok":           not self._integrity_broken,
            "files_deleted":          self._delete_count,
            "files_consolidated":     self._consolidate_count,
            "files_compressed":       self._compress_count,
            "files_kept":             self._keep_count,
            "false_positives":        self._false_positive_count,
            "queue_remaining":        max(0, len(self._fs.files) - self._queue_index),
            "step_reward":            round(reward, 4),
            "feedback":               feedback,
        }

    def _render_step(self, info: Dict) -> None:
        """Print a one-line step summary to stdout."""
        print(
            f"  reward={info['step_reward']:+.3f}  "
            f"freed={info['total_recovered_mb']:.2f}MB  "
            f"integrity={'OK' if info['integrity_ok'] else 'BROKEN'}  "
            f"| {info['feedback'][:70]}"
        )
