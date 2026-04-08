

"""AI Digital Bloat Detector - Environment Client."""

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import AISignal, BloatAction, BloatObservation, FileFingerprint
except ImportError:
    from models import AISignal, BloatAction, BloatObservation, FileFingerprint


class AiBloatDetector(
    EnvClient[BloatAction, BloatObservation, State]
):
    """
    Client for the AI Digital Bloat Detector Environment.

    Maintains a persistent WebSocket connection for efficient multi-step
    interactions.  Each instance gets a dedicated episode on the server.

    Example - rule-based oracle::

        with AiBloatDetector(base_url="http://localhost:8000") as env:
            result = env.reset()
            while not result.observation.done:
                item = result.observation.current_item
                action = BloatAction(
                    action_type="delete" if item.ai_probability > 0.6 else "skip"
                )
                result = env.step(action)
            print(result.observation.episode_summary)

    Example - Docker launch::

        client = AiBloatDetector.from_docker_image("ai_bloat_detector-env:latest")
        try:
            result = client.reset()
            ...
        finally:
            client.close()
    """

    def _step_payload(self, action: BloatAction) -> Dict:
        return {"action_type": action.action_type}

    def _parse_result(self, payload: Dict) -> StepResult[BloatObservation]:
        obs_data = payload.get("observation", {})

        # Reconstruct nested FileFingerprint
        current_item = None
        ci_data = obs_data.get("current_item")
        if ci_data:
            signals = [
                AISignal(
                    signal_type=s["signal_type"],
                    description=s["description"],
                    confidence=s["confidence"],
                )
                for s in ci_data.get("ai_signals", [])
            ]
            current_item = FileFingerprint(
                path=ci_data.get("path", ""),
                is_directory=ci_data.get("is_directory", False),
                size_bytes=ci_data.get("size_bytes", 0),
                child_count=ci_data.get("child_count", 0),
                extension=ci_data.get("extension", ""),
                ctime=ci_data.get("ctime", 0.0),
                mtime=ci_data.get("mtime", 0.0),
                atime=ci_data.get("atime", 0.0),
                sha256_hash=ci_data.get("sha256_hash", ""),
                magic_header=ci_data.get("magic_header", ""),
                declared_type=ci_data.get("declared_type", ""),
                detected_type=ci_data.get("detected_type", ""),
                type_mismatch=ci_data.get("type_mismatch", False),
                ai_signals=signals,
                ai_probability=ci_data.get("ai_probability", 0.05),
            )

        observation = BloatObservation(
            current_item=current_item,
            queue_size=obs_data.get("queue_size", 0),
            step_count=obs_data.get("step_count", 0),
            bytes_freed=obs_data.get("bytes_freed", 0),
            true_positives=obs_data.get("true_positives", 0),
            false_positives=obs_data.get("false_positives", 0),
            true_negatives=obs_data.get("true_negatives", 0),
            false_negatives=obs_data.get("false_negatives", 0),
            precision=obs_data.get("precision", 0.0),
            recall=obs_data.get("recall", 0.0),
            f1_score=obs_data.get("f1_score", 0.0),
            last_action_result=obs_data.get("last_action_result", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            episode_summary=obs_data.get("episode_summary"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
