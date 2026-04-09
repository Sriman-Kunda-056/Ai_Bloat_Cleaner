"""
Inference Script for AI Digital Bloat Detector.

Uses OpenAI-compatible client to make LLM-based decisions on file classification.
Follows mandatory [START], [STEP], [END] structured logging format.

Environment Variables:
    HF_TOKEN        - HuggingFace token (auto-injected in HF Spaces; used for HF Router API)
    OPENAI_API_KEY  - OpenAI key (alternative when using OpenAI directly)
    API_BASE_URL    - LLM endpoint. Defaults to HF Router API when HF_TOKEN is set,
                      or https://api.openai.com/v1 when OPENAI_API_KEY is set.
    MODEL_NAME      - Model identifier. Default: Qwen/Qwen2.5-72B-Instruct (HF) or gpt-4o-mini (OpenAI)
    BLOAT_DETECTOR_URL - Environment server URL. Default: http://localhost:8000

Usage:
    # On HuggingFace Spaces — HF_TOKEN is injected automatically, no action needed.

    # Local with HF Router API:
    HF_TOKEN=hf_xxx python inference.py

    # Local with OpenAI:
    OPENAI_API_KEY=sk-xxx python inference.py

    # Local with custom endpoint:
    HF_TOKEN=hf_xxx API_BASE_URL=https://router.huggingface.co/v1 MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
"""

import json
import os
import sys
import time
import asyncio
import uuid
from typing import Dict, Optional, Tuple

from openai import OpenAI

try:
    from my_env import AiBloatDetector, BloatAction
except (ModuleNotFoundError, ImportError):
    from client import AiBloatDetector
    from models import BloatAction

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_api_base_url_env = os.getenv("API_BASE_URL", "").strip()

# Auto-select sensible defaults depending on which credential is present.
_using_hf = bool(HF_TOKEN) and not OPENAI_API_KEY

_default_base_url = (
    "https://router.huggingface.co/v1" if _using_hf else "https://api.openai.com/v1"
)
if _api_base_url_env and "api-inference.huggingface.co" in _api_base_url_env:
    # Auto-migrate old HF endpoint to avoid HTTP 410 errors.
    _api_base_url_env = _api_base_url_env.replace(
        "https://api-inference.huggingface.co",
        "https://router.huggingface.co",
    )
_default_model = (
    "Qwen/Qwen2.5-72B-Instruct" if _using_hf else "gpt-4o-mini"
)

API_BASE_URL = _api_base_url_env or _default_base_url
MODEL_NAME = os.getenv("MODEL_NAME", _default_model)

# Server endpoint
BLOAT_DETECTOR_URL = os.getenv("BLOAT_DETECTOR_URL", "http://localhost:8000")

# Inference limits
MAX_STEPS = 500
DECISION_TIMEOUT_SECS = 30


def _fallback_decision(ai_probability: float) -> str:
    """Deterministic fallback when LLM is unavailable."""
    if ai_probability > 0.8:
        return "delete"
    if ai_probability > 0.5:
        return "flag"
    return "skip"


def _is_fatal_llm_error(error_text: str) -> bool:
    """Detect provider/auth/config errors that should disable further LLM calls."""
    fatal_markers = (
        "error code: 401",
        "error code: 403",
        "error code: 404",
        "error code: 410",
        "invalid api key",
        "authentication",
        "not supported",
        "model not found",
    )
    lower = error_text.lower()
    return any(marker in lower for marker in fatal_markers)


def _select_api_key() -> str:
    """Choose the right key for the configured backend."""
    base = API_BASE_URL.lower()

    if "openai.com" in base and OPENAI_API_KEY:
        return OPENAI_API_KEY
    if "huggingface.co" in base and HF_TOKEN:
        return HF_TOKEN

    # Generic OpenAI-compatible endpoint fallback.
    return OPENAI_API_KEY or HF_TOKEN


def _age_days(timestamp: float) -> float:
    """Return item age in days from a Unix timestamp."""
    if not timestamp:
        return 0.0
    return max(0.0, (time.time() - timestamp) / 86400.0)


def _signal_counts(ai_signals: list) -> Tuple[int, int, int]:
    """Summarize signal strength for prompt conditioning."""
    high = medium = low = 0
    for sig in ai_signals:
        confidence = float(sig.get("confidence", 0.0))
        if confidence >= 0.8:
            high += 1
        elif confidence >= 0.5:
            medium += 1
        else:
            low += 1
    return high, medium, low


def _build_feature_summary(file_info: Dict, ai_probability: float, ai_signals: list) -> str:
    """Build a compact feature summary that adds derived evidence to the prompt."""
    size_bytes = int(file_info.get("size_bytes", 0) or 0)
    child_count = int(file_info.get("child_count", 0) or 0)
    extension = str(file_info.get("extension", "") or "")
    is_directory = bool(file_info.get("is_directory", False))
    is_recent = _age_days(float(file_info.get("mtime", 0.0) or 0.0)) < 0.5
    age_days = _age_days(float(file_info.get("mtime", 0.0) or 0.0))
    type_mismatch = bool(file_info.get("type_mismatch", False))
    high_signals, medium_signals, low_signals = _signal_counts(ai_signals)

    if is_directory:
        size_bucket = "directory"
    elif size_bytes == 0:
        size_bucket = "empty"
    elif size_bytes < 2048:
        size_bucket = "tiny"
    elif size_bytes < 10240:
        size_bucket = "small"
    elif size_bytes < 100000:
        size_bucket = "medium"
    else:
        size_bucket = "large"

    return (
        f"FILE_EXT={extension or '(none)'}\n"
        f"FILE_KIND={'directory' if is_directory else 'file'}\n"
        f"SIZE_BUCKET={size_bucket} SIZE_BYTES={size_bytes}\n"
        f"CHILD_COUNT={child_count}\n"
        f"AGE_DAYS={age_days:.2f} IS_RECENT={str(is_recent).lower()}\n"
        f"TYPE_MISMATCH={str(type_mismatch).lower()}\n"
        f"AI_PROBABILITY={ai_probability:.2f}\n"
        f"SIGNAL_COUNTS high={high_signals} medium={medium_signals} low={low_signals}"
    )


# ---------------------------------------------------------------------------
# LLM Decision Engine
# ---------------------------------------------------------------------------


def get_llm_decision(
    client: OpenAI,
    file_path: str,
    file_info: Dict,
    ai_probability: float,
    ai_signals: list,
) -> Tuple[str, Optional[str]]:
    """
    Query LLM to classify the file as delete/flag/skip.

    Args:
        client: Initialized OpenAI client
        file_path: Relative path of the file
        file_info: FileFingerprint details (size, type, etc.)
        ai_probability: Composite AI-generation probability [0, 1]
        ai_signals: List of AI signal objects with confidence scores

    Returns:
        (action_type, error_message)
    """
    signal_summary = "\n".join(
        [
            f"  - {sig.get('signal_type', 'UNKNOWN')}: {sig.get('description', '')} "
            f"(confidence: {sig.get('confidence', 0):.2f})"
            for sig in ai_signals
        ]
    )

    feature_summary = _build_feature_summary(file_info, ai_probability, ai_signals)

    prompt = f"""You are an expert AI filesystem forensics analyst.
Use the derived features and signal details below to decide whether the item is likely AI-generated bloat.
Prefer evidence-based decisions over raw probability alone.

Return exactly one word: delete, flag, or skip.

DERIVED FEATURES:
{feature_summary}

FILE PATH: {file_path}

AI SIGNALS DETECTED:
{signal_summary if signal_summary else "  (none)"}

DECISION RULES:
1. If ai_probability > 0.85 AND multiple signals align, respond "delete"
2. If 0.6 < ai_probability <= 0.85, respond "flag" for human review
3. If ai_probability <= 0.6, respond "skip" (likely human-created)
4. For directories with high ai_probability, prefer "flag" over "delete"

RESPOND WITH EXACTLY ONE WORD: delete | flag | skip
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=10,
            timeout=DECISION_TIMEOUT_SECS,
        )
        decision = response.choices[0].message.content.strip().lower()

        if decision in ("delete", "flag", "skip"):
            return decision, None

        return _fallback_decision(ai_probability), "unexpected model output"
    except Exception as e:
        return _fallback_decision(ai_probability), str(e)


# ---------------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------------


def log_start(episode_id: str, env_name: str) -> None:
    print(
        f"[START] task={env_name} episode_id={episode_id} "
        f"timestamp={time.time():.6f} model={MODEL_NAME}",
        flush=True,
    )


def log_step(
    episode_id: str,
    step_num: int,
    file_path: str,
    action_type: str,
    reward: float,
    ai_probability: float,
) -> None:
    safe_file = file_path.replace(" ", "_")
    print(
        f"[STEP] episode_id={episode_id} step={step_num} file={safe_file} "
        f"action={action_type} reward={reward:.4f} ai_probability={ai_probability:.4f}",
        flush=True,
    )


def log_end(
    episode_id: str,
    total_steps: int,
    episode_summary: Dict,
) -> None:
    print(
        f"[END] task=ai_bloat_detector episode_id={episode_id} "
        f"score={episode_summary.get('f1_score', 0.0):.4f} steps={total_steps} "
        f"precision={episode_summary.get('precision', 0.0):.4f} "
        f"recall={episode_summary.get('recall', 0.0):.4f} "
        f"bytes_freed={episode_summary.get('bytes_freed', 0)} "
        f"reward_total={episode_summary.get('reward_total', 0.0):.4f} "
        f"timestamp={time.time():.6f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main Inference Loop
# ---------------------------------------------------------------------------


async def main():
    """Run inference on the bloat detector environment."""

    api_key = _select_api_key()
    if not api_key:
        print(
            "[ERROR] No API key found.\n"
            "  On HuggingFace Spaces: HF_TOKEN is injected automatically — nothing to do.\n"
            "  Locally: set HF_TOKEN (for HF Router API) or OPENAI_API_KEY (for OpenAI).\n"
            "  Example: HF_TOKEN=hf_xxx python inference.py",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
    llm_disabled_reason: Optional[str] = None

    try:
        env = AiBloatDetector(base_url=BLOAT_DETECTOR_URL)
    except Exception as e:
        print(
            f"[ERROR] Failed to connect to environment at {BLOAT_DETECTOR_URL}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        result = await env.reset()
        episode_id = str(uuid.uuid4())
        log_start(episode_id, "ai_bloat_detector")

        step_count = 0
        start_time = time.time()

        while not result.observation.done and step_count < MAX_STEPS:
            step_count += 1

            current_item = result.observation.current_item
            if current_item is None:
                break

            file_path = current_item.path
            ai_prob = current_item.ai_probability

            try:
                ai_signals = [
                    {
                        "signal_type": sig.signal_type,
                        "description": sig.description,
                        "confidence": sig.confidence,
                    }
                    for sig in current_item.ai_signals
                ]
            except Exception:
                ai_signals = []

            if llm_disabled_reason is None:
                action_type, llm_error = get_llm_decision(
                    client,
                    file_path,
                    {
                        "size_bytes": current_item.size_bytes,
                        "is_directory": current_item.is_directory,
                        "type": current_item.detected_type,
                        "child_count": current_item.child_count,
                        "extension": current_item.extension,
                        "mtime": current_item.mtime,
                        "ctime": current_item.ctime,
                        "atime": current_item.atime,
                        "type_mismatch": current_item.type_mismatch,
                    },
                    ai_prob,
                    ai_signals,
                )
                if llm_error and _is_fatal_llm_error(llm_error):
                    llm_disabled_reason = llm_error
            else:
                action_type = _fallback_decision(ai_prob)

            action = BloatAction(action_type=action_type)
            try:
                result = await env.step(action)
            except Exception:
                break

            log_step(
                episode_id,
                step_count,
                file_path,
                action_type,
                result.reward or 0.0,
                ai_prob,
            )

            elapsed = time.time() - start_time
            if elapsed > 1200:  # 20 minutes
                break

        # --- Episode complete ---
        final_summary = result.observation.episode_summary or {}
        log_end(episode_id, step_count, final_summary)

        return

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
