
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
from typing import Dict, Optional

from openai import OpenAI

try:
    from my_env import AiBloatDetector, BloatAction
except ModuleNotFoundError:
    from client import AiBloatDetector
    from models import BloatAction

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Auto-select sensible defaults depending on which credential is present.
# HF_TOKEN → HuggingFace Router API (OpenAI-compatible)
# OPENAI_API_KEY → OpenAI directly
_using_hf = bool(HF_TOKEN) and not OPENAI_API_KEY

_default_base_url = (
    "https://router.huggingface.co/v1" if _using_hf else "https://api.openai.com/v1"
)
_default_model = (
    "Qwen/Qwen2.5-72B-Instruct" if _using_hf else "gpt-4o-mini"
)

API_BASE_URL = os.getenv("API_BASE_URL", _default_base_url)
MODEL_NAME = os.getenv("MODEL_NAME", _default_model)

# Server endpoint
BLOAT_DETECTOR_URL = os.getenv("BLOAT_DETECTOR_URL", "http://localhost:8000")

# Inference limits
MAX_STEPS = 500
DECISION_TIMEOUT_SECS = 30


# 
# LLM Decision Engine
# 


def get_llm_decision(
    client: OpenAI,
    file_path: str,
    file_info: Dict,
    ai_probability: float,
    ai_signals: list,
) -> str:
    """
    Query LLM to classify the file as delete/flag/skip.

    Args:
        client: Initialized OpenAI client
        file_path: Relative path of the file
        file_info: FileFingerprint details (size, type, etc.)
        ai_probability: Composite AI-generation probability [0, 1]
        ai_signals: List of AI signal objects with confidence scores

    Returns:
        action_type: "delete", "flag", or "skip"
    """
    # Format signals for the prompt
    signal_summary = "\n".join(
        [
            f"  - {sig.get('signal_type', 'UNKNOWN')}: {sig.get('description', '')} "
            f"(confidence: {sig.get('confidence', 0):.2f})"
            for sig in ai_signals
        ]
    )

    prompt = f"""You are an expert AI filesystem forensics analyst. Classify the following file as likely AI-generated bloat ("delete"), uncertain ("flag"), or human-created ("skip").

FILE PATH: {file_path}
IS_DIRECTORY: {file_info.get('is_directory', False)}
SIZE_BYTES: {file_info.get('size_bytes', 0):,}
AI_PROBABILITY: {ai_probability:.2f}

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

        # Validate response
        if decision in ("delete", "flag", "skip"):
            return decision
        else:
            # Fallback: use ai_probability as tiebreaker
            if ai_probability > 0.85:
                return "delete"
            elif ai_probability > 0.6:
                return "flag"
            else:
                return "skip"
    except Exception as e:
        print(f"[ERROR] LLM decision failed: {e}", file=sys.stderr)
        # Fallback strategy based on ai_probability
        if ai_probability > 0.8:
            return "delete"
        elif ai_probability > 0.5:
            return "flag"
        else:
            return "skip"


# 
# Structured Logging
# 


def log_start(episode_id: str, env_name: str) -> None:
    """Log episode start."""
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
    """Log a single step."""
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
    """Log episode end."""
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


# 
# Main Inference Loop
# 


async def main():
    """Run inference on the bloat detector environment."""

    # Initialize OpenAI-compatible client
    api_key = HF_TOKEN or OPENAI_API_KEY
    if not api_key:
        print(
            "[ERROR] No API key found.\n"
            "  On HuggingFace Spaces: HF_TOKEN is injected automatically — nothing to do.\n"
            "  Locally: set HF_TOKEN (for HF Inference API) or OPENAI_API_KEY (for OpenAI).\n"
            "  Example: HF_TOKEN=hf_xxx python inference.py",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=API_BASE_URL)

    # Connect to environment
    try:
        env = AiBloatDetector(base_url=BLOAT_DETECTOR_URL)
    except Exception as e:
        print(
            f"[ERROR] Failed to connect to environment at {BLOAT_DETECTOR_URL}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        # Reset environment
        # NOTE: StepResult has no .state attribute; episode_id is internal
        # server state never sent to the client. We generate a local UUID
        # for structured logging correlation.
        result = await env.reset()
        episode_id = str(uuid.uuid4())
        log_start(episode_id, "ai_bloat_detector")

        step_count = 0
        start_time = time.time()

        # Main loop
        while not result.observation.done and step_count < MAX_STEPS:
            step_count += 1

            # Extract current file info
            current_item = result.observation.current_item
            if current_item is None:
                break

            file_path = current_item.path
            ai_prob = current_item.ai_probability

            # Safely parse AI signals (network/parsing can fail)
            try:
                ai_signals = [
                    {
                        "signal_type": sig.signal_type,
                        "description": sig.description,
                        "confidence": sig.confidence,
                    }
                    for sig in current_item.ai_signals
                ]
            except Exception as e:
                print(f"[WARNING] Failed to parse ai_signals at step {step_count}: {e}", file=sys.stderr)
                ai_signals = []

            # Get LLM decision (wrapped: network call can fail)
            try:
                action_type = get_llm_decision(
                    client,
                    file_path,
                    {
                        "size_bytes": current_item.size_bytes,
                        "is_directory": current_item.is_directory,
                        "type": current_item.detected_type,
                    },
                    ai_prob,
                    ai_signals,
                )
            except Exception as e:
                print(f"[WARNING] LLM decision error at step {step_count}: {e}", file=sys.stderr)
                # Fallback: use probability threshold
                action_type = "delete" if ai_prob > 0.85 else "flag" if ai_prob > 0.6 else "skip"

            # Execute action (wrapped: env step can fail)
            action = BloatAction(action_type=action_type)
            try:
                result = await env.step(action)
            except Exception as e:
                print(f"[ERROR] env.step() failed at step {step_count}: {e}", file=sys.stderr)
                break

            # Log step
            # NOTE: reward lives on StepResult, not on the observation dict.
            log_step(
                episode_id,
                step_count,
                file_path,
                action_type,
                result.reward or 0.0,
                ai_prob,
            )

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > 1200:  # 20 minutes
                print(
                    f"[WARNING] Runtime exceeded 20 minutes, terminating early",
                    file=sys.stderr,
                )
                break

        # Episode complete — episode_summary may be None if we broke out early
        final_summary = result.observation.episode_summary or {}
        log_end(episode_id, step_count, final_summary)

        print(
            f"\n[SUMMARY] Episode {episode_id}: "
            f"F1={final_summary.get('f1_score', 0):.3f}, "
            f"Bytes freed={final_summary.get('bytes_freed', 0):,}, "
            f"Steps={step_count}",
            file=sys.stderr,
        )

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
