"""
Inference Script for AI Digital Bloat Detector.

Uses OpenAI-compatible async client to make LLM-based decisions on file classification.
Follows mandatory [START], [STEP], [END] structured logging format.
"""

import os
import sys
import time
import asyncio
import uuid
from typing import Dict, Optional, Tuple

from openai import AsyncOpenAI   # ← FIXED: async client

try:
    from client import AiBloatDetector
    from models import BloatAction
    from tasks import run_all_graders
except (ModuleNotFoundError, ImportError):
    try:
        from .client import AiBloatDetector
        from .models import BloatAction
        from .tasks import run_all_graders
    except (ModuleNotFoundError, ImportError):
        from server.my_env_environment import AiBloatDetector, BloatAction
        run_all_graders = None

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "Qwen/Qwen2.5-72B-Instruct").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_api_base_url_env = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").strip()

_using_hf = bool(HF_TOKEN) and not OPENAI_API_KEY

_default_base_url = (
    "https://router.huggingface.co/v1" if _using_hf else "https://api.openai.com/v1"
)
if _api_base_url_env and "api-inference.huggingface.co" in _api_base_url_env:
    _api_base_url_env = _api_base_url_env.replace(
        "https://api-inference.huggingface.co",
        "https://router.huggingface.co",
    )

_default_model = "Qwen/Qwen2.5-72B-Instruct" if _using_hf else "gpt-4o-mini"

API_BASE_URL = _api_base_url_env or _default_base_url
MODEL_NAME = os.getenv("MODEL_NAME", _default_model)
BLOAT_DETECTOR_URL = os.getenv("BLOAT_DETECTOR_URL", "http://localhost:8000")

MAX_STEPS = 500
DECISION_TIMEOUT_SECS = 30

ENV_NAME = "ai_bloat_detector"

try:
    from tasks.definitions import TASK_NAMES as DEFINED_TASK_NAMES
except Exception:
    DEFINED_TASK_NAMES = ["precision", "recall", "f1_score", "efficiency"]

TASKS = list(DEFINED_TASK_NAMES)


def _fallback_decision(ai_probability: float) -> str:
    if ai_probability > 0.8:
        return "delete"
    if ai_probability > 0.5:
        return "flag"
    return "skip"


def _is_fatal_llm_error(error_text: str) -> bool:
    fatal_markers = (
        "error code: 401", "error code: 403", "error code: 404",
        "error code: 410", "invalid api key", "authentication",
        "not supported", "model not found",
    )
    return any(m in error_text.lower() for m in fatal_markers)


def _select_api_key() -> str:
    base = API_BASE_URL.lower()
    if "openai.com" in base and OPENAI_API_KEY:
        return OPENAI_API_KEY
    if "huggingface.co" in base and HF_TOKEN:
        return HF_TOKEN
    return OPENAI_API_KEY or HF_TOKEN


def _age_days(timestamp: float) -> float:
    if not timestamp:
        return 0.0
    return max(0.0, (time.time() - timestamp) / 86400.0)


def _signal_counts(ai_signals: list) -> Tuple[int, int, int]:
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
    size_bytes = int(file_info.get("size_bytes", 0) or 0)
    child_count = int(file_info.get("child_count", 0) or 0)
    extension = str(file_info.get("extension", "") or "")
    is_directory = bool(file_info.get("is_directory", False))
    age_days = _age_days(float(file_info.get("mtime", 0.0) or 0.0))
    is_recent = age_days < 0.5
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
# LLM Decision Engine — FIXED: async
# ---------------------------------------------------------------------------

async def get_llm_decision(
    client: AsyncOpenAI,          # ← AsyncOpenAI, not OpenAI
    file_path: str,
    file_info: Dict,
    ai_probability: float,
    ai_signals: list,
) -> Tuple[str, Optional[str]]:
    signal_summary = "\n".join([
        f"  - {sig.get('signal_type', 'UNKNOWN')}: {sig.get('description', '')} "
        f"(confidence: {sig.get('confidence', 0):.2f})"
        for sig in ai_signals
    ])

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
        response = await client.chat.completions.create(   # ← await, non-blocking
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

def log_start(task_name: str, episode_id: str, env_name: str) -> None:
    print(
        f"[START] task={task_name} env={env_name} episode_id={episode_id} "
        f"timestamp={time.time():.6f} model={MODEL_NAME}",
        flush=True,
    )


def log_step(task_name, episode_id, step_num, file_path, action_type, reward, ai_probability):
    safe_file = file_path.replace(" ", "_")
    print(
        f"[STEP] task={task_name} episode_id={episode_id} step={step_num} file={safe_file} "
        f"action={action_type} reward={reward:.4f} ai_probability={ai_probability:.4f}",
        flush=True,
    )


def _analyze_decision_history(episode_summary):
    """Analyze history to debug perfect scores.
    
    Returns tuple: (tp_list, fp_list, fn_list, tn_list)
    where each list contains {'path': str, 'action': str, 'is_bloat': bool}
    """
    tp_list, fp_list, fn_list, tn_list = [], [], [], []
    history = episode_summary.get('history', [])
    
    for entry in history:
        if isinstance(entry, dict):
            path = entry.get('path', '')
            action = entry.get('action', '')
            is_bloat = entry.get('is_bloat', False)
            ai_prob = entry.get('ai_probability', 0.0)
            
            entry_dict = {'path': path, 'action': action, 'ai_prob': ai_prob}
            
            if is_bloat and action == 'delete':
                tp_list.append(entry_dict)
            elif not is_bloat and action == 'delete':
                fp_list.append(entry_dict)
            elif is_bloat and action in ('skip', 'flag'):
                fn_list.append(entry_dict)
            elif not is_bloat and action in ('skip', 'flag'):
                tn_list.append(entry_dict)
    
    return tp_list, fp_list, fn_list, tn_list


def log_end(task_name, episode_id, total_steps, episode_summary):
    env_f1 = episode_summary.get('f1_score', 0.0)
    env_prec = episode_summary.get('precision', 0.0)
    env_rec = episode_summary.get('recall', 0.0)
    
    tp = episode_summary.get('true_positives', 0)
    fp = episode_summary.get('false_positives', 0)
    tn = episode_summary.get('true_negatives', 0)
    fn = episode_summary.get('false_negatives', 0)
    
    # Main log line
    print(
        f"[END] task={task_name} env={ENV_NAME} episode_id={episode_id} "
        f"env_score={env_f1:.4f} env_precision={env_prec:.4f} env_recall={env_rec:.4f} "
        f"steps={total_steps} bytes_freed={episode_summary.get('bytes_freed', 0)} "
        f"reward_total={episode_summary.get('reward_total', 0.0):.4f} "
        f"tp={tp} fp={fp} tn={tn} fn={fn} "
        f"timestamp={time.time():.6f}",
        flush=True,
    )
    
    # Diagnostic: Analyze decision history to verify confusion matrix
    tp_list, fp_list, fn_list, tn_list = _analyze_decision_history(episode_summary)
    
    print(
        f"[DIAG] confusion_matrix_verification: "
        f"computed_tp={len(tp_list)} computed_fp={len(fp_list)} "
        f"computed_fn={len(fn_list)} computed_tn={len(tn_list)} | "
        f"reported_tp={tp} reported_fp={fp} reported_fn={fn} reported_tn={tn}",
        flush=True,
    )
    
    # Show problematic decisions that caused incorrect classifications
    if fp_list:
        print(f"[WARN] False positives (deleted human files): {len(fp_list)}", flush=True)
        for item in fp_list[:3]:  # Show first 3
            print(f"       - {item['path']} (ai_prob={item['ai_prob']:.2f})", flush=True)
    
    if fn_list:
        print(f"[WARN] False negatives (missed bloat): {len(fn_list)}", flush=True)
        for item in fn_list[:3]:  # Show first 3
            print(f"       - {item['path']} (ai_prob={item['ai_prob']:.2f})", flush=True)
    
    # If perfect metrics, explain why
    if env_f1 == 1.0 and fp == 0 and fn == 0:
        total_bloat = episode_summary.get('total_ai_bloat_items', 0)
        print(
            f"[INFO] Perfect F1 score (1.0): agent correctly deleted all {tp} bloat items "
            f"and skipped all {tn} human items (0 mistakes)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main Inference Loop
# ---------------------------------------------------------------------------

async def run_task(client: AsyncOpenAI, task_name: str) -> None:
    llm_disabled_reason: Optional[str] = None
    env = None   # ← FIXED: initialise to None before try

    try:
        env = AiBloatDetector(base_url=BLOAT_DETECTOR_URL)
        # Some OpenEnv servers support task_id on reset. Fall back when not supported.
        try:
            result = await env.reset(task_id=task_name)
        except TypeError:
            result = await env.reset()
        except Exception:
            result = await env.reset()

        episode_id = str(uuid.uuid4())
        log_start(task_name, episode_id, ENV_NAME)

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
                action_type, llm_error = await get_llm_decision(  # ← await
                    client, file_path,
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
                    ai_prob, ai_signals,
                )
                if llm_error and _is_fatal_llm_error(llm_error):
                    llm_disabled_reason = llm_error
            else:
                action_type = _fallback_decision(ai_prob)

            action = BloatAction(action_type=action_type)
            try:
                result = await env.step(action)
            except Exception as e:
                print(f"[WARN] step failed: {e}", file=sys.stderr)
                break

            log_step(task_name, episode_id, step_count, file_path, action_type,
                     result.reward or 0.0, ai_prob)

            elapsed = time.time() - start_time
            if elapsed > 1200:  # 20-minute hard limit
                # ← FIXED: send done to get a real episode_summary
                try:
                    result = await env.step(BloatAction(action_type="done"))
                except Exception:
                    pass
                break

        # Ensure we have a terminal summary
        if not result.observation.done:
            try:
                result = await env.step(BloatAction(action_type="done"))
            except Exception:
                pass

        final_summary = result.observation.episode_summary or {}
        log_end(task_name, episode_id, step_count, final_summary)

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if env is not None:       # ← FIXED: guard against uninitialized env
            await env.close()


async def main():
    api_key = _select_api_key()
    if not api_key:
        print(
            "[ERROR] No API key found.\n"
            "  On HuggingFace Spaces: HF_TOKEN is injected automatically.\n"
            "  Locally: set HF_TOKEN or OPENAI_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key, base_url=API_BASE_URL)  # ← AsyncOpenAI
    for task_name in TASKS:
        await run_task(client, task_name)


if __name__ == "__main__":
    asyncio.run(main())