"""
Inference script for the AI Bloat Detector environment.

Uses an OpenAI-compatible LLM to decide delete / flag / skip for each file.
Logs in the mandatory [START] / [STEP] / [END] format.

Environment variables:
    HF_TOKEN        Hugging Face API token (auto-injected on HF Spaces)
    OPENAI_API_KEY  OpenAI API key (alternative)
    API_BASE_URL    Override the inference endpoint
    MODEL_NAME      Override the model name
    ENV_URL         URL of the running environment server

Stdout format (must not deviate):
    [START] task=<task> env=<env> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_TOKEN     = os.getenv("HF_TOKEN", "").strip()
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")
MAX_STEPS    = 500

ENV_NAME          = "ai_bloat_detector"
SUCCESS_THRESHOLD = 0.5   # score >= this -> success=true

TASKS = ["precision", "recall", "f1_score", "efficiency"]

# ---------------------------------------------------------------------------
# Structured logging  -- format is contractual, do not change field names/order
# ---------------------------------------------------------------------------

def log_start(task: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_post(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _probe_server(url: str) -> None:
    """Verify the server is reachable without assuming a JSON response."""
    with urllib.request.urlopen(url, timeout=10) as resp:
        resp.read()

# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an AI bloat detection agent.\n"
    "You receive a file forensic report and must decide what to do.\n\n"
    "Rules:\n"
    "- Reply with EXACTLY one word: DELETE, FLAG, or SKIP.\n"
    "- DELETE: remove this file (AI-generated bloat such as __pycache__, node_modules, .cursor, venv).\n"
    "- FLAG: mark for human review (uncertain, moderate AI probability).\n"
    "- SKIP: keep this file (human-authored, e.g. README, source code, tests).\n\n"
    "Respond with only one word. Nothing else."
)


def _fallback_decision(ai_probability: float) -> str:
    """Rule-based decision used when the LLM is unavailable."""
    p = float(ai_probability) if ai_probability is not None else 0.5
    if p >= 0.85:
        return "delete"
    if p >= 0.50:
        return "flag"
    return "skip"


def _parse_action(text: str) -> str:
    t = (text or "").lower().strip()
    for word in ("delete", "flag", "skip"):
        if word in t:
            return word
    return "skip"


async def _llm_decision(client, obs_text: str) -> Optional[str]:
    """Call LLM; return action string or None on any error."""
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": obs_text},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        return _parse_action(response.choices[0].message.content or "")
    except Exception as e:
        print(f"[WARN] LLM error: {e} -- using rule-based fallback",
              file=sys.stderr, flush=True)
        return None

# ---------------------------------------------------------------------------
# Safe environment wrappers
# ---------------------------------------------------------------------------

def _safe_reset() -> dict:
    try:
        return _http_post(f"{ENV_URL}/reset", {})
    except Exception as e:
        print(f"[ERROR] /reset failed: {e}", file=sys.stderr, flush=True)
        return {"done": True, "episode_summary": {}}


def _safe_step(action: str):
    """Returns (obs_dict, error_string_or_None)."""
    try:
        obs = _http_post(f"{ENV_URL}/step", {"action_type": action})
        return obs, None
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        msg = f"HTTP {e.code}: {body[:120]}"
        print(f"[ERROR] /step {msg}", file=sys.stderr, flush=True)
        return {"done": True, "last_reward": 0.5, "episode_summary": {}}, msg
    except Exception as e:
        msg = str(e)
        print(f"[ERROR] /step failed: {msg}", file=sys.stderr, flush=True)
        return {"done": True, "last_reward": 0.5, "episode_summary": {}}, msg


def _get_ai_prob(obs: dict) -> float:
    try:
        cf = obs.get("current_file")
        if isinstance(cf, dict):
            return float(cf.get("ai_probability") or 0.5)
    except Exception:
        pass
    return 0.5

# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

async def run_task(client, task_name: str) -> None:
    log_start(task_name)

    obs              = _safe_reset()
    rewards: List[float] = []
    steps            = 0
    deadline         = time.time() + 1200   # 20-minute hard limit

    for _ in range(MAX_STEPS):
        done = obs.get("done", False)
        if done:
            break

        obs_text = obs.get("observation_text", "")
        ai_prob  = _get_ai_prob(obs)

        # Decide: try LLM, fall back to rules
        action = None
        if client is not None:
            action = await _llm_decision(client, obs_text)
        if action is None:
            action = _fallback_decision(ai_prob)

        obs, step_error = _safe_step(action)

        reward = float(obs.get("last_reward") or 0.5)
        done   = obs.get("done", False)
        steps += 1
        rewards.append(reward)

        log_step(steps, action, reward, done, step_error)

        if time.time() > deadline:
            print(f"[WARN] Time limit reached for task={task_name}", flush=True)
            break

    # Flush remaining steps to get the terminal summary
    if not obs.get("done"):
        obs, _ = _safe_step("done")

    score   = sum(rewards) / len(rewards) if rewards else 0.0
    score   = max(1e-6, min(score, 1 - 1e-6))   # open interval per OpenEnv spec
    success = score >= SUCCESS_THRESHOLD

    log_end(success, steps, score, rewards)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    # Verify the environment server is up before doing anything else
    try:
        try:
            _probe_server(f"{ENV_URL}/health")
        except Exception:
            _probe_server(f"{ENV_URL}/")
        print(f"[INFO] Environment server: {ENV_URL}", flush=True)
    except Exception as e:
        print(f"[ERROR] Cannot reach environment at {ENV_URL}: {e}",
              file=sys.stderr, flush=True)
        print("[ERROR] Start the server first:  uvicorn server.app:app --port 8000",
              file=sys.stderr, flush=True)
        sys.exit(1)

    # Build the LLM client (optional -- falls back to rule-based if unavailable)
    api_key = HF_TOKEN or OPENAI_KEY
    client  = None
    if api_key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key, base_url=API_BASE_URL)
            print(f"[INFO] LLM client ready: {MODEL_NAME} @ {API_BASE_URL}", flush=True)
        except ImportError:
            print("[WARN] openai package not installed -- rule-based fallback only",
                  file=sys.stderr, flush=True)
    else:
        print("[WARN] No API key found -- rule-based fallback only",
              file=sys.stderr, flush=True)

    for task in TASKS:
        await run_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())
