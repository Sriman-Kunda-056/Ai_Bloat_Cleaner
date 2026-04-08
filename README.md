---
title: AI Digital Bloat Detector
emoji: "\U0001F50D"
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - filesystem
  - ai-safety
---

# AI Digital Bloat Detector

> **Meta  Scaler Hackathon**  "AI-Generated Digital Bloat" Track

An RL environment that trains an agent to forensically scan a developer workspace,
identify AI-generated bloat, and delete it  earning rewards for precision and
recall, with heavy penalties for destroying real human work.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ on HF Spaces (auto-injected) | — | HuggingFace token — used as the API key for the HF Inference API |
| `OPENAI_API_KEY` | Alternative to `HF_TOKEN` | — | OpenAI key (use when pointing `API_BASE_URL` at OpenAI) |
| `API_BASE_URL` | No | `https://api-inference.huggingface.co/v1` (HF) / `https://api.openai.com/v1` (OpenAI) | LLM endpoint (any OpenAI-compatible API) |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` (HF) / `gpt-4o-mini` (OpenAI) | Model identifier |
| `BLOAT_DETECTOR_URL` | No | `http://localhost:8000` | URL of the running environment server |

> **HF Spaces**: `HF_TOKEN` is injected automatically — no secrets to configure.

## The Problem

Modern AI coding agents (Cursor, GitHub Copilot, Claude Code, etc.) leave
behind a trail of digital waste:

| Bloat Type | Example | Typical Size |
|---|---|---|
| Hidden agent configs | `.cursorrules`, `.claude/` | KBs |
| Dependency trees | `node_modules/`, `venv/` | 100s of MBs |
| Build caches | `__pycache__/`, `.pytest_cache/` | MBs |
| Batch-scaffolded boilerplate | `utils.py`, `services.py`, `helpers.py` (all same mtime) | KBs |
| Disguised binaries | `secret.png` with Python content | KBs |
| Duplicate drafts | `temp_draft_v1.py` == `temp_draft_v1_copy.py` | KBs |

## Quick Start

```python
from my_env import AiBloatDetector, BloatAction

with AiBloatDetector(base_url="http://localhost:8000") as env:
    result = env.reset()

    while not result.observation.done:
        item = result.observation.current_item

        # Your RL policy here  use ai_probability and ai_signals
        action = BloatAction(
            action_type="delete" if item.ai_probability > 0.6 else "skip"
        )
        result = env.step(action)

    summary = result.observation.episode_summary
    print(f"F1={summary['f1_score']:.3f}  Bytes freed={summary['bytes_freed']:,}")
```

## Action Space

| `action_type` | Effect | Reward |
|---|---|---|
| `"delete"` | Remove item from disk | +1.00 (TP) / **-2.00 (FP)** |
| `"flag"` | Mark for human review | +0.40 (TP) / -0.40 (FP) |
| `"skip"` | Keep the item | +0.30 (TN) / -0.30 (FN) |
| `"done"` | End episode early | 0.00 + F1 bonus |

Terminal bonus: `+3.0  F1` applied at episode end.

## Observation: FileFingerprint

Each step the agent receives a `FileFingerprint` with:

```
path               relative path within the workspace
is_directory       True for directory items
size_bytes         file size (or total subtree size for dirs)
ctime/mtime/atime  filesystem timestamps
sha256_hash        content hash (files only)
magic_header       first 16 bytes in hex (for type verification)
declared_type      type inferred from extension
detected_type      type inferred from magic bytes
type_mismatch      True when extension contradicts content
ai_signals         list of AISignal objects with confidence scores
ai_probability     composite AI-generation probability [0, 1]
```

### AI Signals

| Signal | Description |
|---|---|
| `HIDDEN_ARTIFACT_DIR` | `.cursorrules`, `.claude/`, `.cursor/` |
| `DEPENDENCY_BLOAT` | `node_modules/`, `venv/` |
| `BUILD_CACHE` | `__pycache__/`, `.pytest_cache/` |
| `BATCH_CREATION` | 3 files sharing the same modification timestamp |
| `DUPLICATE_CONTENT` | Identical SHA-256 across multiple files |
| `TEMP_DRAFT` | Filename contains `temp_`, `draft_`, `_copy`, `.bak` |
| `BYTECODE_ARTIFACT` | `.pyc` files |
| `VIRTUALENV_INTERNAL` | `pyvenv.cfg` manifest |
| `TYPE_MISMATCH` | Extension says image/binary but magic bytes say text |
| `AI_SCAFFOLD_NAME` | `utils.py`, `services.py`, `helpers.py` etc. |

## Synthetic Workspace

Each `reset()` creates a fresh temp directory with **ground-truth labels**:

```
workspace/
 .cursorrules               AI bloat (agent config)
 .claude/settings.json      AI bloat (agent config)
 .github/prompts/...        AI bloat (Copilot prompts)
 node_modules/              AI bloat (dependency tree)
 __pycache__/               AI bloat (bytecode cache)
 venv/pyvenv.cfg            AI bloat (virtual env)
 src/utils.py               AI bloat (batch scaffold, same mtime)
 src/services.py            AI bloat (batch scaffold)
 src/controllers.py         AI bloat (batch scaffold)
 src/helpers.py             AI bloat (batch scaffold)
 temp_draft_v1.py           AI bloat (temp file)
 temp_draft_v1_copy.py      AI bloat (duplicate)
 assets/secret.png          AI bloat (TYPE_MISMATCH: Python in .png)
 README.md                  HUMAN (30 days old)
 notes.txt                  HUMAN (7 days old)
 requirements.txt           HUMAN (5 days old, 2 packages)
```

Items are shuffled on each reset so the agent cannot exploit order.

## Building & Running

```bash
# Build Docker image
docker build -t ai_bloat_detector-env:latest -f server/Dockerfile .

# Run server
docker run -p 8000:8000 ai_bloat_detector-env:latest

# Or locally with uvicorn
uvicorn server.app:app --reload
```

## Project Structure

```
my_env/
 __init__.py                    # Exports: AiBloatDetector, BloatAction, BloatObservation
 models.py                      # BloatAction, BloatObservation, FileFingerprint, AISignal
 client.py                      # AiBloatDetector WebSocket client
 openenv.yaml                   # OpenEnv manifest
 pyproject.toml                 # Package metadata
 server/
     my_env_environment.py      # AiBloatDetectorEnvironment (core logic)
     app.py                     # FastAPI HTTP + WebSocket server
```
