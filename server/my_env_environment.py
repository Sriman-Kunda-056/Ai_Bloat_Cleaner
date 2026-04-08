

"""
AI Digital Bloat Detector - RL Environment.

The environment generates a synthetic developer workspace pre-populated with a
realistic mix of AI-generated bloat and genuine human files.  On each step the
agent receives a full forensic FileFingerprint and must decide whether to
delete, flag, skip, or declare done.

Reward Structure
----------------
    delete  + AI bloat  -> +1.00  (true positive  - bloat correctly removed)
    delete  + human     -> -2.00  (false positive  - penalise destroying real work)
    flag    + AI bloat  -> +0.40  (partial credit  - correctly uncertain)
    flag    + human     -> -0.40  (false flag)
    skip    + human     -> +0.30  (true negative   - correctly preserved)
    skip    + AI bloat  -> -0.30  (false negative  - missed bloat)
    done (early exit)   ->  0.00  + F1 bonus applied to episode total

Episode Termination
-------------------
The episode ends when the agent sends action_type="done" OR the queue is
exhausted.  A terminal F1-score bonus ( max_bonus * F1 ) is added to the
final step reward.
"""

import hashlib
import os
import random
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AISignal, BloatAction, BloatObservation, FileFingerprint
except ImportError:
    from models import AISignal, BloatAction, BloatObservation, FileFingerprint


# ---------------------------------------------------------------------------
# Constants - forensic knowledge base
# ---------------------------------------------------------------------------

# Magic-byte signatures  {prefix_bytes: detected_type}
MAGIC_SIGNATURES: List[Tuple[bytes, str]] = [
    (b"\x89PNG",         "png"),
    (b"\xff\xd8\xff",    "jpeg"),
    (b"GIF8",            "gif"),
    (b"%PDF",            "pdf"),
    (b"PK\x03\x04",      "zip"),
    (b"\x7fELF",         "elf_binary"),
    (b"MZ",              "windows_exe"),
    (b"\xca\xfe\xba\xbe","macho_binary"),
    (b"\xfe\xed\xfa\xce","macho_binary"),
]

EXTENSION_TYPES: Dict[str, str] = {
    ".py": "python",  ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript", ".go": "go",
    ".rs": "rust",    ".java": "java",      ".cpp": "cpp",
    ".c": "c",        ".png": "png",        ".jpg": "jpeg",
    ".gif": "gif",    ".pdf": "pdf",        ".zip": "zip",
    ".exe": "windows_exe", ".json": "json",  ".yaml": "yaml",
    ".yml": "yaml",   ".md": "markdown",    ".txt": "text",
    ".cfg": "config", ".pyc": "bytecode",   ".class": "java_class",
}

# Directory names that are strong AI/tool indicators
AI_DIR_SIGNALS: Dict[str, Tuple[str, str, float]] = {
    # name -> (signal_type, description_template, confidence)
    ".cursorrules": (
        "HIDDEN_ARTIFACT_DIR",
        "'.cursorrules' is a Cursor AI agent configuration file",
        0.97,
    ),
    ".claude": (
        "HIDDEN_ARTIFACT_DIR",
        "'.claude/' is an Anthropic Claude agent workspace directory",
        0.97,
    ),
    ".cursor": (
        "HIDDEN_ARTIFACT_DIR",
        "'.cursor/' is a Cursor IDE AI configuration directory",
        0.95,
    ),
    ".github": (
        "HIDDEN_ARTIFACT_DIR",
        "'.github/' may contain AI-generated prompt templates and Copilot configs",
        0.60,
    ),
    "node_modules": (
        "DEPENDENCY_BLOAT",
        "'node_modules/' is a reinstallable Node.js dependency tree - classic AI scaffolding bloat",
        0.88,
    ),
    "venv": (
        "DEPENDENCY_BLOAT",
        "'venv/' is a Python virtual environment - regenerable, large, often AI-spawned",
        0.82,
    ),
    ".venv": (
        "DEPENDENCY_BLOAT",
        "'.venv/' is a Python virtual environment directory",
        0.82,
    ),
    "__pycache__": (
        "BUILD_CACHE",
        "'__pycache__/' is auto-generated Python bytecode cache",
        0.99,
    ),
    ".pytest_cache": ("BUILD_CACHE", "pytest auto-generated cache", 0.99),
    ".mypy_cache":   ("BUILD_CACHE", "mypy type-checker auto-generated cache", 0.99),
    ".ruff_cache":   ("BUILD_CACHE", "ruff linter auto-generated cache", 0.99),
    "dist":          ("BUILD_OUTPUT", "'dist/' is a compiled distribution artifact directory", 0.78),
    "build":         ("BUILD_OUTPUT", "'build/' is a compiled output directory", 0.75),
    ".next":         ("BUILD_OUTPUT", "'.next/' is a Next.js build-output directory", 0.95),
}

# Filename stems that appear in AI-generated boilerplate scaffolds
AI_SCAFFOLD_STEMS = {
    "utils", "helpers", "services", "constants", "types",
    "interfaces", "schemas", "routes", "controllers",
    "middleware", "config", "settings",
}

# Substrings in filenames that suggest temp / draft files
TEMP_PATTERNS = ["temp_", "tmp_", "draft_", "_backup", "_copy", "_old", ".bak", ".swp", ".tmp"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _magic_header(path: Path) -> str:
    try:
        with open(path, "rb") as fh:
            return fh.read(16).hex()
    except OSError:
        return ""


def _detect_type(magic_hex: str) -> str:
    if not magic_hex:
        return "unknown"
    try:
        raw = bytes.fromhex(magic_hex[:16])
    except ValueError:
        return "unknown"
    for sig, ftype in MAGIC_SIGNATURES:
        if raw[: len(sig)] == sig:
            return ftype
    try:
        raw.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "binary"


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


# ---------------------------------------------------------------------------
# Synthetic workspace factory
# ---------------------------------------------------------------------------

_AI_BLOAT_SCAFFOLD_CONTENT = '''"""Auto-generated scaffold module."""

from typing import Any, Dict, List, Optional


def process(data: Any) -> Dict:
    """Process input data."""
    raise NotImplementedError


def validate(data: Any) -> bool:
    """Validate input data."""
    raise NotImplementedError


def transform(items: List[Any]) -> List[Any]:
    """Transform a list of items."""
    return [process(item) for item in items]
'''

_PYVENV_CFG = """home = /usr/bin
include-system-site-packages = false
version = 3.11.0
virtualenv = 20.26.6
"""

_CURSORRULES = """{
  "rules": [
    "Always use TypeScript",
    "Prefer functional components",
    "Use async/await over callbacks",
    "Add JSDoc comments to all exported functions",
    "Follow the repository naming convention: camelCase for variables, PascalCase for components"
  ],
  "ignore": ["node_modules", "dist", "build", ".next"]
}
"""

_CLAUDE_SETTINGS = """{
  "model": "claude-opus-4-5",
  "temperature": 0.2,
  "max_tokens": 8192,
  "system_prompt": "You are an expert full-stack developer. Generate clean, production-ready code.",
  "auto_commit": true,
  "workspace": "/project"
}
"""

_GITHUB_COPILOT_PROMPT = """---
name: Code Review Assistant
description: Automated code review using GitHub Copilot
---

You are a senior engineer reviewing pull requests.
Always check for: security vulnerabilities, performance issues,
missing tests, and style guide violations.
"""

_README = """# My Project

A personal project I am working on.

## Setup

    pip install -r requirements.txt
    python main.py

## Notes

This is a work-in-progress.  See notes.txt for TODO items.
"""

_NOTES_TXT = """TODO:
- finish the authentication module
- write tests for the parser
- update README once API is stable

DONE:
- set up project structure
- connected to database
"""

_REQUIREMENTS_TXT = """requests>=2.28.0
pydantic>=2.0.0
"""

_SECRET_PNG_CONTENT = b"""# This file is disguised as a PNG but is actually Python source.
# A real attacker (or careless AI agent) might use this to hide payloads.
def exploit():
    pass
"""


def _plant_workspace(root: Path) -> Dict[str, bool]:
    """
    Create a synthetic developer workspace under *root* and return a dict
    mapping each relative path to True (AI bloat) or False (human file).
    """
    ground_truth: Dict[str, bool] = {}

    now = time.time()
    old_human = now - 30 * 86400   # 30 days ago  human authored files
    recent_ai = now - 600          # 10 min ago   AI agent session
    batch_ts = recent_ai           # all scaffold files share this mtime

    def _w(rel: str, content: bytes | str, mtime: float, is_bloat: bool):
        """Write a file and record ground truth."""
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            content = content.encode()
        p.write_bytes(content)
        os.utime(p, (mtime, mtime))
        ground_truth[rel] = is_bloat

    #  AI Bloat: hidden agent config files 
    _w(".cursorrules", _CURSORRULES, recent_ai, True)
    _w(".claude/settings.json", _CLAUDE_SETTINGS, recent_ai, True)
    _w(".github/prompts/copilot_system.md", _GITHUB_COPILOT_PROMPT, recent_ai, True)

    #  AI Bloat: virtual environment 
    _w("venv/pyvenv.cfg", _PYVENV_CFG, recent_ai, True)
    _w("venv/lib/python3.11/site-packages/placeholder.txt", b"# placeholder\n", recent_ai, True)
    os.utime(root / "venv", (recent_ai, recent_ai))

    #  AI Bloat: __pycache__ (bytecode) 
    # Fake .pyc files (real .pyc has magic header 0x0d0d0a0a etc.)
    pyc_magic = b"\x6f\x0d\x0d\x0a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    _w("__pycache__/main.cpython-311.pyc", pyc_magic + b"\x00" * 32, recent_ai, True)
    _w("__pycache__/utils.cpython-311.pyc", pyc_magic + b"\x00" * 28, recent_ai, True)
    os.utime(root / "__pycache__", (recent_ai, recent_ai))

    #  AI Bloat: node_modules (dependency bloat) 
    _w("node_modules/express/index.js", b"module.exports = require('./lib/express');", recent_ai, True)
    _w("node_modules/express/package.json", b'{"name":"express","version":"4.18.2"}', recent_ai, True)
    _w("node_modules/lodash/lodash.js", b"// lodash v4.17.21\n" + b"var _ = {};\n" * 200, recent_ai, True)
    os.utime(root / "node_modules", (recent_ai, recent_ai))

    #  AI Bloat: batch-created scaffold files (all same mtime) 
    for stem in ["utils", "services", "controllers", "helpers"]:
        _w(f"src/{stem}.py", _AI_BLOAT_SCAFFOLD_CONTENT, batch_ts, True)

    #  AI Bloat: temp / draft files 
    draft_content = b"# Work in progress - do not commit\nresult = None\n"
    _w("temp_draft_v1.py", draft_content, recent_ai, True)
    # Exact duplicate (same SHA-256)
    _w("temp_draft_v1_copy.py", draft_content, recent_ai, True)

    #  AI Bloat: type-mismatch file (PNG extension, Python content) 
    _w("assets/secret.png", _SECRET_PNG_CONTENT, recent_ai, True)

    #  Human files (older timestamps, minimal, hand-crafted feel) 
    _w("README.md", _README, old_human, False)
    _w("notes.txt", _NOTES_TXT, now - 7 * 86400, False)
    _w("requirements.txt", _REQUIREMENTS_TXT, now - 5 * 86400, False)

    # Record ground truth for directories too (use top-level dir path)
    for d in [".claude", ".github", "venv", "__pycache__", "node_modules"]:
        ground_truth[d] = True

    return ground_truth


# ---------------------------------------------------------------------------
# Fingerprint builder
# ---------------------------------------------------------------------------

def _build_fingerprint(
    path: Path,
    workspace: Path,
    hash_registry: Dict[str, List[str]],
) -> FileFingerprint:
    """Compute a full FileFingerprint for *path* (relative to *workspace*)."""
    rel = str(path.relative_to(workspace)).replace("\\", "/")
    stat = path.stat()
    is_dir = path.is_dir()
    ext = path.suffix.lower() if not is_dir else ""

    if is_dir:
        size = _dir_size(path)
        child_count = len(list(path.iterdir()))
        sha256, magic_hex, declared_type, detected_type = "", "", "", ""
        type_mismatch = False
    else:
        size = stat.st_size
        child_count = 0
        sha256 = _sha256(path)
        magic_hex = _magic_header(path)
        declared_type = EXTENSION_TYPES.get(ext, "")
        detected_type = _detect_type(magic_hex)
        # Mismatch: extension says binary-image/exe but content says text or different binary
        type_mismatch = bool(
            declared_type
            and declared_type not in ("text", "markdown", "json", "yaml", "config", "unknown")
            and detected_type == "text"
            and declared_type in ("png", "jpeg", "gif", "pdf", "windows_exe", "elf_binary", "zip")
        )
        if sha256:
            hash_registry[sha256].append(rel)

    signals: List[AISignal] = []
    name_lower = path.name.lower()

    # Signal: known AI artifact directory / name
    if name_lower in AI_DIR_SIGNALS:
        sig_type, desc, conf = AI_DIR_SIGNALS[name_lower]
        signals.append(AISignal(signal_type=sig_type, description=desc, confidence=conf))

    # Signal: temp/draft naming
    for pat in TEMP_PATTERNS:
        if pat in name_lower:
            signals.append(AISignal(
                signal_type="TEMP_DRAFT",
                description=f"Filename '{path.name}' matches temp/draft pattern '{pat}'",
                confidence=0.75,
            ))
            break

    # Signal: Python bytecode
    if ext == ".pyc":
        signals.append(AISignal(
            signal_type="BYTECODE_ARTIFACT",
            description="Compiled Python bytecode  always auto-generated, never hand-written",
            confidence=0.99,
        ))

    # Signal: pyvenv.cfg
    if path.name == "pyvenv.cfg":
        signals.append(AISignal(
            signal_type="VIRTUALENV_INTERNAL",
            description="pyvenv.cfg is an internal virtual-environment manifest, indicating an AI-spawned venv",
            confidence=0.90,
        ))

    # Signal: type mismatch
    if type_mismatch:
        signals.append(AISignal(
            signal_type="TYPE_MISMATCH",
            description=(
                f"Extension '{ext}' ({declared_type}) contradicts detected content type "
                f"'{detected_type}'  possible payload disguised as media"
            ),
            confidence=0.94,
        ))

    # Signal: AI scaffold naming (low weight on its own  strengthened by BATCH_CREATION later)
    if not is_dir and path.stem.lower() in AI_SCAFFOLD_STEMS:
        signals.append(AISignal(
            signal_type="AI_SCAFFOLD_NAME",
            description=(
                f"'{path.name}' is a canonical boilerplate filename found in AI-generated scaffolds"
            ),
            confidence=0.50,
        ))

    # Composite probability
    if signals:
        max_conf = max(s.confidence for s in signals)
        avg_conf = sum(s.confidence for s in signals) / len(signals)
        prob = min(0.99, max_conf * 0.65 + avg_conf * 0.35)
    else:
        prob = 0.05

    return FileFingerprint(
        path=rel,
        is_directory=is_dir,
        size_bytes=size,
        child_count=child_count,
        extension=ext,
        ctime=stat.st_ctime,
        mtime=stat.st_mtime,
        atime=stat.st_atime,
        sha256_hash=sha256,
        magic_header=magic_hex,
        declared_type=declared_type,
        detected_type=detected_type,
        type_mismatch=type_mismatch,
        ai_signals=signals,
        ai_probability=prob,
    )


def _enrich_batch_and_duplicates(
    fingerprints: List[FileFingerprint],
    hash_registry: Dict[str, List[str]],
) -> None:
    """
    Post-process fingerprints to add cross-file signals:
    BATCH_CREATION (many files with identical mtime) and
    DUPLICATE_CONTENT (identical SHA-256 hashes).
    These can only be computed once all files are fingerprinted.
    """
    # Batch creation: group non-directory files by rounded mtime
    mtime_groups: Dict[int, List[FileFingerprint]] = defaultdict(list)
    fp_by_path: Dict[str, FileFingerprint] = {fp.path: fp for fp in fingerprints}

    for fp in fingerprints:
        if not fp.is_directory:
            mtime_groups[round(fp.mtime)].append(fp)

    for ts, group in mtime_groups.items():
        if len(group) >= 3:
            for fp in group:
                fp.ai_signals.append(AISignal(
                    signal_type="BATCH_CREATION",
                    description=(
                        f"One of {len(group)} files all sharing the same modification timestamp "
                        f" characteristic of AI agent batch-scaffolding"
                    ),
                    confidence=0.82,
                ))
                fp.ai_probability = min(0.99, fp.ai_probability + 0.15)

    # Duplicate content: same SHA-256 across different paths
    for sha, paths in hash_registry.items():
        if len(paths) >= 2 and sha:
            for p in paths:
                fp = fp_by_path.get(p)
                if fp:
                    fp.ai_signals.append(AISignal(
                        signal_type="DUPLICATE_CONTENT",
                        description=(
                            f"Identical SHA-256 content found in {len(paths)} files  "
                            f"AI agents frequently copy boilerplate without renaming"
                        ),
                        confidence=0.85,
                    ))
                    fp.ai_probability = min(0.99, fp.ai_probability + 0.10)


# ---------------------------------------------------------------------------
# Queue builder  what items does the agent actually see?
# ---------------------------------------------------------------------------

def _build_queue(workspace: Path, ground_truth: Dict[str, bool]) -> List[Tuple[FileFingerprint, bool]]:
    """
    Build a randomised queue of (FileFingerprint, is_ai_bloat) pairs.

    We expose:
      - Top-level directories that are bloat in their entirety
      - Top-level files
      - Files inside 'src/' (batch-created scaffold)
      - The disguised PNG in 'assets/'

    We do NOT recurse into every file inside node_modules/venv/etc. 
    the agent acts at the top-level directory granularity for those.
    """
    TOP_LEVEL_DIRS = {".cursorrules", ".claude", ".github", "venv", "__pycache__", "node_modules"}
    hash_registry: Dict[str, List[str]] = defaultdict(list)
    queue_items: List[Tuple[FileFingerprint, bool]] = []

    # Top-level entries
    for entry in sorted(workspace.iterdir()):
        rel = entry.name
        if rel in TOP_LEVEL_DIRS:
            fp = _build_fingerprint(entry, workspace, hash_registry)
            is_bloat = ground_truth.get(rel, False)
            queue_items.append((fp, is_bloat))
        elif entry.is_dir():
            # Recurse one level into non-bloat dirs (src/, assets/)
            for child in sorted(entry.iterdir()):
                child_rel = str(child.relative_to(workspace)).replace("\\", "/")
                fp = _build_fingerprint(child, workspace, hash_registry)
                is_bloat = ground_truth.get(child_rel, False)
                queue_items.append((fp, is_bloat))
        else:
            child_rel = entry.name
            fp = _build_fingerprint(entry, workspace, hash_registry)
            is_bloat = ground_truth.get(child_rel, False)
            queue_items.append((fp, is_bloat))

    # Enrich with cross-file signals
    all_fps = [item[0] for item in queue_items]
    _enrich_batch_and_duplicates(all_fps, hash_registry)

    # Randomise order so the agent cannot exploit position
    random.shuffle(queue_items)
    return queue_items


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

REWARD_TRUE_POSITIVE  = +1.00   # delete + AI bloat
REWARD_FALSE_POSITIVE = -2.00   # delete + human file
REWARD_FLAG_TP        = +0.40   # flag   + AI bloat
REWARD_FLAG_FP        = -0.40   # flag   + human file
REWARD_TRUE_NEGATIVE  = +0.30   # skip   + human file
REWARD_FALSE_NEGATIVE = -0.30   # skip   + AI bloat
F1_BONUS_MAX          = +3.00   # maximum terminal bonus for perfect F1

# Small process-quality shaping: rewards how well an action matches confidence.
PROCESS_ALIGNMENT_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AiBloatDetectorEnvironment(Environment):
    """
    RL environment for training an agent to detect and delete AI-generated
    digital bloat from a developer workspace.

    Each episode:
      1. A fresh synthetic workspace is created in a temp directory.
      2. The agent receives one FileFingerprint per step and must classify it.
      3. The episode ends when the queue is exhausted or the agent sends "done".
      4. A terminal F1-bonus is applied on the final step.

    Example::

        env = AiBloatDetectorEnvironment()
        obs = env.reset()
        while not obs.done:
            # Rule-based oracle: delete if ai_probability > 0.6
            action_type = "delete" if obs.current_item.ai_probability > 0.6 else "skip"
            obs = env.step(BloatAction(action_type=action_type))
        print(obs.episode_summary)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._workspace: Optional[Path] = None
        self._queue: List[Tuple[FileFingerprint, bool]] = []
        self._queue_index: int = 0
        self._ground_truth: Dict[str, bool] = {}

        # Running counters
        self._tp = self._fp = self._tn = self._fn = 0
        self._bytes_freed = 0
        self._history: List[Dict] = []
        self._reward_total: float = 0.0
        self._process_score_total: float = 0.0
        self._process_steps: int = 0

    #  Lifecycle 

    def reset(self) -> BloatObservation:
        # Tear down previous workspace
        if self._workspace and self._workspace.exists():
            shutil.rmtree(self._workspace, ignore_errors=True)

        self._workspace = Path(tempfile.mkdtemp(prefix="bloat_workspace_"))
        self._ground_truth = _plant_workspace(self._workspace)
        self._queue = _build_queue(self._workspace, self._ground_truth)
        self._queue_index = 0
        self._tp = self._fp = self._tn = self._fn = 0
        self._bytes_freed = 0
        self._history = []
        self._reward_total = 0.0
        self._process_score_total = 0.0
        self._process_steps = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)

        first_fp, _ = self._queue[0] if self._queue else (None, False)
        return BloatObservation(
            current_item=first_fp,
            queue_size=len(self._queue),
            step_count=0,
            done=False,
            reward=0.0,
            last_action_result="Environment reset. Workspace ready for forensic scan.",
        )

    def step(self, action: BloatAction) -> BloatObservation:  # type: ignore[override]
        self._state.step_count += 1

        #  Early termination 
        if action.action_type == "done" or self._queue_index >= len(self._queue):
            return self._terminal_obs(triggered_by_done=True)

        fp, is_bloat = self._queue[self._queue_index]
        self._queue_index += 1
        act = action.action_type

        #  Compute reward 
        base_reward, result_msg = self._compute_reward(act, fp, is_bloat)
        process_bonus, process_msg = self._process_reward(act, fp)
        reward = round(base_reward + process_bonus, 4)
        self._reward_total = round(self._reward_total + reward, 4)
        self._process_score_total = round(self._process_score_total + process_bonus, 4)
        self._process_steps += 1
        result_msg = f"{result_msg} Process({process_bonus:+.2f}): {process_msg}"

        #  Apply action to disk 
        if act == "delete" and self._workspace:
            target = self._workspace / fp.path
            freed = fp.size_bytes
            try:
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                elif target.exists():
                    target.unlink()
                self._bytes_freed += freed
            except OSError:
                pass

        #  Record history 
        self._history.append({
            "path": fp.path,
            "action": act,
            "is_bloat": is_bloat,
            "reward": reward,
            "ai_probability": fp.ai_probability,
        })

        #  Advance queue 
        done = self._queue_index >= len(self._queue)
        if done:
            return self._terminal_obs(extra_reward=reward, last_msg=result_msg)

        next_fp, _ = self._queue[self._queue_index]
        prec, rec, f1 = self._metrics()

        return BloatObservation(
            current_item=next_fp,
            queue_size=len(self._queue) - self._queue_index,
            step_count=self._state.step_count,
            bytes_freed=self._bytes_freed,
            true_positives=self._tp,
            false_positives=self._fp,
            true_negatives=self._tn,
            false_negatives=self._fn,
            precision=prec,
            recall=rec,
            f1_score=f1,
            last_action_result=result_msg,
            done=False,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state

    #  Reward logic 

    def _compute_reward(
        self, act: str, fp: FileFingerprint, is_bloat: bool
    ) -> Tuple[float, str]:
        if act == "delete":
            if is_bloat:
                self._tp += 1
                return REWARD_TRUE_POSITIVE, (
                    f"CORRECT DELETE (+{REWARD_TRUE_POSITIVE:.2f}): '{fp.path}' was AI-generated bloat. "
                    f"{fp.size_bytes:,} bytes freed."
                )
            else:
                self._fp += 1
                return REWARD_FALSE_POSITIVE, (
                    f"FALSE POSITIVE ({REWARD_FALSE_POSITIVE:.2f}): '{fp.path}' was a human file! "
                    "Destroying real work is heavily penalised."
                )
        elif act == "flag":
            if is_bloat:
                self._tp += 1  # count as TP for precision/recall
                return REWARD_FLAG_TP, (
                    f"FLAG (+{REWARD_FLAG_TP:.2f}): '{fp.path}' is AI bloat  flagged for review."
                )
            else:
                self._fp += 1
                return REWARD_FLAG_FP, (
                    f"FALSE FLAG ({REWARD_FLAG_FP:.2f}): '{fp.path}' is a human file."
                )
        elif act == "skip":
            if not is_bloat:
                self._tn += 1
                return REWARD_TRUE_NEGATIVE, (
                    f"CORRECT SKIP (+{REWARD_TRUE_NEGATIVE:.2f}): '{fp.path}' is a human file. Good preservation."
                )
            else:
                self._fn += 1
                return REWARD_FALSE_NEGATIVE, (
                    f"MISSED BLOAT ({REWARD_FALSE_NEGATIVE:.2f}): '{fp.path}' was AI-generated bloat. "
                    f"Signals: {[s.signal_type for s in fp.ai_signals]}"
                )
        return 0.0, "No-op."

    #  Metrics 

    def _metrics(self) -> Tuple[float, float, float]:
        prec = self._tp / (self._tp + self._fp) if (self._tp + self._fp) > 0 else 0.0
        rec  = self._tp / (self._tp + self._fn) if (self._tp + self._fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return round(prec, 4), round(rec, 4), round(f1, 4)

    def _process_reward(self, act: str, fp: FileFingerprint) -> Tuple[float, str]:
        """Reward the quality of decision process, not only final correctness."""
        p = fp.ai_probability
        if p >= 0.85:
            alignment = {"delete": 1.0, "flag": 0.4, "skip": -1.0}.get(act, 0.0)
            band = "high-confidence bloat"
        elif p >= 0.60:
            alignment = {"flag": 1.0, "delete": 0.5, "skip": -0.5}.get(act, 0.0)
            band = "uncertain / review band"
        else:
            alignment = {"skip": 1.0, "flag": 0.2, "delete": -1.0}.get(act, 0.0)
            band = "likely human"

        # Be slightly conservative on directory deletion unless confidence is very high.
        if fp.is_directory and act == "delete" and p < 0.95:
            alignment -= 0.3

        bonus = round(max(-1.0, min(1.0, alignment)) * PROCESS_ALIGNMENT_WEIGHT, 4)
        return bonus, f"action='{act}' in {band} (ai_probability={p:.2f})"

    #  Terminal observation 

    def _terminal_obs(
        self,
        triggered_by_done: bool = False,
        extra_reward: float = 0.0,
        last_msg: str = "",
    ) -> BloatObservation:
        prec, rec, f1 = self._metrics()
        f1_bonus = round(F1_BONUS_MAX * f1, 4)
        total_reward = extra_reward + f1_bonus
        process_avg = round(self._process_score_total / max(self._process_steps, 1), 4)
        reward_total_with_bonus = round(self._reward_total + f1_bonus, 4)

        total_bloat = sum(1 for _, bloat in self._queue if bloat)
        summary = {
            "triggered_by": "done_action" if triggered_by_done else "queue_exhausted",
            "steps": self._state.step_count,
            "total_items": len(self._queue),
            "total_ai_bloat_items": total_bloat,
            "true_positives": self._tp,
            "false_positives": self._fp,
            "true_negatives": self._tn,
            "false_negatives": self._fn,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "f1_bonus": f1_bonus,
            "bytes_freed": self._bytes_freed,
            "process_score_total": self._process_score_total,
            "process_score_avg": process_avg,
            "reward_base_total": self._reward_total,
            "reward_total": reward_total_with_bonus,
            "history": self._history,
        }

        term_msg = (
            f"Episode complete. F1={f1:.3f} | "
            f"Precision={prec:.3f} | Recall={rec:.3f} | "
            f"Bytes freed={self._bytes_freed:,} | "
            f"F1 bonus=+{f1_bonus:.2f}"
        )

        return BloatObservation(
            current_item=None,
            queue_size=0,
            step_count=self._state.step_count,
            bytes_freed=self._bytes_freed,
            true_positives=self._tp,
            false_positives=self._fp,
            true_negatives=self._tn,
            false_negatives=self._fn,
            precision=prec,
            recall=rec,
            f1_score=f1,
            last_action_result=last_msg or term_msg,
            done=True,
            reward=total_reward,
            episode_summary=summary,
        )


# ---------------------------------------------------------------------------
# Smoke test  run directly to verify the environment works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== AI Digital Bloat Detector  Smoke Test ===\n")
    env = AiBloatDetectorEnvironment()
    obs = env.reset()

    print(f"Queue size: {obs.queue_size}")
    print(f"First item: {obs.current_item.path if obs.current_item else 'None'}")
    print()

    step = 0
    while not obs.done:
        step += 1
        item = obs.current_item
        # Oracle: trust the AI probability score
        if item.ai_probability >= 0.60:
            act = BloatAction(action_type="delete")
        elif item.ai_probability >= 0.40:
            act = BloatAction(action_type="flag")
        else:
            act = BloatAction(action_type="skip")

        print(f"[{step:02d}] {act.action_type:6s} | p={item.ai_probability:.2f} | {item.path}")
        obs = env.step(act)
        print(f"       -> {obs.last_action_result[:80]}")

    print("\n=== Episode Summary ===")
    if obs.episode_summary:
        s = obs.episode_summary
        print(f"  F1       : {s['f1_score']:.3f}")
        print(f"  Precision: {s['precision']:.3f}")
        print(f"  Recall   : {s['recall']:.3f}")
        print(f"  TP/FP/TN/FN: {s['true_positives']}/{s['false_positives']}/{s['true_negatives']}/{s['false_negatives']}")
        print(f"  Bytes freed: {s['bytes_freed']:,}")
        print(f"  F1 bonus   : +{s['f1_bonus']:.2f}")
