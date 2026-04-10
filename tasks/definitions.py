"""Pre-computed synthetic task observations and signals for AI bloat grading."""

TASKS = {
    "precision": {
        "description": (
            "Minimise false positives: delete only obvious AI bloat and avoid human files."
        ),
        "ideal_action": "delete",
        "steps": [
            {
                "observation": (
                    "Workspace Scan — .cursor/rules.json:\n"
                    "FILE_KIND=directory\n"
                    "SIZE_BUCKET=directory SIZE_BYTES=4096\n"
                    "CHILD_COUNT=4\n"
                    "AGE_DAYS=0.10 IS_RECENT=true\n"
                    "TYPE_MISMATCH=false\n"
                    "AI_PROBABILITY=0.98\n"
                    "SIGNAL_COUNTS high=2 medium=1 low=0\n"
                    "Decide: DELETE, FLAG, or SKIP."
                ),
                "signals": {
                    "file_kind": "directory",
                    "ai_probability": 0.98,
                    "hidden_artifact_dir": True,
                    "recent": True,
                    "size_bytes": 4096,
                    "child_count": 4,
                    "ideal_action": "delete",
                },
            },
            {
                "observation": (
                    "Workspace Scan — README.md:\n"
                    "FILE_KIND=file\n"
                    "SIZE_BUCKET=small SIZE_BYTES=2432\n"
                    "CHILD_COUNT=0\n"
                    "AGE_DAYS=18.40 IS_RECENT=false\n"
                    "TYPE_MISMATCH=false\n"
                    "AI_PROBABILITY=0.09\n"
                    "SIGNAL_COUNTS high=0 medium=0 low=0\n"
                    "Decide: DELETE, FLAG, or SKIP."
                ),
                "signals": {
                    "file_kind": "file",
                    "ai_probability": 0.09,
                    "human_documentation": True,
                    "size_bytes": 2432,
                    "ideal_action": "skip",
                },
            },
            {
                "observation": (
                    "Workspace Scan — assets/secret.png:\n"
                    "FILE_KIND=file\n"
                    "SIZE_BUCKET=small SIZE_BYTES=8192\n"
                    "CHILD_COUNT=0\n"
                    "AGE_DAYS=0.02 IS_RECENT=true\n"
                    "TYPE_MISMATCH=true\n"
                    "AI_PROBABILITY=0.94\n"
                    "SIGNAL_COUNTS high=2 medium=1 low=0\n"
                    "Decide: DELETE, FLAG, or SKIP."
                ),
                "signals": {
                    "file_kind": "file",
                    "ai_probability": 0.94,
                    "type_mismatch": True,
                    "disguised_payload": True,
                    "size_bytes": 8192,
                    "ideal_action": "delete",
                },
            },
        ],
    },

    "recall": {
        "description": (
            "Minimise false negatives: catch every obvious AI bloat artifact."
        ),
        "ideal_action": "delete",
        "steps": [
            {
                "observation": (
                    "Workspace Scan — venv/pyvenv.cfg:\n"
                    "FILE_KIND=file\n"
                    "SIZE_BUCKET=tiny SIZE_BYTES=101\n"
                    "CHILD_COUNT=0\n"
                    "AGE_DAYS=0.03 IS_RECENT=true\n"
                    "TYPE_MISMATCH=false\n"
                    "AI_PROBABILITY=0.97\n"
                    "SIGNAL_COUNTS high=2 medium=0 low=0\n"
                    "Decide: DELETE, FLAG, or SKIP."
                ),
                "signals": {
                    "file_kind": "file",
                    "ai_probability": 0.97,
                    "virtualenv_internal": True,
                    "size_bytes": 101,
                    "ideal_action": "delete",
                },
            },
            {
                "observation": (
                    "Workspace Scan — __pycache__/main.cpython-311.pyc:\n"
                    "FILE_KIND=file\n"
                    "SIZE_BUCKET=tiny SIZE_BYTES=2048\n"
                    "CHILD_COUNT=0\n"
                    "AGE_DAYS=0.04 IS_RECENT=true\n"
                    "TYPE_MISMATCH=false\n"
                    "AI_PROBABILITY=0.99\n"
                    "SIGNAL_COUNTS high=3 medium=0 low=0\n"
                    "Decide: DELETE, FLAG, or SKIP."
                ),
                "signals": {
                    "file_kind": "file",
                    "ai_probability": 0.99,
                    "bytecode_artifact": True,
                    "size_bytes": 2048,
                    "ideal_action": "delete",
                },
            },
            {
                "observation": (
                    "Workspace Scan — node_modules/lodash/lodash.js:\n"
                    "FILE_KIND=file\n"
                    "SIZE_BUCKET=large SIZE_BYTES=512000\n"
                    "CHILD_COUNT=0\n"
                    "AGE_DAYS=0.01 IS_RECENT=true\n"
                    "TYPE_MISMATCH=false\n"
                    "AI_PROBABILITY=0.88\n"
                    "SIGNAL_COUNTS high=2 medium=1 low=0\n"
                    "Decide: DELETE, FLAG, or SKIP."
                ),
                "signals": {
                    "file_kind": "file",
                    "ai_probability": 0.88,
                    "dependency_bloat": True,
                    "size_bytes": 512000,
                    "ideal_action": "delete",
                },
            },
        ],
    },

    "f1_score": {
        "description": (
            "Balance precision and recall when the signal mix contains both obvious and marginal bloat."
        ),
        "ideal_action": "delete",
        "steps": [
            {
                "observation": (
                    "Workspace Scan — src/helpers.py:\n"
                    "FILE_KIND=file\n"
                    "SIZE_BUCKET=small SIZE_BYTES=6144\n"
                    "CHILD_COUNT=0\n"
                    "AGE_DAYS=0.12 IS_RECENT=true\n"
                    "TYPE_MISMATCH=false\n"
                    "AI_PROBABILITY=0.86\n"
                    "SIGNAL_COUNTS high=1 medium=1 low=0\n"
                    "Decide: DELETE, FLAG, or SKIP."
                ),
                "signals": {
                    "file_kind": "file",
                    "ai_probability": 0.86,
                    "ai_scaffold_name": True,
                    "batch_creation": True,
                    "size_bytes": 6144,
                    "ideal_action": "delete",
                },
            },
            {
                "observation": (
                    "Workspace Scan — notes.txt:\n"
                    "FILE_KIND=file\n"
                    "SIZE_BUCKET=small SIZE_BYTES=2048\n"
                    "CHILD_COUNT=0\n"
                    "AGE_DAYS=21.00 IS_RECENT=false\n"
                    "TYPE_MISMATCH=false\n"
                    "AI_PROBABILITY=0.18\n"
                    "SIGNAL_COUNTS high=0 medium=0 low=1\n"
                    "Decide: DELETE, FLAG, or SKIP."
                ),
                "signals": {
                    "file_kind": "file",
                    "ai_probability": 0.18,
                    "human_note": True,
                    "size_bytes": 2048,
                    "ideal_action": "skip",
                },
            },
            {
                "observation": (
                    "Workspace Scan — temp_draft_v1_copy.py:\n"
                    "FILE_KIND=file\n"
                    "SIZE_BUCKET=tiny SIZE_BYTES=1280\n"
                    "CHILD_COUNT=0\n"
                    "AGE_DAYS=0.05 IS_RECENT=true\n"
                    "TYPE_MISMATCH=false\n"
                    "AI_PROBABILITY=0.83\n"
                    "SIGNAL_COUNTS high=1 medium=1 low=0\n"
                    "Decide: DELETE, FLAG, or SKIP."
                ),
                "signals": {
                    "file_kind": "file",
                    "ai_probability": 0.83,
                    "temp_draft": True,
                    "duplicate_content": True,
                    "size_bytes": 1280,
                    "ideal_action": "delete",
                },
            },
        ],
    },

    "efficiency": {
        "description": (
            "Maximise bytes freed per step by prioritising large, high-confidence bloat first."
        ),
        "ideal_action": "delete",
        "steps": [
            {
                "observation": (
                    "Workspace Scan — node_modules/:\n"
                    "FILE_KIND=directory\n"
                    "SIZE_BUCKET=directory SIZE_BYTES=78643200\n"
                    "CHILD_COUNT=1240\n"
                    "AGE_DAYS=0.02 IS_RECENT=true\n"
                    "TYPE_MISMATCH=false\n"
                    "AI_PROBABILITY=0.95\n"
                    "SIGNAL_COUNTS high=3 medium=0 low=0\n"
                    "Decide: DELETE, FLAG, or SKIP."
                ),
                "signals": {
                    "file_kind": "directory",
                    "ai_probability": 0.95,
                    "dependency_bloat": True,
                    "size_bytes": 78643200,
                    "child_count": 1240,
                    "ideal_action": "delete",
                },
            },
            {
                "observation": (
                    "Workspace Scan — dist/:\n"
                    "FILE_KIND=directory\n"
                    "SIZE_BUCKET=directory SIZE_BYTES=31457280\n"
                    "CHILD_COUNT=312\n"
                    "AGE_DAYS=0.02 IS_RECENT=true\n"
                    "TYPE_MISMATCH=false\n"
                    "AI_PROBABILITY=0.89\n"
                    "SIGNAL_COUNTS high=2 medium=1 low=0\n"
                    "Decide: DELETE, FLAG, or SKIP."
                ),
                "signals": {
                    "file_kind": "directory",
                    "ai_probability": 0.89,
                    "build_output": True,
                    "size_bytes": 31457280,
                    "child_count": 312,
                    "ideal_action": "delete",
                },
            },
            {
                "observation": (
                    "Workspace Scan — requirements.txt:\n"
                    "FILE_KIND=file\n"
                    "SIZE_BUCKET=tiny SIZE_BYTES=512\n"
                    "CHILD_COUNT=0\n"
                    "AGE_DAYS=6.00 IS_RECENT=false\n"
                    "TYPE_MISMATCH=false\n"
                    "AI_PROBABILITY=0.14\n"
                    "SIGNAL_COUNTS high=0 medium=0 low=0\n"
                    "Decide: DELETE, FLAG, or SKIP."
                ),
                "signals": {
                    "file_kind": "file",
                    "ai_probability": 0.14,
                    "human_dependency_list": True,
                    "size_bytes": 512,
                    "ideal_action": "skip",
                },
            },
        ],
    },
}

TASK_NAMES = list(TASKS.keys())

__all__ = ["TASKS", "TASK_NAMES"]