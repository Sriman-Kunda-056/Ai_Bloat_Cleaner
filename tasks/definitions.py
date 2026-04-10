# tasks/definitions.py
# Pre-computed synthetic file observations for grading — no live server needed.

# Each scenario: list of file_meta dicts with known labels.
# Fields match AiBloatDetectorEnvironment's file_meta schema.

TASKS = {
    "precision": {
        "description": (
            "Minimise false positives: only delete files with very high AI signature. "
            "Correct action: delete only obvious bloat (ai_signature > 0.8, label='bloat')."
        ),
        "files": [
            {
                "path": "/tmp/ai_gen/report_v1.txt",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 2.5,
                "ai_signature": 0.92,
                "content_hash_group": "unique",
                "ideal_action": "delete",
            },
            {
                "path": "/home/user/notes.md",
                "label": "human",
                "is_critical": False,
                "size_mb": 0.1,
                "ai_signature": 0.15,
                "content_hash_group": "unique",
                "ideal_action": "keep",
            },
            {
                "path": "/tmp/ai_cache/output_42.json",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 5.0,
                "ai_signature": 0.88,
                "content_hash_group": "unique",
                "ideal_action": "delete",
            },
            {
                "path": "/etc/hosts",
                "label": "human",
                "is_critical": True,
                "size_mb": 0.01,
                "ai_signature": 0.05,
                "content_hash_group": "unique",
                "ideal_action": "keep",
            },
            {
                "path": "/tmp/ai_gen/summary.txt",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 1.2,
                "ai_signature": 0.95,
                "content_hash_group": "cluster_a",
                "ideal_action": "delete",
            },
        ],
    },

    "recall": {
        "description": (
            "Minimise false negatives: catch all bloat files. "
            "Correct action: delete every file with label='bloat'."
        ),
        "files": [
            {
                "path": "/tmp/ai_outputs/draft_001.txt",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 3.0,
                "ai_signature": 0.85,
                "content_hash_group": "unique",
                "ideal_action": "delete",
            },
            {
                "path": "/tmp/ai_outputs/draft_002.txt",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 3.0,
                "ai_signature": 0.83,
                "content_hash_group": "cluster_b",
                "ideal_action": "delete",
            },
            {
                "path": "/var/log/syslog",
                "label": "human",
                "is_critical": False,
                "size_mb": 1.5,
                "ai_signature": 0.10,
                "content_hash_group": "unique",
                "ideal_action": "keep",
            },
            {
                "path": "/tmp/ai_gen/bulk_export.csv",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 8.0,
                "ai_signature": 0.91,
                "content_hash_group": "unique",
                "ideal_action": "delete",
            },
            {
                "path": "/home/user/project/main.py",
                "label": "human",
                "is_critical": False,
                "size_mb": 0.05,
                "ai_signature": 0.20,
                "content_hash_group": "unique",
                "ideal_action": "keep",
            },
            {
                "path": "/tmp/ai_gen/low_conf.txt",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 0.5,
                "ai_signature": 0.72,
                "content_hash_group": "unique",
                "ideal_action": "delete",
            },
        ],
    },

    "f1_score": {
        "description": (
            "Balance precision and recall for an optimal F1 score. "
            "Mix of obvious bloat, marginal bloat, and human files."
        ),
        "files": [
            {
                "path": "/tmp/ai_cache/high_conf.bin",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 10.0,
                "ai_signature": 0.94,
                "content_hash_group": "unique",
                "ideal_action": "delete",
            },
            {
                "path": "/home/user/config.yaml",
                "label": "human",
                "is_critical": False,
                "size_mb": 0.02,
                "ai_signature": 0.12,
                "content_hash_group": "unique",
                "ideal_action": "keep",
            },
            {
                "path": "/tmp/ai_cache/medium_conf.txt",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 4.0,
                "ai_signature": 0.75,
                "content_hash_group": "unique",
                "ideal_action": "delete",
            },
            {
                "path": "/usr/local/bin/script.sh",
                "label": "human",
                "is_critical": False,
                "size_mb": 0.01,
                "ai_signature": 0.08,
                "content_hash_group": "unique",
                "ideal_action": "keep",
            },
            {
                "path": "/tmp/ai_cache/dup_a.json",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 2.0,
                "ai_signature": 0.90,
                "content_hash_group": "cluster_c",
                "ideal_action": "delete",
            },
            {
                "path": "/home/user/data.csv",
                "label": "human",
                "is_critical": False,
                "size_mb": 0.8,
                "ai_signature": 0.30,
                "content_hash_group": "unique",
                "ideal_action": "keep",
            },
            {
                "path": "/tmp/ai_gen/marginal.txt",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 0.3,
                "ai_signature": 0.82,
                "content_hash_group": "unique",
                "ideal_action": "delete",
            },
        ],
    },

    "efficiency": {
        "description": (
            "Maximise bytes freed per step under a limited budget. "
            "Prefer deleting large bloat files first."
        ),
        "files": [
            {
                "path": "/tmp/ai_bulk/large_export.bin",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 50.0,
                "ai_signature": 0.93,
                "content_hash_group": "unique",
                "ideal_action": "delete",
            },
            {
                "path": "/tmp/ai_bulk/medium_export.bin",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 20.0,
                "ai_signature": 0.89,
                "content_hash_group": "unique",
                "ideal_action": "delete",
            },
            {
                "path": "/home/user/important.doc",
                "label": "human",
                "is_critical": False,
                "size_mb": 0.5,
                "ai_signature": 0.18,
                "content_hash_group": "unique",
                "ideal_action": "keep",
            },
            {
                "path": "/tmp/ai_bulk/small_bloat.txt",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 0.05,
                "ai_signature": 0.86,
                "content_hash_group": "unique",
                "ideal_action": "delete",
            },
            {
                "path": "/tmp/ai_bulk/large_bloat_2.csv",
                "label": "bloat",
                "is_critical": False,
                "size_mb": 35.0,
                "ai_signature": 0.91,
                "content_hash_group": "unique",
                "ideal_action": "delete",
            },
        ],
    },
}

TASK_NAMES = list(TASKS.keys())
