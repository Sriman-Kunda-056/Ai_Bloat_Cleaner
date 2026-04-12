"""Typed client for the AI Bloat Detector environment."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Optional

try:
    from openenv.core.env_client.client import EnvClient

    class AiBloatDetector(EnvClient):
        """Async client for AiBloatDetectorEnvironment."""
        pass

except ImportError:
    # Minimal synchronous client used when openenv is not installed
    class AiBloatDetector:  # type: ignore[no-redef]
        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url.rstrip("/")

        def _post(self, path: str, data: Optional[dict] = None) -> dict:
            body = json.dumps(data or {}).encode("utf-8")
            req  = urllib.request.Request(
                f"{self.base_url}{path}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _get(self, path: str) -> dict:
            with urllib.request.urlopen(
                f"{self.base_url}{path}", timeout=10
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def health(self) -> dict:
            return self._get("/")

        def reset(self) -> dict:
            return self._post("/reset", {})

        def step(self, action_type: str) -> dict:
            return self._post("/step", {"action_type": action_type})

        def state(self) -> dict:
            return self._get("/state")
