
"""
FastAPI application for the My Env Environment.

This module creates an HTTP server that exposes the MyEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    from ..models import BloatAction, BloatObservation
    from .environment import AiBloatDetectorEnvironment
except ImportError:
    from models import BloatAction, BloatObservation
    from server.environment import AiBloatDetectorEnvironment


# Create the app directly without openenv wrapper to avoid parsing issues
app = FastAPI(title="AI Bloat Detector")
_env = AiBloatDetectorEnvironment()


@app.get("/")
def health():
    """Health check endpoint."""
    return {"status": "ok", "env": "ai_bloat_detector"}


@app.post("/reset")
def reset():
    """Reset the environment and start a new episode."""
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: BloatAction):
    """Execute one action in the environment."""
    obs = _env.step(action)
    return obs.model_dump()


@app.get("/state")
def state():
    """Get the current environment state."""
    return _env.state


# Web interface routes for Hugging Face Spaces base_path=/web
@app.get("/web")
def web_entry():
    return {
        "name": "ai_bloat_detector",
        "status": "ok",
        "message": "Space web entrypoint is available.",
        "endpoints": ["/", "/reset", "/step", "/state"],
    }


@app.get("/web/")
def web_entry_slash():
    return web_entry()


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m my_env.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn my_env.server.app:app --workers 4
    """
    import os
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
