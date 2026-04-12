"""
FastAPI server for the AI Bloat Detector environment.

Endpoints:
    GET  /          health check
    POST /reset     reset the environment
    POST /step      send an action, receive next observation
    GET  /state     current environment state (metrics)
"""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
    from ..models import BloatAction, BloatObservation
    from .environment import AiBloatDetectorEnvironment

    app = create_app(
        AiBloatDetectorEnvironment,
        BloatAction,
        BloatObservation,
        env_name="ai_bloat_detector",
        max_concurrent_envs=1,
    )

except Exception:
    # Fallback: minimal FastAPI app so the file is always importable
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    try:
        from ..models import BloatAction, BloatObservation
        from .environment import AiBloatDetectorEnvironment
    except ImportError:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from models import BloatAction, BloatObservation
        from server.environment import AiBloatDetectorEnvironment

    app = FastAPI(title="AI Bloat Detector")
    _env = AiBloatDetectorEnvironment()

    @app.get("/")
    def health():
        return {"status": "ok", "env": "ai_bloat_detector"}

    @app.post("/reset")
    def reset():
        obs = _env.reset()
        return obs.model_dump()

    @app.post("/step")
    def step(action: BloatAction):
        obs = _env.step(action)
        return obs.model_dump()

    @app.get("/state")
    def state():
        return dict(_env.state)


def main():
    import os
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
