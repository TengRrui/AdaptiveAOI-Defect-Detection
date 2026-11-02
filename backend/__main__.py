"""Module entry point for ``python -m backend``."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the AdaptiveAOI integration API.")
    parser.add_argument("--config", default=None, help="Path to configs/system_integration.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # Set the requested configuration before importing ``backend.app``.  That
    # module exposes one ASGI application; importing it and then calling
    # create_app() again previously built every service and resource index
    # twice on every formal launch.
    if args.config:
        os.environ["ADAPTIVEAOI_SYSTEM_CONFIG"] = str(args.config)
    from .app import app as application
    from .config import IntegrationSettings

    settings = IntegrationSettings.load(args.config)
    uvicorn.run(
        application,
        host=args.host or settings.host(),
        port=args.port or settings.port(),
        reload=args.reload,
        log_level=str(settings.runtime.get("log_level", "info")).lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
