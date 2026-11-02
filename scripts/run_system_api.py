from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the AdaptiveAOI integration API.")
    parser.add_argument("--config", default=None, help="Path to configs/system_integration.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    if args.config:
        os.environ["ADAPTIVEAOI_SYSTEM_CONFIG"] = str(args.config)
    from backend.app import app
    from backend.config import IntegrationSettings

    settings = IntegrationSettings.load(args.config)
    uvicorn.run(app, host=args.host or settings.host(), port=args.port or settings.port(), reload=args.reload, log_level=str(settings.runtime.get("log_level", "info")).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
