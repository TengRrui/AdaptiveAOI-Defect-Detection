from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import IntegrationSettings


def configure_logging(settings: IntegrationSettings) -> logging.Logger:
    settings.ensure_runtime_dirs()
    logger = logging.getLogger("adaptiveaoi.integration")
    if logger.handlers:
        return logger
    level = getattr(logging, str(settings.runtime.get("log_level", "INFO")).upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler = RotatingFileHandler(settings.log_root / "integration.log", maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger
