"""Centralized logger for ShieldCommerce AI.

Configuration lives in `configs/logging.yaml` (path, level, format).
Read once and cached. Just pass the module name to instantiate:

    from utils.logger import ProjectLogger
    logger = ProjectLogger(__name__).get_logger()
    logger.info("loaded %s rows", n)
"""

import logging
import sys
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml

_DEFAULTS = {
    "level": "INFO",
    "log_file": "logs/shieldcommerce.log",
    "max_bytes": 10 * 1024 * 1024,
    "backup_count": 5,
    "format": "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
}

_CONFIG_PATH = "configs/logging.yaml"


@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Load logging config from YAML. Fall back to defaults if missing."""
    cfg = dict(_DEFAULTS)
    path = Path(_CONFIG_PATH)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in loaded.items() if v is not None})
    return cfg


class ProjectLogger:
    """Centralized logger. Just pass the module name."""

    def __init__(self, name: str) -> None:
        """Initialize the logger for a given module.

        Args:
            name: Logger name — typically `__name__`.
        """
        cfg = _load_config()
        level = getattr(logging, cfg["level"].upper(), logging.INFO)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid duplicate handlers when instantiated multiple times
        if not self.logger.handlers:
            formatter = logging.Formatter(cfg["format"], datefmt=cfg["datefmt"])

            # Console handler
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(formatter)
            self.logger.addHandler(console)

            # Rotating file handler
            log_path = Path(cfg["log_file"])
            if log_path.parent != Path(""):
                log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                cfg["log_file"],
                maxBytes=cfg["max_bytes"],
                backupCount=cfg["backup_count"],
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Return the configured logger instance."""
        return self.logger