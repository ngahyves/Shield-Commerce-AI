"""Application config loader. Reads config.yaml.

Independent from the logging config (configs/logging.yaml).
"""

from pathlib import Path
from typing import Any
import yaml


class ConfigLoader:
    """Load and access the application configuration from YAML.

    Single access method: `get(*keys, default=None)`.
    Works for top-level and nested keys.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config_path = Path(config_path)
        self._data = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load the YAML config securely."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path.resolve()}"
            )
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if data is not None else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in config file: {e}") from e

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a config value at any depth.
        Args:
            *keys: Path to the value (one key = top-level, multiple = nested).
            default: Value returned if the path is missing.
        Examples:
            cfg.get("app")                    # dict at top level
            cfg.get("thresholds", "block")    # 0.37
            cfg.get("model", "params")        # dict of hyperparams
            cfg.get("missing", default=42)    # 42
        """
        val: Any = self._data
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

