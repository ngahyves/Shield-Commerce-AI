"""Tests for ConfigLoader — reads config.yaml."""

from src.utils.logging_config import ConfigLoader


def test_get_existing_keys():
    """ConfigLoader.get returns the correct value for top-level and nested keys."""
    cfg = ConfigLoader()

    # 1. Top-level key returns a dict
    app_section = cfg.get("app")
    assert isinstance(app_section, dict)
    assert "name" in app_section

    # 2. Nested key returns a scalar
    test_start_day = cfg.get("split", "test_start_day")
    assert test_start_day == 155

    # 3. Nested key from another section
    block_threshold = cfg.get("thresholds", "block")
    assert isinstance(block_threshold, float)


def test_get_missing_key_returns_default():
    """ConfigLoader.get returns the default when the key does not exist."""
    cfg = ConfigLoader()

    # No default → returns None
    assert cfg.get("does_not_exist") is None

    # With default → returns the default value
    assert cfg.get("does_not_exist", default=42) == 42

    # Missing nested key → returns default
    assert cfg.get("does_not_exist", "nested_key", default="fallback") == "fallback"