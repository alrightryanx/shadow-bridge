"""Centralized configuration for ShadowBridge web API.

Loads from ~/.shadowai/bridge_config.json with sensible defaults.
All Tier 1-4 hardcoded values are consolidated here.
"""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(str(Path.home()), ".shadowai", "bridge_config.json")

# ---- Default Values ----
DEFAULTS = {
    "api_rate_limit_rpm": 120,
    "task_lease_timeout_seconds": 900,
    "allowed_project_roots": [
        str(Path.home()),
        "C:\\shadow",
        "/c/shadow",
    ],
    "cors_origins": [
        "http://localhost:*",
        "http://127.0.0.1:*",
    ],
    "daemon_poll_interval": 30,
    "daemon_task_timeout": 600,
    "max_push_size_mb": 50,
}


def _load_config() -> dict:
    """Load config from disk, merged with defaults."""
    config = dict(DEFAULTS)
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            config.update(user_config)
            log.info(f"Loaded bridge config from {CONFIG_FILE}")
    except Exception as e:
        log.warning(f"Failed to load config from {CONFIG_FILE}: {e}")
    return config


# Module-level config loaded once at import
_config = _load_config()


def get(key: str, default=None):
    """Get a config value by key."""
    return _config.get(key, default)


def get_all() -> dict:
    """Return the full config dict."""
    return dict(_config)


def reload():
    """Reload config from disk."""
    global _config
    _config = _load_config()
