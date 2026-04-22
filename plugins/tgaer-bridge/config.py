"""Configuration loader for the tgaer-bridge plugin.

Reads from hermes config.yaml under ``plugins.tgaer-bridge``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "output_dir": "~/.hermes/tgaer-bridge",
    "max_result_chars": 4096,
    "exclude_tools": [],
    "reward_strategy": "terminal_only",
    "score_weights": {
        "completion": 0.5,
        "tool_success": 0.3,
        "efficiency": 0.2,
    },
    "min_steps_to_export": 1,
}


def _hermes_home() -> Path:
    return Path.home() / ".hermes"


def load_config() -> Dict[str, Any]:
    """Load plugin config, merging defaults with user overrides."""
    config = dict(DEFAULT_CONFIG)
    config_path = _hermes_home() / "config.yaml"
    if not config_path.exists():
        return config

    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("tgaer-bridge: failed to read config.yaml: %s", exc)
        return config

    plugin_cfg = raw.get("plugins", {}).get("tgaer-bridge", {})
    if not isinstance(plugin_cfg, dict):
        return config

    for key, value in plugin_cfg.items():
        if key == "score_weights" and isinstance(value, dict):
            config["score_weights"] = {**config["score_weights"], **value}
        else:
            config[key] = value

    return config


def output_dir(config: Dict[str, Any]) -> Path:
    """Resolve and ensure the output directory exists."""
    p = Path(config.get("output_dir", DEFAULT_CONFIG["output_dir"])).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def episodes_dir(config: Dict[str, Any]) -> Path:
    """Resolve and ensure the episodes subdirectory exists."""
    p = output_dir(config) / "episodes"
    p.mkdir(parents=True, exist_ok=True)
    return p
