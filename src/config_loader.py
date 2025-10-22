from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATHS = [
    Path("config/config.json"),
    Path("./config.json"),
]


def _find_config_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.getenv("APP_CONFIG")
    if env:
        return Path(env)
    for p in DEFAULT_CONFIG_PATHS:
        if p.exists():
            return p
    # Fallback to first default (even if it doesn't exist yet)
    return DEFAULT_CONFIG_PATHS[0]


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = _find_config_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_cli_defaults(command: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    return dict(cfg.get("cli", {}).get(command, {}))


def get_strategy_params(name: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    return dict(cfg.get("strategies", {}).get(name, {}))


def get_paths(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    return dict(cfg.get("paths", {}))


def get_binance(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    return dict(cfg.get("binance", {}))

