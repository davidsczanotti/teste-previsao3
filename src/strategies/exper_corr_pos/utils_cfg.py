from __future__ import annotations

import math
from typing import List, Dict, Any

import pandas as pd

from .models import MoEPolicy


DEFAULT_NAMES = ["TrendML", "MultiFrame", "Spread", "Pattern"]


def get_all_expert_names(cfg: Dict[str, Any]) -> List[str]:
    names = cfg.get("model", {}).get("expert_names")
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    return DEFAULT_NAMES.copy()


def get_enable_map(cfg: Dict[str, Any]) -> Dict[str, bool]:
    model = cfg.get("model", {})
    # suportar chaves alternativas solicitadas pelo usuário (pt-br)
    enable = model.get("expert_enable") or model.get("experts") or model.get("especialistas") or {}
    if isinstance(enable, dict):
        norm = {str(k): bool(v) for k, v in enable.items()}
    else:
        norm = {}
    return norm


def enabled_expert_names(cfg: Dict[str, Any]) -> List[str]:
    names = get_all_expert_names(cfg)
    enable = get_enable_map(cfg)
    if not enable:
        return names
    return [n for n in names if enable.get(n, True)]


def build_policy(input_dim: int, cfg: Dict[str, Any]) -> MoEPolicy:
    model_cfg = cfg.get("model", {})
    names = enabled_expert_names(cfg)
    num_experts = max(1, len(names))

    policy = MoEPolicy(
        input_dim=input_dim,
        num_actions=3,
        expert_hidden=model_cfg.get("expert_hidden", [64, 32]),
        gating_hidden=model_cfg.get("gating_hidden", [64, 32]),
        num_experts=num_experts,
        temperature=model_cfg.get("temperature", 1.6),
        top_k=model_cfg.get("top_k", 3),
    )
    return policy


def timeframe_to_timedelta(tf: str | None) -> pd.Timedelta:
    tf = str(tf or "1h").strip()
    try:
        return pd.to_timedelta(tf)
    except ValueError:
        mapping = {
            "1min": "1T",
            "5min": "5T",
            "15min": "15T",
            "30min": "30T",
            "60min": "1H",
            "1m": "1T",
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "60m": "1H",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
            "1w": "1W",
            "1W": "1W",
        }
        alias = mapping.get(tf.lower())
        if alias:
            return pd.to_timedelta(alias)
        raise


def bars_for_days(timeframe: str | None, days: int) -> int:
    delta = timeframe_to_timedelta(timeframe)
    if delta <= pd.Timedelta(0):
        return max(1, int(days))
    total = pd.to_timedelta(days, unit="D")
    bars = total / delta
    if bars <= 1:
        return 1
    return int(math.ceil(float(bars)))


def hours_per_bar(timeframe: str | None) -> float:
    delta = timeframe_to_timedelta(timeframe)
    return float(delta / pd.Timedelta(hours=1))
