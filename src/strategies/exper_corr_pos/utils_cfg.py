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
        temperature=float(model_cfg.get("temperature", 1.6)),
        top_k=int(model_cfg.get("top_k", 3)),
        gating_use_attention=bool(model_cfg.get("use_attention", False)),
        attention_dim=int(model_cfg.get("attention_dim", 64)),
        attention_heads=int(model_cfg.get("attention_heads", 4)),
        attention_dropout=float(model_cfg.get("attention_dropout", 0.0)),
        attention_weight=float(model_cfg.get("attention_weight", 1.0)),
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


def merged_env_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Combine base env config with final curriculum overrides for eval/backtests."""
    base_env = dict(cfg.get("env", {}))
    curriculum = cfg.get("train", {}).get("curriculum", {}) if isinstance(cfg.get("train", {}), dict) else {}
    final_env = (curriculum.get("final") or {}).get("env") or {}
    if isinstance(final_env, dict):
        base_env.update(final_env)
    return base_env
