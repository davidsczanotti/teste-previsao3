from __future__ import annotations

from typing import List, Dict, Any

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

