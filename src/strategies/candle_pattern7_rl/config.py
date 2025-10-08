from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Candle7RlConfig:
    """Config for RL training/evaluation of Candle7 pattern strategy."""

    ticker: str = "BTCUSDT"
    interval: str = "15m"
    days: int = 3650

    # Trading params
    lot_size: float = 0.001
    fee_rate: float = 0.001
    slippage_bps: float = 0.0
    action_cost_open: float = 0.0
    action_cost_close: float = 0.0
    invalid_action_penalty: float = 0.0  # Penalidade por ação inválida
    min_hold_bars: int = 0
    reopen_cooldown_bars: int = 0
    max_position_bars: Optional[int] = None
    long_only: bool = False

    # Reward shaping
    realized_weight: float = 1.0
    m2m_weight: float = 0.05
    exec_next_open: bool = True
    switch_penalty: float = 0.0
    switch_window_bars: int = 5
    idle_penalty: float = 0.0
    idle_grace_bars: int = 0
    idle_ramp: float = 0.0
    reward_atr_norm: bool = True
    atr_period: int = 14

    # Episode control
    episode_len: Optional[int] = 2048
    random_start: bool = True
    max_steps: Optional[int] = None

    # RL params
    hidden_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    normalize_advantages: bool = True
    seed: int = 42
    episodes: int = 200
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    entropy_beta: float = 0.0
    entropy_beta_end: float = 0.0
    bc_weight: float = 0.0
    gate_on_heuristic: bool = False
