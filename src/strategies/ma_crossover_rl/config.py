from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MaCrossoverBacktestConfig:
    ticker: str = "BTCUSDT"
    interval: str = "15m"
    days: int = 120
    lot_size: float = 0.001
    fee_rate: float = 0.001  # 0.1% per side
    min_hold_bars: int = 0   # minimum bars to hold before allowing exit
    cooldown_bars: int = 0   # bars to wait after closing before re-entry
    allow_short: bool = True
    ma_short_window: int = 7
    ma_mid_window: int = 40
    ma_long_window: int = 120
    ma_type: str = "sma"


@dataclass
class MaCrossoverRlConfig:
    # Data/env
    ticker: str = "BTCUSDT"
    interval: str = "15m"
    days: int = 3650  # usar o máximo possível (~10 anos se disponível)
    lot_size: float = 0.001
    fee_rate: float = 0.001
    slippage_bps: float = 0.0
    action_cost_open: float = 0.0
    action_cost_close: float = 0.0
    min_hold_bars: int = 0
    reopen_cooldown_bars: int = 0
    max_position_bars: int | None = None
    long_only: bool = False
    m2m_weight: float = 0.05
    ma_short_window: int = 7
    ma_mid_window: int = 40
    ma_long_window: int = 120
    ma_type: str = "sma"
    exit_only: bool = False
    episode_len: int | None = 4096
    random_start: bool = True
    # idle penalty shaping
    idle_penalty: float = 0.0
    idle_grace_bars: int = 0
    idle_ramp: float = 0.0
    # execution/anti-churn/normalization
    exec_next_open: bool = True
    switch_penalty: float = 0.0
    switch_window_bars: int = 5
    reward_atr_norm: bool = False
    atr_period: int = 14
    gate_on_heuristic: bool = False

    # Model/opt
    seed: int | None = 42
    hidden_size: int = 64
    learning_rate: float = 3e-3
    gamma: float = 0.99
    grad_clip: float = 1.0
    normalize_advantages: bool = True
    episodes: int = 100
    max_steps: int | None = None
    epsilon_start: float = 0.2
    epsilon_end: float = 0.02
    entropy_beta: float = 0.0
    entropy_beta_end: float = 0.0
    bc_weight: float = 0.0  # behavior cloning aux loss (desativado aqui)
    
