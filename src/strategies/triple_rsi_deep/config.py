from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepTripleRsiConfig:
    """Config for RL training/evaluation of Triple RSI.

    Defaults target 15m candles on BTCUSDT.
    """

    ticker: str = "BTCUSDT"
    interval: str = "15m"
    days: int = 120

    # Trading params
    lot_size: float = 0.001  # BTC amount per position
    fee_rate: float = 0.001  # 0.1%
    slippage_bps: float = 0.0  # additional slippage in basis points (1e-4)
    action_cost_open: float = 0.5  # fixed USD penalty per open (reduces churn)
    action_cost_close: float = 0.5  # fixed USD penalty per close
    invalid_action_penalty: float = 0.5  # stronger penalty for invalid actions/gating
    min_hold_bars: int = 8  # require N bars before closing
    reopen_cooldown_bars: int = 8  # wait N bars after close before opening again
    max_position_bars: Optional[int] = None  # optional max hold bars
    # Market side restriction
    long_only: bool = True  # if True, disable short actions (open/close short)

    # RSI periods (features)
    short_period: int = 33
    med_period: int = 44
    long_period: int = 115

    # Stochastic (slow) features
    stoch_period: int = 14
    stoch_upper: float = 0.75  # 75%
    stoch_lower: float = 0.25  # 25%

    # Gating around stochastic thresholds
    gate_enabled: bool = True
    gate_margin: float = 0.05  # allow opens when within margin of thresholds
    gate_recent_k: int = 3     # or if crossed threshold within last K bars

    # Reward shaping
    realized_weight: float = 1.0
    m2m_weight: float = 0.1
    midrange_penalty: float = 0.5  # penalty when opening in mid (e.g., 0.4..0.6)
    close_bonus_factor: float = 0.2  # bonus when closing near the target extreme

    # Episode control
    episode_len: Optional[int] = 2048
    random_start: bool = True

    # RL params
    hidden_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 5e-4
    entropy_beta: float = 1e-3  # small entropy bonus to encourage exploration
    entropy_beta_end: float = 1e-4
    grad_clip: float = 1.0
    normalize_advantages: bool = True
    seed: int = 42
    episodes: int = 50
    max_steps: Optional[int] = None  # by default, walk the full dataset per episode

    # Exploration schedule (epsilon-greedy applied to sampling during training)
    epsilon_start: float = 0.2
    epsilon_end: float = 0.02

    # Auxiliary imitation loss toward heuristic (0.25/0.75) – set >0 for A/B phases
    bc_weight: float = 0.0
