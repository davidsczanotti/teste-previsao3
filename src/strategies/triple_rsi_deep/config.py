from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepTripleRsiConfig:
    """
    Configuration for a PPO-based trading agent using Multi-Timeframe Analysis.
    
    The agent operates on a primary interval (e.g., 1m) and uses features from a 
    longer 'trend' interval (e.g., 15m) to provide trend context.
    """

    symbol: str = "BTCUSDT"
    interval: str = "1m"
    days: int = 365  # Use a larger dataset for more robust training

    # --- Feature Engineering ---
    # Primary interval features
    rsi_periods: tuple[int, ...] = (14, 21, 33)
    stoch_period: int = 14
    stoch_upper: float = 80.0
    stoch_lower: float = 20.0
    
    # Trend interval features (Multi-Timeframe Analysis)
    trend_interval: str = "15m"
    trend_adx_period: int = 14

    # --- Trading & Risk Management ---
    lot_size: float = 0.001
    fee_rate: float = 0.00075  # More realistic fee for a high-volume trader
    slippage_bps: float = 0.5  # 0.5 bps = 0.00005
    action_cost_open: float = 0.1  # Reduced penalty, let the model learn churn
    action_cost_close: float = 0.1
    invalid_action_penalty: float = 1.0  # Strong penalty for invalid actions
    min_hold_bars: int = 4
    reopen_cooldown_bars: int = 4
    max_position_bars: Optional[int] = 240  # e.g., 4 hours on 1m candles
    long_only: bool = True

    # --- Reward System ---
    # The primary reward will be based on a risk-adjusted metric like Sharpe Ratio,
    # calculated at the end of the episode. These weights are for step-by-step rewards.
    reward_pnl_weight: float = 1.0  # Weight for realized PnL
    reward_m2m_weight: float = 0.0  # Mark-to-market PnL (can be noisy, start with 0)

    # --- Episode & Data Handling ---
    episode_len: int = 4096  # Longer episodes for more context
    random_start: bool = True
    train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15)

    # --- PPO Algorithm Parameters ---
    hidden_size: int = 64  # Increased complexity for more features
    gamma: float = 0.995  # Discount factor for future rewards
    learning_rate: float = 1e-4
    entropy_beta: float = 0.01  # Higher entropy bonus to encourage exploration
    grad_clip: float = 1.0
    normalize_advantages: bool = True
    ppo_clip_epsilon: float = 0.2  # PPO-specific clipping parameter
    ppo_epochs: int = 10  # Number of optimization epochs per data batch
    ppo_batch_size: int = 64

    # --- Training ---
    seed: int = 42
    episodes: int = 200  # More training episodes
    max_steps: Optional[int] = None
