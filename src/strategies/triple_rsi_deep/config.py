from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import json


@dataclass
class DeepTripleRsiConfig:
    # Market / data
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    days: int = 180
    trend_intervals: List[str] = field(default_factory=lambda: ["5m", "15m", "1h"])  # Multi-timeframe context

    # Feature engineering
    rsi_periods: List[int] = field(default_factory=lambda: [6, 14, 21])
    stoch_period: int = 14
    trend_adx_period: int = 14
    trend_rsi_periods: List[int] = field(default_factory=lambda: [14])
    trend_ema_periods: List[int] = field(default_factory=lambda: [20, 50])

    # Episode / environment
    episode_len: int = 2_000
    random_start: bool = True
    long_only: bool = True
    cache_only: bool = False
    base_lot_size: float = 0.001
    lot_size: Optional[float] = None  # Backward compatibility alias
    dynamic_position_sizing: bool = False
    kelly_fraction_cap: float = 0.6
    target_atr_pct: float = 0.015  # aim for 1.5% ATR risk

    # Trading costs & execution
    fee_rate: float = 0.001
    slippage_bps: float = 1.0
    action_cost_open: float = 0.0
    action_cost_close: float = 0.0
    invalid_action_penalty: float = 0.01
    min_hold_bars: int = 5
    reopen_cooldown_bars: int = 5
    max_position_bars: Optional[int] = 120  # e.g., 2 hours on 1m

    # Reward shaping / risk
    reward_pnl_weight: float = 0.3
    reward_sharpe_weight: float = 0.4
    reward_sortino_weight: float = 0.2
    reward_calmar_weight: float = 0.1
    reward_kelly_weight: float = 0.0
    reward_profile: Optional[str] = None  # 'conservative'|'balanced'|'aggressive'
    calmar_window: int = 24 * 60 * 30  # ~30 days of 1m bars
    max_drawdown_limit: Optional[float] = None  # e.g., 0.2 => 20% max DD hard stop

    # PPO / training
    seed: int = 1337
    episodes: int = 50
    learning_rate: float = 1e-4
    gamma: float = 0.99
    ppo_clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    ppo_batch_size: int = 256
    entropy_beta: float = 0.02
    entropy_beta_start: float = 0.02
    entropy_beta_end: float = 0.005
    grad_clip: float = 0.5
    normalize_advantages: bool = True
    max_steps: Optional[int] = None

    # Architectures
    use_transformer: bool = False
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_dim: int = 64
    dropout: float = 0.1

    mlp_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    use_skip_connections: bool = True

    # Training-only overrides (for stability)
    training_relaxed_costs: bool = True
    train_fee_rate: float = 0.0
    train_slippage_bps: float = 0.0
    train_invalid_action_penalty: float = 0.0
    train_dynamic_position_sizing: bool = False
    training_simple_reward: bool = True
    simple_reward_scale: float = 0.001

    # Convenience properties for backward compatibility
    @property
    def SYMBOL(self) -> str:  # noqa: N802
        return self.symbol

    @property
    def TIMEFRAME(self) -> str:  # noqa: N802
        return self.interval


# --- Simple helpers to persist active configs for optimization results ---

def save_active_config_record(strategy_name: str, symbol: str, interval: str, best_params: Dict[str, Any], reports_dir: str = "reports") -> Path:
    out = Path(reports_dir) / "active"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{strategy_name}_{symbol}_{interval}.json"
    rec = {
        "strategy": strategy_name,
        "symbol": symbol,
        "interval": interval,
        "best_params": best_params,
    }
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# Legacy constants kept for backwards references in optimize.py/strategy.py
SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
INITIAL_CAPITAL = 1_000.0
POSITION_SIZE_PCT = 0.50
FEE = 0.001
