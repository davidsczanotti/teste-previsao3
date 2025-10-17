from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REPORTS_DIR = Path("reports")
ACTIVE_CONFIG_DIR = REPORTS_DIR / "active"


@dataclass
class AlBrooksBookConfig:
    """Config for the Al Brooks (book-style) strategy.

    Notes
    -----
    - This config encodes parameters for programmable approximations of
      Al Brooks setups (trend continuation via inside bars, H2/L2, BO-PB).
    - Cost model (fees/slippage) is applied consistently in backtest/live.
    """

    ticker: str
    interval: str
    days: int = 365
    lot_size: float = 0.1

    # Core context
    ema_fast_period: int = 20
    ema_medium_period: int = 50
    ema_slow_period: int = 200
    slope_lookback: int = 5  # for ema20 slope

    # Swings and bar classification
    swing_lookback: int = 3
    bar_body_min_pct: float = 55.0  # min body% of range to call a trend bar
    near_extreme_frac: float = 0.25  # close near high/low fraction of range

    # ATR-based filters & risk
    atr_period: int = 14
    risk_reward_ratio: float = 1.4
    atr_stop_multiplier: float = 0.0  # 0 = disabled; else overrides signal stop
    atr_trail_multiplier: float = 0.5
    min_atr: float = 0.0

    # Setup toggles
    enable_inside_trend: bool = True
    enable_h2_l2: bool = True
    enable_bo_pb: bool = True

    # BO-PB tuning
    bo_lookback: int = 20
    max_ema_distance_atr: float = 1.0  # PB proximity to EMA20 (in ATR)

    # Optional trend strength filter
    use_trend_slope: bool = True
    min_ema_slope: float = 0.0  # absolute slope threshold (price units)

    # Execution costs
    taker_fee_pct: float = 0.0004
    slippage_pct: float = 0.0005

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AlBrooksBookConfig":
        return cls(**data)


def _active_filename(ticker: str, interval: str) -> str:
    return f"ALBROOKS_BOOK_{ticker}_{interval}.json"


def save_active_config(config: AlBrooksBookConfig) -> Path:
    ACTIVE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    filepath = ACTIVE_CONFIG_DIR / _active_filename(config.ticker, config.interval)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    return filepath


def load_active_config(ticker: str, interval: str) -> Optional[AlBrooksBookConfig]:
    filepath = ACTIVE_CONFIG_DIR / _active_filename(ticker, interval)
    if not filepath.exists():
        return None
    try:
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return AlBrooksBookConfig.from_dict(data)
    except (json.JSONDecodeError, TypeError):
        return None

