from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import pandas_ta as ta

from ...binance_client import get_historical_klines
from .config import DeepTripleRsiConfig


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class TripleRsiEnv:
    """RL environment for a Multi-Timeframe PPO trading agent.

    Observations:
      - Cyclical time features (day_of_week, hour_of_day)
      - Primary interval indicators (RSIs, Stoch, ATR, OBV)
      - Trend interval indicators (ADX)
      - Agent state (position, bars in position)

    Actions (discrete):
      0 = Hold
      1 = Open Long
      2 = Close Long

    Reward:
      Step-wise reward is based on realized PnL, adjusted for costs.
      The primary optimization objective should be a risk-adjusted metric
      like Sharpe Ratio, calculated over an entire episode.
    """

    def __init__(self, config: DeepTripleRsiConfig, df_primary: Optional[pd.DataFrame] = None, df_trend: Optional[pd.DataFrame] = None):
        self.cfg = config

        # Runtime state
        self._df_primary: Optional[pd.DataFrame] = df_primary
        self._df_trend: Optional[pd.DataFrame] = df_trend
        self._features: Optional[np.ndarray] = None
        self._prices: Optional[np.ndarray] = None
        self._i: int = 0
        self._start_i: int = 0
        self._end_i: int = 0

        # Episode state
        self._pos: int = 0  # 0 flat, 1 long, -1 short
        self._entry_price: float = 0.0
        self._bars_in_pos: int = 0
        self._bars_since_exit: int = 1_000_000
        self.portfolio_values: List[float] = []
        self.initial_capital: float = 10000.0

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads primary and trend interval data."""
        from datetime import datetime, timedelta, UTC
        start_dt = datetime.now(UTC) - timedelta(days=int(self.cfg.days))
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        df_primary = get_historical_klines(self.cfg.symbol, self.cfg.interval, start_str)
        if df_primary.empty:
            raise RuntimeError(f"No primary data for {self.cfg.symbol}|{self.cfg.interval}")

        df_trend = get_historical_klines(self.cfg.symbol, self.cfg.trend_interval, start_str)
        if df_trend.empty:
            raise RuntimeError(f"No trend data for {self.cfg.symbol}|{self.cfg.trend_interval}")

        return df_primary.sort_values("Date").reset_index(drop=True), df_trend.sort_values("Date").reset_index(drop=True)

    def _prepare_features(self) -> None:
        """Prepares the multi-timeframe feature set for the agent."""
        if self._df_primary is None or self._df_trend is None:
            self._df_primary, self._df_trend = self._load_data()

        # 1. Calculate Trend Interval Features (ADX)
        trend_df = self._df_trend.copy()
        trend_df.ta.adx(length=self.cfg.trend_adx_period, append=True)
        trend_features = trend_df[['Date', f'ADX_{self.cfg.trend_adx_period}', f'DMP_{self.cfg.trend_adx_period}', f'DMN_{self.cfg.trend_adx_period}']]
        trend_features.set_index('Date', inplace=True)

        # 2. Calculate Primary Interval Features
        primary_df = self._df_primary.copy()
        primary_df.set_index('Date', inplace=True)
        for period in self.cfg.rsi_periods:
            primary_df.ta.rsi(length=period, append=True)
        primary_df.ta.stoch(k=self.cfg.stoch_period, append=True)
        primary_df.ta.atr(append=True)
        primary_df.ta.obv(append=True)

        # 3. Merge Trend Features into Primary DataFrame
        primary_df = pd.merge_asof(primary_df, trend_features, left_index=True, right_index=True, direction='backward')
        primary_df.reset_index(inplace=True) # Reset index to get 'Date' back as a column

        # 4. Time-based cyclical features
        primary_df['day_of_week'] = primary_df['Date'].dt.dayofweek
        primary_df['hour_of_day'] = primary_df['Date'].dt.hour
        day_of_week_sin = np.sin(2 * np.pi * primary_df['day_of_week'] / 7)
        hour_of_day_sin = np.sin(2 * np.pi * primary_df['hour_of_day'] / 24)

        # 5. Select and normalize features
        feature_cols = [
            f'RSI_{p}' for p in self.cfg.rsi_periods
        ] + [
            f'STOCHk_{self.cfg.stoch_period}_3_3',
            f'STOCHd_{self.cfg.stoch_period}_3_3',
            f'ADX_{self.cfg.trend_adx_period}',
            f'DMP_{self.cfg.trend_adx_period}',
            f'DMN_{self.cfg.trend_adx_period}',
            primary_df.columns[primary_df.columns.str.startswith('ATRr')][0],
            primary_df.columns[primary_df.columns.str.startswith('OBV')][0]
        ]
        
        df_feats = primary_df[feature_cols].copy()
        df_feats.insert(0, 'hour_of_day_sin', hour_of_day_sin)
        df_feats.insert(0, 'day_of_week_sin', day_of_week_sin)
        
        df_feats.dropna(inplace=True)
        align_index = df_feats.index
        
        # Normalize data (using mean/std of the full dataset for simplicity here,
        # but in a rigorous setup, this should be fit on training data only)
        for col in df_feats.columns:
            if df_feats[col].std() > 1e-6:
                df_feats[col] = (df_feats[col] - df_feats[col].mean()) / df_feats[col].std()

        self._prices = primary_df['close'].loc[align_index].values.astype(np.float32)
        self._features = df_feats.values.astype(np.float32)

    @property
    def observation_size(self) -> int:
        # Size of features + position flags + bars in position
        return self._features.shape[1] + 2 + 1

    @property
    def action_size(self) -> int:
        return 3 if self.cfg.long_only else 5

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        if self._features is None:
            self._prepare_features()

        # Random start index for the episode
        if self.cfg.random_start:
            max_start = max(0, len(self._features) - self.cfg.episode_len - 2)
            self._start_i = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        else:
            self._start_i = 0
        
        self._i = self._start_i
        self._end_i = self._i + self.cfg.episode_len

        self._pos = 0
        self._entry_price = 0.0
        self._bars_in_pos = 0
        self._bars_since_exit = 1_000_000
        self.portfolio_values = [self.initial_capital]
        return self._obs()

    def _obs(self) -> np.ndarray:
        base_features = self._features[self._i]
        
        pos_long = 1.0 if self._pos == 1 else 0.0
        pos_short = 1.0 if self._pos == -1 else 0.0
        pos_flags = np.array([pos_long, pos_short], dtype=np.float32)

        # Normalize bars_in_pos (e.g., 0-1 scale up to max_position_bars)
        bars_in_pos_norm = min(self._bars_in_pos / (self.cfg.max_position_bars or 100), 1.0)
        
        return np.concatenate([base_features, pos_flags, [bars_in_pos_norm]]).astype(np.float32)

    def step(self, action: int) -> StepResult:
        if self._i >= len(self._features) - 1:
            return StepResult(self._obs(), 0.0, True, {"reason": "end_of_data"})

        price = self._prices[self._i]
        reward = 0.0
        info: Dict[str, Any] = {}

        # --- Action Execution ---
        # Action 1: Open Long
        if action == 1 and self._pos == 0 and self._bars_since_exit >= self.cfg.reopen_cooldown_bars:
            exec_price = price * (1.0 + self.cfg.slippage_bps * 1e-4)
            fee = exec_price * self.cfg.lot_size * self.cfg.fee_rate
            reward -= (fee + self.cfg.action_cost_open)
            self._pos = 1
            self._entry_price = exec_price
            self._bars_in_pos = 0
            info["trade"] = "OPEN_LONG"
        # Action 2: Close Long
        elif action == 2 and self._pos == 1 and self._bars_in_pos >= self.cfg.min_hold_bars:
            exec_price = price * (1.0 - self.cfg.slippage_bps * 1e-4)
            fee = exec_price * self.cfg.lot_size * self.cfg.fee_rate
            pnl = (exec_price - self._entry_price) * self.cfg.lot_size
            reward += (self.cfg.reward_pnl_weight * pnl - fee - self.cfg.action_cost_close)
            self._pos = 0
            self._entry_price = 0.0
            self._bars_since_exit = 0
            info["trade"] = "CLOSE_LONG"
        # Invalid actions
        elif (action == 1 and self._pos != 0) or (action == 2 and self._pos != 1):
             reward -= self.cfg.invalid_action_penalty
             info["invalid_action"] = True
        
        # --- Forced Exit ---
        if self._pos == 1 and self.cfg.max_position_bars is not None and self._bars_in_pos >= self.cfg.max_position_bars:
            exec_price = price * (1.0 - self.cfg.slippage_bps * 1e-4)
            fee = exec_price * self.cfg.lot_size * self.cfg.fee_rate
            pnl = (exec_price - self._entry_price) * self.cfg.lot_size
            reward += (self.cfg.reward_pnl_weight * pnl - fee - self.cfg.action_cost_close)
            self._pos = 0
            self._entry_price = 0.0
            self._bars_since_exit = 0
            info["trade_forced"] = "CLOSE_LONG_MAX_HOLD"

        # --- Update State ---
        self._i += 1
        if self._pos != 0:
            self._bars_in_pos += 1
        else:
            self._bars_since_exit += 1

        # Update portfolio value
        current_value = self.portfolio_values[-1]
        if self._pos == 1:
            unrealized_pnl = (self._prices[self._i] - self._entry_price) * self.cfg.lot_size
            current_value += unrealized_pnl - (self._prices[self._i-1] - self._entry_price) * self.cfg.lot_size
        self.portfolio_values.append(current_value + reward) # Add step reward to value

        obs = self._obs()
        done = self._i >= (len(self._features) - 1) or self._i >= self._end_i
        return StepResult(obs, float(reward), done, info)

    def calculate_sharpe_ratio(self) -> float:
        """Calculates the annualized Sharpe Ratio for the episode."""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return 0.0
        
        # Assuming 1m candles, 24*60 candles per day
        trading_days = 252
        candles_per_day = 24 * 60
        annualization_factor = np.sqrt(candles_per_day * trading_days)

        sharpe = (returns.mean() / returns.std()) * annualization_factor
        return sharpe if np.isfinite(sharpe) else 0.0
