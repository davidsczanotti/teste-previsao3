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
        self._pos_size: float = 0.0
        self._bars_in_pos: int = 0
        self._bars_since_exit: int = 1_000_000
        self._force_stop: bool = False
        
        # Portfolio tracking
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.portfolio_values: List[float] = []
        self.initial_capital: float = 10000.0

        # Apply reward profile if configured
        self._apply_reward_profile()

    def _apply_reward_profile(self) -> None:
        profile = getattr(self.cfg, 'reward_profile', None)
        if not profile:
            return
        p = profile.lower().strip()
        if p == 'conservative':
            self.cfg.reward_sharpe_weight = 0.5
            self.cfg.reward_sortino_weight = 0.3
            self.cfg.reward_calmar_weight = 0.2
            self.cfg.reward_pnl_weight = 0.0
        elif p == 'aggressive':
            self.cfg.reward_sharpe_weight = 0.3
            self.cfg.reward_sortino_weight = 0.2
            self.cfg.reward_calmar_weight = 0.1
            self.cfg.reward_pnl_weight = 0.4
        else:  # balanced
            self.cfg.reward_sharpe_weight = 0.4
            self.cfg.reward_sortino_weight = 0.2
            self.cfg.reward_calmar_weight = 0.1
            self.cfg.reward_pnl_weight = 0.3

    def _load_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Loads primary and trend interval data; supports multiple trend intervals."""
        from datetime import datetime, timedelta, UTC
        start_dt = datetime.now(UTC) - timedelta(days=int(self.cfg.days))
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Primary
        df_primary = get_historical_klines(self.cfg.symbol, self.cfg.interval, start_str)
        if df_primary.empty:
            raise RuntimeError(f"No primary data for {self.cfg.symbol}|{self.cfg.interval}")

        # Trend intervals (MTF)
        trend_dfs: Dict[str, pd.DataFrame] = {}
        intervals = getattr(self.cfg, 'trend_intervals', ["15m"]) or ["15m"]
        for tf in intervals:
            tdf = get_historical_klines(self.cfg.symbol, tf, start_str)
            if tdf.empty:
                continue
            trend_dfs[tf] = tdf.sort_values("Date").reset_index(drop=True)

        if not trend_dfs:
            raise RuntimeError(f"No trend data for {self.cfg.symbol} | {intervals}")

        return df_primary.sort_values("Date").reset_index(drop=True), trend_dfs

    def _prepare_features(self) -> None:
        """Prepares the multi-timeframe feature set for the agent."""
        # Load data if needed
        if self._df_primary is None and self._df_trend is None:
            # Fresh load of both
            primary, trend_map = self._load_data()
            self._df_primary = primary
            # Backward compatibility: keep one trend df for older code paths
            # but also carry the dict in a local var for MTF feature merge below
            self._df_trend = next(iter(trend_map.values()))
            local_trend_map = trend_map
        else:
            local_trend_map = {}
            if self._df_primary is None:
                primary, trend_map = self._load_data()
                self._df_primary = primary
                local_trend_map = trend_map
            # If trend data is missing, load just trend frames
            if self._df_trend is None:
                _, trend_map = self._load_data()
                self._df_trend = next(iter(trend_map.values()))
                local_trend_map = trend_map
            # If both exist but we don't have the trend map, build from the single df_trend
            if not local_trend_map and self._df_trend is not None:
                # Only single trend available
                local_trend_map = {getattr(self.cfg, 'trend_intervals', ["15m"])[0]: self._df_trend}

        # 1. Calculate Trend Interval Features (ADX/RSI/EMA) for each trend timeframe
        trend_features_list = []
        for tf, tdf in local_trend_map.items():
            trend_df = tdf.copy()
            trend_df.ta.adx(length=self.cfg.trend_adx_period, append=True)
            # Add RSI set for trend timeframe
            for r in getattr(self.cfg, 'trend_rsi_periods', [14]):
                trend_df.ta.rsi(length=int(r), append=True)
            # Add EMA set for trend timeframe
            for e in getattr(self.cfg, 'trend_ema_periods', [20, 50]):
                trend_df.ta.ema(length=int(e), append=True)

            cols = [f'ADX_{self.cfg.trend_adx_period}', f'DMP_{self.cfg.trend_adx_period}', f'DMN_{self.cfg.trend_adx_period}']
            cols += [f'RSI_{int(r)}' for r in getattr(self.cfg, 'trend_rsi_periods', [14])]
            cols += [f'EMA_{int(e)}' for e in getattr(self.cfg, 'trend_ema_periods', [20, 50])]

            feat = trend_df[['Date'] + cols].copy()
            # Rename with TF suffix to avoid collisions
            ren = {f'ADX_{self.cfg.trend_adx_period}': f'ADX_{self.cfg.trend_adx_period}@{tf}',
                   f'DMP_{self.cfg.trend_adx_period}': f'DMP_{self.cfg.trend_adx_period}@{tf}',
                   f'DMN_{self.cfg.trend_adx_period}': f'DMN_{self.cfg.trend_adx_period}@{tf}'}
            for r in getattr(self.cfg, 'trend_rsi_periods', [14]):
                ren[f'RSI_{int(r)}'] = f'RSI_{int(r)}@{tf}'
            for e in getattr(self.cfg, 'trend_ema_periods', [20, 50]):
                ren[f'EMA_{int(e)}'] = f'EMA_{int(e)}@{tf}'
            feat = feat.rename(columns=ren)
            feat.set_index('Date', inplace=True)
            trend_features_list.append(feat)

        # 2. Calculate Primary Interval Features
        primary_df = self._df_primary.copy()
        primary_df.set_index('Date', inplace=True)
        for period in self.cfg.rsi_periods:
            primary_df.ta.rsi(length=period, append=True)
        primary_df.ta.stoch(k=self.cfg.stoch_period, append=True)
        primary_df.ta.atr(append=True)
        primary_df.ta.obv(append=True)

        # 3. Merge Trend Features into Primary DataFrame
        for feat in trend_features_list:
            primary_df = pd.merge_asof(primary_df, feat, left_index=True, right_index=True, direction='backward')
        primary_df.reset_index(inplace=True) # Reset index to get 'Date' back as a column

        # 4. Time-based cyclical features
        primary_df['day_of_week'] = primary_df['Date'].dt.dayofweek
        primary_df['hour_of_day'] = primary_df['Date'].dt.hour
        day_of_week_sin = np.sin(2 * np.pi * primary_df['day_of_week'] / 7)
        hour_of_day_sin = np.sin(2 * np.pi * primary_df['hour_of_day'] / 24)

        # 5. Select and normalize features
        feature_cols = [f'RSI_{p}' for p in self.cfg.rsi_periods]
        feature_cols += [
            f'STOCHk_{self.cfg.stoch_period}_3_3',
            f'STOCHd_{self.cfg.stoch_period}_3_3',
        ]
        # Add all trend ADX/RSI/EMA features with suffixes
        feature_cols += [c for c in primary_df.columns if c.startswith(f'ADX_{self.cfg.trend_adx_period}@')]
        feature_cols += [c for c in primary_df.columns if c.startswith(f'DMP_{self.cfg.trend_adx_period}@')]
        feature_cols += [c for c in primary_df.columns if c.startswith(f'DMN_{self.cfg.trend_adx_period}@')]
        feature_cols += [c for c in primary_df.columns if c.startswith('RSI_') and '@' in c]
        feature_cols += [c for c in primary_df.columns if c.startswith('EMA_') and '@' in c]
        # Core volatility / volume
        feature_cols += [
            primary_df.columns[primary_df.columns.str.startswith('ATRr')][0],
            primary_df.columns[primary_df.columns.str.startswith('OBV')][0],
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

        self._prices = primary_df['close'].iloc[align_index].values.astype(np.float32)
        self._features = df_feats.values.astype(np.float32)

    def _ensure_data(self) -> None:
        if self._features is None or self._prices is None:
            self._prepare_features()

    def _recent_return_vol(self, window: int = 100) -> float:
        if self._prices is None or self._i < 2:
            return 0.0
        start = max(1, self._i - window)
        rets = pd.Series(self._prices[start:self._i+1]).pct_change().dropna()
        return float(rets.std()) if not rets.empty else 0.0

    def _current_lot_size(self) -> float:
        base = float(getattr(self.cfg, 'lot_size', None) or self.cfg.base_lot_size)
        if not getattr(self.cfg, 'dynamic_position_sizing', True):
            return base
        # Kelly fraction from realized path (fallback if too short)
        try:
            kelly = self.calculate_kelly_criterion()
        except Exception:
            kelly = 0.0
        if not np.isfinite(kelly) or kelly <= 0:
            kelly = 0.2
        kelly = min(kelly, max(0.0, float(self.cfg.kelly_fraction_cap)))

        # Volatility targeting using recent returns
        vol = self._recent_return_vol(window=200)
        tgt = max(1e-4, float(getattr(self.cfg, 'target_atr_pct', 0.015)))
        scale = float(np.clip(tgt / max(vol, 1e-6), 0.5, 2.0))
        sized = base * (kelly / max(self.cfg.kelly_fraction_cap, 1e-6)) * scale
        # Ensure a practical floor
        return float(max(sized, base * 0.1))

    @property
    def observation_size(self) -> int:
        # Size of features + position flags + bars in position
        if self._features is None:
            self._prepare_features()
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
        self._pos_size = 0.0
        self._force_stop = False
        
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.portfolio_values = [self.initial_capital]
        return self._obs()

    def _obs(self) -> np.ndarray:
        if self._features is None or self._prices is None:
            raise RuntimeError("Features not prepared. Call reset() first.")

        base_features = self._features[self._i]

        pos_long = 1.0 if self._pos == 1 else 0.0
        pos_short = 1.0 if self._pos == -1 else 0.0
        pos_flags = np.array([pos_long, pos_short], dtype=np.float32)

        # Normalize bars_in_pos (e.g., 0-1 scale up to max_position_bars)
        bars_in_pos_norm = min(self._bars_in_pos / (self.cfg.max_position_bars or 100), 1.0)

        return np.concatenate([base_features, pos_flags, [bars_in_pos_norm]]).astype(np.float32)

    def calculate_step_reward(self, action: int, prev_value: float, current_value: float, info: Dict[str, Any]) -> float:
        """Multi-scale reward system for stable learning."""

        # Get lot size from config (backward compatibility)
        lot_size = getattr(self.cfg, 'lot_size', getattr(self.cfg, 'base_lot_size', 0.001))

        # Base portfolio reward (small weight, normalized)
        portfolio_reward = (current_value - prev_value) / self.initial_capital

        # Action-based rewards (main component)
        action_reward = 0.0

        if info.get("trade") == "OPEN_LONG":
            # Reward opening positions with positive momentum
            # Check if RSI and other indicators are favorable
            if self._features is not None and self._i > 0:
                current_features = self._features[self._i]
                # RSI > 50 suggests bullish momentum
                rsi_values = current_features[:len(self.cfg.rsi_periods)]  # First N features are RSI
                avg_rsi = np.mean(rsi_values)
                if avg_rsi > 0.3:  # Normalized RSI > 50
                    action_reward += 0.1
                else:
                    action_reward -= 0.05  # Penalty for opening against momentum

        elif info.get("trade") == "CLOSE_LONG":
            # Reward profitable closes, penalize losses
            pnl = (self._prices[self._i] - self._entry_price) * lot_size
            pnl_normalized = pnl / self.initial_capital

            if pnl > 0:
                action_reward += min(pnl_normalized * 2, 0.5)  # Cap positive rewards
            else:
                action_reward += max(pnl_normalized * 2, -0.5)  # Cap penalties

            info["realized_pnl"] = pnl  # Store for analysis

        # Invalid action penalty
        if info.get("invalid_action"):
            action_reward -= 0.2

        # Holding penalties (encourage appropriate action)
        if action == 0:  # Hold action
            if self._pos != 0:  # Holding in position
                # Small penalty that increases with position duration
                hold_penalty = 0.005 * (self._bars_in_pos / (self.cfg.max_position_bars or 100))
                action_reward -= min(hold_penalty, 0.05)
            else:  # Holding flat
                # Small penalty for not acting when market moves
                if self._features is not None and self._i > 0:
                    price_change = (self._prices[self._i] - self._prices[self._i-1]) / self._prices[self._i-1]
                    if abs(price_change) > 0.001:  # Significant price movement
                        action_reward -= 0.01

        # Risk management rewards
        if self._pos != 0:
            # Reward reducing position size in high volatility
            if self._features is not None:
                current_features = self._features[self._i]
                # ATR is typically the last few features
                atr_idx = -2  # Approximate ATR position
                if abs(current_features[atr_idx]) > 0.8:  # High volatility (normalized)
                    if action == 2:  # Closing position
                        action_reward += 0.05

        # Combine rewards with carefully tuned weights
        total_reward = (
            self.cfg.reward_pnl_weight * portfolio_reward +      # Portfolio performance (small)
            0.7 * action_reward                                   # Action quality (main)
        )

        return float(total_reward) if np.isfinite(total_reward) else 0.0

    def step(self, action: int) -> StepResult:
        if self._features is None or self._prices is None:
            raise RuntimeError("Features not prepared. Call reset() first.")

        if self._i >= len(self._features) - 1:
            return StepResult(self._obs(), 0.0, True, {"reason": "end_of_data"})

        price = self._prices[self._i]
        info: Dict[str, Any] = {}

        prev_portfolio_value = self.portfolio_values[-1]

        # Get lot size from config (backward compatibility)
        lot_size = getattr(self.cfg, 'lot_size', getattr(self.cfg, 'base_lot_size', 0.001))

        # --- Action Execution ---
        # Action 1: Open Long
        if action == 1 and self._pos == 0 and self._bars_since_exit >= self.cfg.reopen_cooldown_bars:
            exec_price = price * (1.0 + self.cfg.slippage_bps * 1e-4)
            # Determine dynamic size at open
            open_size = self._current_lot_size()
            self._pos_size = open_size
            self.realized_pnl -= (exec_price * open_size * self.cfg.fee_rate) + self.cfg.action_cost_open
            self._pos = 1
            self._entry_price = exec_price
            self._bars_in_pos = 0
            info["trade"] = "OPEN_LONG"
        # Action 2: Close Long
        elif action == 2 and self._pos == 1 and self._bars_in_pos >= self.cfg.min_hold_bars:
            exec_price = price * (1.0 - self.cfg.slippage_bps * 1e-4)
            pnl = (exec_price - self._entry_price) * self._pos_size
            self.realized_pnl += pnl - (exec_price * self._pos_size * self.cfg.fee_rate) - self.cfg.action_cost_close
            self._pos = 0
            self._entry_price = 0.0
            self._pos_size = 0.0
            self._bars_since_exit = 0
            info["trade"] = "CLOSE_LONG"
        # Invalid actions
        elif (action == 1 and self._pos != 0) or (action == 2 and self._pos != 1):
             self.realized_pnl -= self.cfg.invalid_action_penalty
             info["invalid_action"] = True

        # --- Forced Exit ---
        if self._pos == 1 and self.cfg.max_position_bars is not None and self._bars_in_pos >= self.cfg.max_position_bars:
            exec_price = price * (1.0 - self.cfg.slippage_bps * 1e-4)
            pnl = (exec_price - self._entry_price) * self._pos_size
            self.realized_pnl += pnl - (exec_price * self._pos_size * self.cfg.fee_rate) - self.cfg.action_cost_close
            self._pos = 0
            self._entry_price = 0.0
            self._pos_size = 0.0
            self._bars_since_exit = 0
            info["trade_forced"] = "CLOSE_LONG_MAX_HOLD"

        # --- Update State and PnL ---
        self._i += 1
        if self._pos != 0:
            self._bars_in_pos += 1
        else:
            self._bars_since_exit += 1

        # Update unrealized PnL
        if self._pos == 1:
            self.unrealized_pnl = (self._prices[self._i] - self._entry_price) * self._pos_size
        else:
            self.unrealized_pnl = 0.0

        # Update portfolio value
        current_portfolio_value = self.initial_capital + self.realized_pnl + self.unrealized_pnl
        self.portfolio_values.append(current_portfolio_value)

        # Risk hard limits (e.g., max drawdown)
        if getattr(self.cfg, 'max_drawdown_limit', None) is not None:
            dd = self.calculate_max_drawdown()
            if dd >= float(self.cfg.max_drawdown_limit):
                # Force flatten and stop
                if self._pos == 1:
                    # Mark-to-market flatten at current price (already reflected in unrealized)
                    self._pos = 0
                    self._entry_price = 0.0
                    self._pos_size = 0.0
                    info["risk_stop"] = "MAX_DD"
                self._force_stop = True

        # Calculate stable step reward
        reward = self.calculate_step_reward(action, prev_portfolio_value, current_portfolio_value, info)

        obs = self._obs()
        done = self._force_stop or self._i >= (len(self._features) - 1) or self._i >= self._end_i
        return StepResult(obs, reward, done, info)

    def calculate_sharpe_ratio(self) -> float:
        """Calculates the annualized Sharpe Ratio for the episode."""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        if returns.empty or returns.std() < 1e-8:
            return 0.0

        # Assuming 1m candles, 24*60 candles per day
        candles_per_day = 24 * 60
        annualization_factor = np.sqrt(len(self.portfolio_values) / candles_per_day * 252) if len(self.portfolio_values) > 1 else 1

        sharpe = (returns.mean() / returns.std()) * annualization_factor
        return sharpe if np.isfinite(sharpe) else 0.0

    def calculate_sortino_ratio(self) -> float:
        """Calculates the Sortino Ratio (downside deviation only)."""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        if returns.empty:
            return 0.0

        downside_returns = returns[returns < 0]
        if downside_returns.empty or downside_returns.std() < 1e-8:
            return 0.0

        # Assuming 1m candles, 24*60 candles per day
        candles_per_day = 24 * 60
        annualization_factor = np.sqrt(len(self.portfolio_values) / candles_per_day * 252) if len(self.portfolio_values) > 1 else 1

        sortino = (returns.mean() / downside_returns.std()) * annualization_factor
        return sortino if np.isfinite(sortino) else 0.0

    def calculate_calmar_ratio(self, window: Optional[int] = None) -> float:
        """Calculates the Calmar Ratio (annual return / max drawdown)."""
        if len(self.portfolio_values) < 2:
            return 0.0

        window = window or self.cfg.calmar_window
        if len(self.portfolio_values) < window:
            window = len(self.portfolio_values)

        # Calculate rolling max drawdown
        values = pd.Series(self.portfolio_values[-window:])
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        max_drawdown = drawdown.min()

        if max_drawdown >= 0:
            return 0.0

        # Annualized return (simplified)
        total_return = (values.iloc[-1] - values.iloc[0]) / values.iloc[0]
        years = window / (365 * 24 * 60)  # Assuming 1m candles
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        calmar = annualized_return / abs(max_drawdown)
        return calmar if np.isfinite(calmar) else 0.0

    def calculate_kelly_criterion(self) -> float:
        """Calculates optimal position size using Kelly Criterion."""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        if returns.empty:
            return 0.0

        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean()
        avg_loss = abs(returns[returns < 0].mean())

        if avg_loss == 0:
            return 0.0

        kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        return max(0, kelly)  # Kelly can be negative, but we take max with 0

    def calculate_advanced_reward(self) -> float:
        """Calculates multi-objective reward based on advanced risk metrics."""
        sharpe = self.calculate_sharpe_ratio()
        sortino = self.calculate_sortino_ratio()
        calmar = self.calculate_calmar_ratio()
        kelly = self.calculate_kelly_criterion()

        # Weighted combination of risk metrics
        reward = (
            self.cfg.reward_sharpe_weight * sharpe +
            self.cfg.reward_sortino_weight * sortino +
            self.cfg.reward_calmar_weight * calmar +
            self.cfg.reward_kelly_weight * kelly
        )

        return reward if np.isfinite(reward) else 0.0

    def calculate_var(self, confidence: float = 0.95) -> float:
        """Calculates Value at Risk."""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        if returns.empty:
            return 0.0

        return abs(np.percentile(returns, (1 - confidence) * 100))

    def get_portfolio_stats(self) -> Dict[str, float]:
        """Returns comprehensive portfolio statistics."""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()

        stats = {
            'total_return': (self.portfolio_values[-1] - self.portfolio_values[0]) / self.portfolio_values[0],
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'kelly_criterion': self.calculate_kelly_criterion(),
            'max_drawdown': self.calculate_max_drawdown(),
            'var_95': self.calculate_var(0.95),
            'volatility': returns.std() * np.sqrt(252),  # Annualized
            'win_rate': (returns > 0).mean(),
            'avg_win': returns[returns > 0].mean(),
            'avg_loss': returns[returns < 0].mean(),
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')
        }

        return {k: v for k, v in stats.items() if np.isfinite(v)}

    def calculate_max_drawdown(self) -> float:
        """Calculates maximum drawdown."""
        if len(self.portfolio_values) < 2:
            return 0.0

        values = pd.Series(self.portfolio_values)
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return float(abs(drawdown.min()))
