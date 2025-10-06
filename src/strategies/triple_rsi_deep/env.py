from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from ...binance_client import get_historical_klines


def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _stochastic_k(df: pd.DataFrame, period: int) -> pd.Series:
    # Uses high/low/close to compute %K
    low_min = df["low"].rolling(window=period).min()
    high_max = df["high"].rolling(window=period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = (df["close"] - low_min) / denom
    return (k.clip(0, 1) * 100.0)


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class TripleRsiEnv:
    """RL environment for Triple RSI + Stochastic.

    Observations:
      - rsi_short (0..1)
      - rsi_med (0..1)
      - rsi_long (0..1)
      - stoch_k (0..1)
      - dist_to_upper, dist_to_lower
      - cross_down_upper (0/1), cross_up_lower (0/1)
      - position_long (0/1), position_short (0/1)

    Actions (discrete):
      0 = Hold
      1 = Open long (if flat)
      2 = Close long (if long)
      3 = Open short (if flat)
      4 = Close short (if short)

    Reward:
      realized_weight * realized PnL on closes, plus m2m_weight * mark-to-market delta,
      minus fees/slippage/costs; optional bonuses/penalties on timing.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        days: int = 120,
        short_period: int = 33,
        med_period: int = 44,
        long_period: int = 115,
        stoch_period: int = 14,
        stoch_upper: float = 0.75,
        stoch_lower: float = 0.25,
        # costs/constraints
        lot_size: float = 0.001,
        fee_rate: float = 0.001,
        slippage_bps: float = 0.0,
        action_cost_open: float = 0.0,
        action_cost_close: float = 0.0,
        invalid_action_penalty: float = 0.0,
        min_hold_bars: int = 0,
        reopen_cooldown_bars: int = 0,
        max_position_bars: Optional[int] = None,
        negative_close_boost: float = 0.0,
        long_only: bool = True,
        # gating and shaping
        gate_enabled: bool = True,
        gate_margin: float = 0.05,
        gate_recent_k: int = 3,
        realized_weight: float = 1.0,
        m2m_weight: float = 0.1,
        midrange_penalty: float = 0.5,
        close_bonus_factor: float = 0.2,
        # episodes
        episode_len: Optional[int] = 2048,
        random_start: bool = True,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.days = days
        self.short_p = short_period
        self.med_p = med_period
        self.long_p = long_period
        self.stoch_p = int(stoch_period)
        self.stoch_upper = float(stoch_upper)
        self.stoch_lower = float(stoch_lower)

        self.lot_size = float(lot_size)
        self.fee_rate = float(fee_rate)
        self.slippage_bps = float(slippage_bps)
        self.action_cost_open = float(action_cost_open)
        self.action_cost_close = float(action_cost_close)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.min_hold_bars = int(min_hold_bars)
        self.reopen_cooldown_bars = int(reopen_cooldown_bars)
        self.max_position_bars = int(max_position_bars) if max_position_bars is not None else None
        self.negative_close_boost = float(negative_close_boost)
        self.long_only = bool(long_only)

        self.gate_enabled = bool(gate_enabled)
        self.gate_margin = float(gate_margin)
        self.gate_recent_k = int(gate_recent_k)
        self.realized_weight = float(realized_weight)
        self.m2m_weight = float(m2m_weight)
        self.midrange_penalty = float(midrange_penalty)
        self.close_bonus_factor = float(close_bonus_factor)

        self.episode_len = int(episode_len) if episode_len is not None else None
        self.random_start = bool(random_start)

        # Runtime state
        self._df: Optional[pd.DataFrame] = df
        self._features: Optional[np.ndarray] = None
        self._closes: Optional[np.ndarray] = None
        self._i: int = 0
        self._pos: int = 0  # 0 flat, 1 long, -1 short
        self._entry: float = 0.0
        self._bars_since_entry: int = 0
        self._bars_since_exit: int = 1_000_000
        self._since_cross_up_lower: int = 1_000_000
        self._since_cross_down_upper: int = 1_000_000
        self.initial_capital: float = 10000.0

    # Data prep
    def _ensure_data(self) -> None:
        if self._df is None:
            from datetime import datetime, timedelta, UTC
            start_dt = datetime.now(UTC) - timedelta(days=int(self.days))
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            df = get_historical_klines(self.symbol, self.interval, start_str)
            if df.empty:
                raise RuntimeError("No data from Binance for RL environment")
            self._df = df.sort_values("Date").reset_index(drop=True)

        df = self._df.copy()
        closes = df["close"].astype(float)
        rsi_s = _calculate_rsi(closes, self.short_p)
        rsi_m = _calculate_rsi(closes, self.med_p)
        rsi_l = _calculate_rsi(closes, self.long_p)
        stoch_k = _stochastic_k(df, self.stoch_p)

        # Align to remove NaNs
        start = max(self.short_p, self.med_p, self.long_p, self.stoch_p)
        df = df.iloc[start:].reset_index(drop=True)
        closes = closes.iloc[start:].reset_index(drop=True)
        rsi_s = rsi_s.iloc[start:].reset_index(drop=True)
        rsi_m = rsi_m.iloc[start:].reset_index(drop=True)
        rsi_l = rsi_l.iloc[start:].reset_index(drop=True)
        stoch_k = stoch_k.iloc[start:].reset_index(drop=True)

        # Compute crosses and distances
        st = (stoch_k.values / 100.0).astype(np.float32)
        st_prev = np.concatenate([[st[0]], st[:-1]])
        cross_down_upper = ((st_prev > self.stoch_upper) & (st <= self.stoch_upper)).astype(np.float32)
        cross_up_lower = ((st_prev < self.stoch_lower) & (st >= self.stoch_lower)).astype(np.float32)
        dist_upper = (st - self.stoch_upper).astype(np.float32)
        dist_lower = (self.stoch_lower - st).astype(np.float32)

        # Features
        feats = np.stack([
            (rsi_s.values / 100.0).astype(np.float32),
            (rsi_m.values / 100.0).astype(np.float32),
            (rsi_l.values / 100.0).astype(np.float32),
            st.astype(np.float32),
            dist_upper.astype(np.float32),
            dist_lower.astype(np.float32),
            cross_down_upper.astype(np.float32),
            cross_up_lower.astype(np.float32),
        ], axis=1)

        self._features = feats
        self._closes = closes.values.astype(np.float32)
        

    @property
    def observation_size(self) -> int:
        return 10  # 3 RSI + stoch + distances + crosses + 2 position flags

    @property
    def action_size(self) -> int:
        return 3 if self.long_only else 5  # hold, open long, close long [, open short, close short]

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self._ensure_data()
        # Random start index
        if self.episode_len is not None and self.random_start:
            max_start = max(0, len(self._features) - self.episode_len - 2)
            self._i = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
            self._end_i = self._i + self.episode_len
        else:
            self._i = 0
            self._end_i = len(self._features) - 1
        self._pos = 0
        self._entry = 0.0
        self._bars_since_entry = 0
        self._bars_since_exit = 1_000_000
        self._since_cross_up_lower = 1_000_000
        self._since_cross_down_upper = 1_000_000
        return self._obs()

    def _obs(self) -> np.ndarray:
        assert self._features is not None
        assert self._i < len(self._features)
        base = self._features[self._i]
        pos_long = 1.0 if self._pos == 1 else 0.0
        pos_short = 1.0 if self._pos == -1 else 0.0
        pos = np.array([pos_long, pos_short], dtype=np.float32)
        return np.concatenate([base, pos]).astype(np.float32)

    def step(self, action: int) -> StepResult:
        assert self._features is not None and self._closes is not None
        if self._i >= len(self._features) - 1:
            return StepResult(self._obs(), 0.0, True, {"reason": "eod"})

        price = float(self._closes[self._i])
        next_price = float(self._closes[self._i + 1])
        reward = 0.0
        info: Dict[str, Any] = {}

        # Extract timing features for gating/shaping
        feat = self._features[self._i]
        st = float(feat[3])  # 0..1
        cross_down_upper_now = bool(feat[6] > 0.5)
        cross_up_lower_now = bool(feat[7] > 0.5)

        # Helper: slippage-adjusted trade price
        def slip_buy(px: float) -> float:
            return px * (1.0 + self.slippage_bps * 1e-4)

        def slip_sell(px: float) -> float:
            return px * (1.0 - self.slippage_bps * 1e-4)

        # Gating helpers
        def _can_open_long() -> bool:
            if self._pos != 0 or self._bars_since_exit < self.reopen_cooldown_bars:
                return False
            if not self.gate_enabled:
                return True
            near = st <= (self.stoch_lower + self.gate_margin)
            recent = (self._since_cross_up_lower <= self.gate_recent_k)
            return bool(near or recent)

        def _can_open_short() -> bool:
            if self._pos != 0 or self._bars_since_exit < self.reopen_cooldown_bars:
                return False
            if not self.gate_enabled:
                return True
            near = st >= (self.stoch_upper - self.gate_margin)
            recent = (self._since_cross_down_upper <= self.gate_recent_k)
            return bool(near or recent)

        # Execute action with validity checks and penalties
        if action == 1:  # open long
            if _can_open_long():
                exec_price = slip_buy(price)
                fee = exec_price * self.lot_size * self.fee_rate
                self._pos = 1
                self._entry = exec_price
                self._bars_since_entry = 0
                reward -= fee
                reward -= self.action_cost_open
                # midrange penalty
                if 0.4 <= st <= 0.6:
                    reward -= self.midrange_penalty
                info["trade"] = "BUY"
            else:
                reward -= self.invalid_action_penalty
                info["invalid"] = "open_long_gated"
        elif action == 2:  # close long
            if self._pos == 1 and self._bars_since_entry >= self.min_hold_bars:
                exec_price = slip_sell(price)
                fee = exec_price * self.lot_size * self.fee_rate
                pnl = (exec_price - self._entry) * self.lot_size
                reward += self.realized_weight * pnl
                reward -= fee
                reward -= self.action_cost_close
                if pnl < 0 and self.negative_close_boost > 0:
                    reward += self.negative_close_boost * pnl
                # bonus for closing near upper extreme
                if st >= self.stoch_upper:
                    reward += self.close_bonus_factor * (st - self.stoch_upper)
                self._pos = 0
                self._entry = 0.0
                self._bars_since_exit = 0
                self._bars_since_entry = 0
                info["trade"] = "SELL"
            else:
                reward -= self.invalid_action_penalty
                info["invalid"] = "close_long_not_allowed"
        elif action == 3:  # open short
            if self.long_only:
                reward -= self.invalid_action_penalty
                info["invalid"] = "open_short_disabled"
            elif _can_open_short():
                exec_price = slip_sell(price)
                fee = exec_price * self.lot_size * self.fee_rate
                self._pos = -1
                self._entry = exec_price
                self._bars_since_entry = 0
                reward -= fee
                reward -= self.action_cost_open
                if 0.4 <= st <= 0.6:
                    reward -= self.midrange_penalty
                info["trade"] = "SELL_SHORT"
            else:
                reward -= self.invalid_action_penalty
                info["invalid"] = "open_short_gated"
        elif action == 4:  # close short
            if self.long_only:
                reward -= self.invalid_action_penalty
                info["invalid"] = "close_short_disabled"
            elif self._pos == -1 and self._bars_since_entry >= self.min_hold_bars:
                exec_price = slip_buy(price)
                fee = exec_price * self.lot_size * self.fee_rate
                pnl = (self._entry - exec_price) * self.lot_size
                reward += self.realized_weight * pnl
                reward -= fee
                reward -= self.action_cost_close
                if pnl < 0 and self.negative_close_boost > 0:
                    reward += self.negative_close_boost * pnl
                if st <= self.stoch_lower:
                    reward += self.close_bonus_factor * (self.stoch_lower - st)
                self._pos = 0
                self._entry = 0.0
                self._bars_since_exit = 0
                self._bars_since_entry = 0
                info["trade"] = "BUY_TO_COVER"
            else:
                reward -= self.invalid_action_penalty
                info["invalid"] = "close_short_not_allowed"
        else:
            # hold
            pass

        # Forced exits by max hold bars
        if self._pos == 1 and self.max_position_bars is not None and self._bars_since_entry >= self.max_position_bars:
            exec_price = slip_sell(price)
            fee = exec_price * self.lot_size * self.fee_rate
            pnl = (exec_price - self._entry) * self.lot_size
            reward += self.realized_weight * pnl
            reward -= fee
            reward -= self.action_cost_close
            if pnl < 0 and self.negative_close_boost > 0:
                reward += self.negative_close_boost * pnl
            self._pos = 0
            self._entry = 0.0
            self._bars_since_exit = 0
            self._bars_since_entry = 0
            info["trade_forced"] = "SELL_MAX_HOLD"
        if self._pos == -1 and self.max_position_bars is not None and self._bars_since_entry >= self.max_position_bars:
            exec_price = slip_buy(price)
            fee = exec_price * self.lot_size * self.fee_rate
            pnl = (self._entry - exec_price) * self.lot_size
            reward += self.realized_weight * pnl
            reward -= fee
            reward -= self.action_cost_close
            if pnl < 0 and self.negative_close_boost > 0:
                reward += self.negative_close_boost * pnl
            self._pos = 0
            self._entry = 0.0
            self._bars_since_exit = 0
            self._bars_since_entry = 0
            info["trade_forced"] = "BUY_TO_COVER_MAX_HOLD"

        # Mark-to-market between timesteps
        if self._pos == 1:
            unreal_now = (price - self._entry) * self.lot_size
            unreal_next = (next_price - self._entry) * self.lot_size
            reward += self.m2m_weight * (unreal_next - unreal_now)
        elif self._pos == -1:
            unreal_now = (self._entry - price) * self.lot_size
            unreal_next = (self._entry - next_price) * self.lot_size
            reward += self.m2m_weight * (unreal_next - unreal_now)

        # Advance time
        # Update cross recency counters using current step's cross flags
        if cross_up_lower_now:
            self._since_cross_up_lower = 0
        else:
            self._since_cross_up_lower += 1
        if cross_down_upper_now:
            self._since_cross_down_upper = 0
        else:
            self._since_cross_down_upper += 1

        self._i += 1
        if self._pos != 0:
            self._bars_since_entry += 1
        else:
            self._bars_since_exit += 1
        obs = self._obs()
        done = self._i >= (len(self._features) - 1) or (self.episode_len is not None and self._i >= self._end_i)
        return StepResult(obs, float(reward), done, info)

    def run_episode(self, policy_fn, max_steps: Optional[int] = None) -> Dict[str, Any]:
        obs = self.reset()
        total_reward = 0.0
        steps = 0
        trades = 0
        while True:
            logits = policy_fn(obs)
            action = int(np.argmax(logits))
            res = self.step(action)
            total_reward += res.reward
            steps += 1
            if "trade" in res.info:
                trades += 1
            if res.done or (max_steps is not None and steps >= max_steps):
                break
            obs = res.obs
        return {"reward": total_reward, "steps": steps, "trades": trades}
