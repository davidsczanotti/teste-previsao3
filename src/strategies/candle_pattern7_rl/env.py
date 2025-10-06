from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List

import numpy as np
import pandas as pd

from ...binance_client import get_historical_klines
from .policy import Candle as PolicyCandle, default_signal_from_last7, decide_action_from_signal


def _sliding_window_features(
    df: pd.DataFrame,
    window: int = 7,
    ma_short_window: int = 7,
    ma_long_window: int = 40,
) -> tuple[np.ndarray, np.ndarray, int, Dict[str, np.ndarray]]:
    """Build observation features and auxiliary series.

    Features include per-candle normalised body/wick/range values and aggregated
    statistics. Additional aggregates incorporate short/long moving averages
    (SMA7 & SMA40) to give the agent trend context.
    """

    if len(df) < max(window, ma_long_window) + 2:
        empty = np.zeros((0, 1), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.float32), 0, {}

    open_s = df["open"].astype(float).to_numpy()
    high_s = df["high"].astype(float).to_numpy()
    low_s = df["low"].astype(float).to_numpy()
    close_series = df["close"].astype(float)
    close_np = close_series.to_numpy()

    ma_short = close_series.rolling(window=ma_short_window, min_periods=ma_short_window).mean()
    ma_long = close_series.rolling(window=ma_long_window, min_periods=ma_long_window).mean()
    ma_short_prev = ma_short.shift(1)
    ma_long_prev = ma_long.shift(1)

    start = max(window, ma_long_window)

    feats: list[np.ndarray] = []
    for idx in range(start, len(df)):
        # window candles end at idx-1
        o_w = open_s[idx - window : idx]
        h_w = high_s[idx - window : idx]
        l_w = low_s[idx - window : idx]
        c_w = close_np[idx - window : idx]

        o_safe = np.where(o_w == 0.0, 1e-12, o_w)
        body = (c_w - o_w) / o_safe
        rng = (h_w - l_w) / o_safe
        upper = (h_w - np.maximum(o_w, c_w)) / o_safe
        lower = (np.minimum(o_w, c_w) - l_w) / o_safe

        greens = np.sum(c_w > o_w)
        reds = window - greens
        sma_window = np.mean(c_w)
        last_close = c_w[-1]
        last_close_rel_sma = (last_close - sma_window) / sma_window if sma_window != 0 else 0.0
        momentum_rel = (last_close / c_w[0] - 1.0) if c_w[0] != 0 else 0.0
        avg_range_rel = float(np.mean(rng))

        ma_s = float(ma_short.iloc[idx]) if not np.isnan(ma_short.iloc[idx]) else last_close
        ma_l = float(ma_long.iloc[idx]) if not np.isnan(ma_long.iloc[idx]) else last_close
        ma_s_prev = float(ma_short_prev.iloc[idx]) if not np.isnan(ma_short_prev.iloc[idx]) else ma_s
        ma_l_prev = float(ma_long_prev.iloc[idx]) if not np.isnan(ma_long_prev.iloc[idx]) else ma_l

        ma_short_rel_close = (ma_s - last_close) / last_close if last_close != 0 else 0.0
        ma_long_rel_close = (ma_l - last_close) / last_close if last_close != 0 else 0.0
        ma_diff_rel = (ma_s - ma_l) / (abs(ma_l) + 1e-6)
        ma_short_slope = ma_s - ma_s_prev
        ma_long_slope = ma_l - ma_l_prev

        agg = np.array([
            greens / window,
            reds / window,
            last_close_rel_sma,
            momentum_rel,
            avg_range_rel,
            ma_short_rel_close,
            ma_long_rel_close,
            ma_diff_rel,
            ma_short_slope,
            ma_long_slope,
        ], dtype=np.float32)

        candle_feats = np.stack([body, rng, upper, lower], axis=1).astype(np.float32).reshape(-1)
        feats.append(np.concatenate([candle_feats, agg], axis=0))

    features = np.stack(feats, axis=0)
    closes = close_np[start:]

    extras = {
        "ma_short": ma_short.iloc[start:].ffill().bfill().astype(float).to_numpy(dtype=np.float32),
        "ma_long": ma_long.iloc[start:].ffill().bfill().astype(float).to_numpy(dtype=np.float32),
        "ma_short_prev": ma_short_prev.iloc[start:].ffill().bfill().astype(float).to_numpy(dtype=np.float32),
        "ma_long_prev": ma_long_prev.iloc[start:].ffill().bfill().astype(float).to_numpy(dtype=np.float32),
    }

    return features, closes.astype(np.float32), start, extras


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class Candle7Env:
    """RL environment for 7-candle pattern trading.

    Observation (float32 vector):
      - Per-candle normalized features for last 7 closed candles (4 each = 28)
      - Aggregates: pattern stats (5) + moving-average context (5)
      - Position flags: pos_long, pos_short (2)
      Total: 40 dims

    Actions (discrete):
      0 = Hold
      1 = Open long (if flat)
      2 = Close long (if long)
      3 = Open short (if flat, allow_short=True)
      4 = Close short (if short)

    Reward = realized_weight * realized PnL (on trade closes)
           + m2m_weight * mark-to-market delta between steps
           - fees, slippage, action costs, and optional penalties.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        days: int = 120,
        lot_size: float = 0.001,
        fee_rate: float = 0.001,
        slippage_bps: float = 0.0,
        action_cost_open: float = 0.0,
        action_cost_close: float = 0.0,
        invalid_action_penalty: float = 0.0,
        min_hold_bars: int = 0,
        reopen_cooldown_bars: int = 0,
        max_position_bars: Optional[int] = None,
        long_only: bool = False,
        realized_weight: float = 1.0,
        m2m_weight: float = 0.05,
        exec_at_next_open: bool = True,
        switch_penalty: float = 0.0,
        switch_window_bars: int = 5,
        # idle shaping
        idle_penalty: float = 0.0,           # penalização por ficar flat e segurar "Hold" (USD por barra)
        idle_grace_bars: int = 0,            # período de carência sem penalidade
        idle_ramp: float = 0.0,              # rampa linear adicional após a carência (penal *= 1 + idle_ramp * (bars - grace))
        reward_atr_norm: bool = False,
        atr_period: int = 14,
        atr_eps: float = 1e-6,
        gate_on_heuristic: bool = False,
        episode_len: Optional[int] = 2048,
        random_start: bool = True,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.days = days
        self.lot_size = float(lot_size)
        self.fee_rate = float(fee_rate)
        self.slippage_bps = float(slippage_bps)
        self.action_cost_open = float(action_cost_open)
        self.action_cost_close = float(action_cost_close)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.min_hold_bars = int(min_hold_bars)
        self.reopen_cooldown_bars = int(reopen_cooldown_bars)
        self.max_position_bars = int(max_position_bars) if max_position_bars is not None else None
        self.long_only = bool(long_only)

        self.realized_weight = float(realized_weight)
        self.m2m_weight = float(m2m_weight)
        self.exec_at_next_open = bool(exec_at_next_open)
        self.switch_penalty = float(switch_penalty)
        self.switch_window_bars = int(switch_window_bars)
        self.idle_penalty = float(idle_penalty)
        self.idle_grace_bars = int(idle_grace_bars)
        self.idle_ramp = float(idle_ramp)
        self.reward_atr_norm = bool(reward_atr_norm)
        self.atr_period = int(atr_period)
        self.atr_eps = float(atr_eps)
        self.gate_on_heuristic = bool(gate_on_heuristic)

        self.episode_len = int(episode_len) if episode_len is not None else None
        self.random_start = bool(random_start)

        # runtime
        self._df: Optional[pd.DataFrame] = df
        self._base_features: Optional[np.ndarray] = None
        self._closes: Optional[np.ndarray] = None
        self._opens: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray] = None
        self._ma_short: Optional[np.ndarray] = None
        self._ma_long: Optional[np.ndarray] = None
        self._ma_short_prev: Optional[np.ndarray] = None
        self._ma_long_prev: Optional[np.ndarray] = None
        self._start_idx: int = 0
        self._i: int = 0
        self._end_i: int = 0
        self._pos: int = 0
        self._entry: float = 0.0
        self._bars_since_entry: int = 0
        self._bars_since_exit: int = 0
        self.initial_capital: float = 10000.0

    def _ensure_data(self) -> None:
        if self._df is None:
            from datetime import datetime, timedelta, UTC
            start_dt = datetime.now(UTC) - timedelta(days=int(self.days))
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            df = get_historical_klines(self.symbol, self.interval, start_str)
            if df.empty:
                raise RuntimeError("No data for Candle7Env")
            self._df = df.sort_values("Date").reset_index(drop=True)

        # Compute features once per dataset (cache across episodes)
        if self._base_features is None or self._closes is None:
            feats, closes, start_idx, extras = _sliding_window_features(
                self._df,
                window=7,
                ma_short_window=7,
                ma_long_window=40,
            )
            if len(feats) < 3:
                raise RuntimeError("Not enough data to build 7-candle features")

            self._start_idx = int(start_idx)
            self._base_features = feats.astype(np.float32)
            self._closes = closes.astype(np.float32)
            self._ma_short = extras["ma_short"]
            self._ma_long = extras["ma_long"]
            self._ma_short_prev = extras["ma_short_prev"]
            self._ma_long_prev = extras["ma_long_prev"]

            opens = self._df["open"].astype(float).to_numpy()
            self._opens = opens[self._start_idx :].astype(np.float32)

            high = self._df["high"].astype(float)
            low = self._df["low"].astype(float)
            close = self._df["close"].astype(float)
            prev_close = close.shift(1)
            tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1 / max(1, self.atr_period), adjust=False).mean().bfill().fillna(0.0)
            self._atr = atr.values[self._start_idx :].astype(np.float32)

    @property
    def observation_size(self) -> int:
        return (4 * 7 + 10) + 2  # base features + aggregates + position flags

    @property
    def action_size(self) -> int:
        return 3 if self.long_only else 5

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self._ensure_data()
        assert self._base_features is not None
        n = len(self._base_features)
        if self.episode_len is not None and self.random_start:
            max_start = max(0, n - self.episode_len - 2)
            self._i = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
            self._end_i = min(n - 1, self._i + self.episode_len)
        else:
            self._i = 0
            self._end_i = n - 1
        self._pos = 0
        self._entry = 0.0
        self._bars_since_entry = 0
        self._bars_since_exit = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        assert self._base_features is not None
        base = self._base_features[self._i]
        pos_long = 1.0 if self._pos == 1 else 0.0
        pos_short = 1.0 if self._pos == -1 else 0.0
        return np.concatenate([base, np.array([pos_long, pos_short], dtype=np.float32)]).astype(np.float32)

    def step(self, action: int) -> StepResult:
        assert self._base_features is not None and self._closes is not None
        if self._i >= len(self._base_features) - 1:
            return StepResult(self._obs(), 0.0, True, {"reason": "eod"})

        price = float(self._closes[self._i])
        next_price = float(self._closes[self._i + 1])
        next_open = float(self._opens[self._i + 1]) if (self._opens is not None and (self._i + 1) < len(self._opens)) else next_price
        reward = 0.0
        info: Dict[str, Any] = {}

        # Heuristic guidance (for BC/gating): compute signal from last 7 candles + MAs
        heur_action: Optional[int] = None
        try:
            if self._df is not None and self._ma_short is not None and self._ma_long is not None:
                data_idx = self._start_idx + self._i
                if data_idx >= 7:
                    rows = self._df.iloc[data_idx - 7 : data_idx]
                    if len(rows) == 7:
                        last7: List[PolicyCandle] = [
                            PolicyCandle(
                                open=float(r["open"]),
                                high=float(r["high"]),
                                low=float(r["low"]),
                                close=float(r["close"]),
                            )
                            for _, r in rows.iterrows()
                        ]
                        ma_s = float(self._ma_short[self._i])
                        ma_l = float(self._ma_long[self._i])
                        ma_s_prev = float(self._ma_short_prev[self._i]) if self._ma_short_prev is not None else ma_s
                        ma_l_prev = float(self._ma_long_prev[self._i]) if self._ma_long_prev is not None else ma_l
                        sig = default_signal_from_last7(
                            last7,
                            ma_short=ma_s,
                            ma_long=ma_l,
                            prev_ma_short=ma_s_prev,
                            prev_ma_long=ma_l_prev,
                        )
                        heur_action = decide_action_from_signal(sig, self._pos, allow_short=not self.long_only)
                        info["heuristic_action"] = int(heur_action)
        except Exception:
            pass

        def slip_buy(px: float) -> float:
            return px * (1.0 + self.slippage_bps * 1e-4)

        def slip_sell(px: float) -> float:
            return px * (1.0 - self.slippage_bps * 1e-4)

        # Execute action
        trade_executed = False
        pos_before = int(self._pos)

        if action == 1:  # open long
            if self.gate_on_heuristic and heur_action is not None and heur_action != 1:
                reward -= self.invalid_action_penalty
                info["invalid"] = "open_long_gated"
            elif self._pos == 0 and self._bars_since_exit >= self.reopen_cooldown_bars:
                exec_price = slip_buy(next_open) if self.exec_at_next_open else slip_buy(price)
                fee = exec_price * self.lot_size * self.fee_rate
                self._pos = 1
                self._entry = exec_price
                self._bars_since_entry = 0
                reward -= fee
                reward -= self.action_cost_open
                if self._bars_since_exit <= self.switch_window_bars:
                    reward -= self.switch_penalty
                info["trade"] = "BUY"
                trade_executed = True
            else:
                reward -= self.invalid_action_penalty
                info["invalid"] = "open_long_not_allowed"
        elif action == 2:  # close long
            if self._pos == 1 and self._bars_since_entry >= self.min_hold_bars:
                exec_price = slip_sell(next_open) if self.exec_at_next_open else slip_sell(price)
                fee = exec_price * self.lot_size * self.fee_rate
                pnl = (exec_price - self._entry) * self.lot_size
                reward += self.realized_weight * pnl
                reward -= fee
                reward -= self.action_cost_close
                self._pos = 0
                self._entry = 0.0
                self._bars_since_exit = 0
                info["hold_bars"] = int(self._bars_since_entry)
                self._bars_since_entry = 0
                info["trade"] = "SELL"
                trade_executed = True
            else:
                reward -= self.invalid_action_penalty
                info["invalid"] = "close_long_not_allowed"
        elif action == 3:  # open short
            if self.long_only:
                reward -= self.invalid_action_penalty
                info["invalid"] = "open_short_disabled"
            elif self.gate_on_heuristic and heur_action is not None and heur_action != 3:
                reward -= self.invalid_action_penalty
                info["invalid"] = "open_short_gated"
            elif self._pos == 0 and self._bars_since_exit >= self.reopen_cooldown_bars:
                exec_price = slip_sell(next_open) if self.exec_at_next_open else slip_sell(price)
                fee = exec_price * self.lot_size * self.fee_rate
                self._pos = -1
                self._entry = exec_price
                self._bars_since_entry = 0
                reward -= fee
                reward -= self.action_cost_open
                if self._bars_since_exit <= self.switch_window_bars:
                    reward -= self.switch_penalty
                info["trade"] = "SELL_SHORT"
                trade_executed = True
            else:
                reward -= self.invalid_action_penalty
                info["invalid"] = "open_short_not_allowed"
        elif action == 4:  # close short
            if self.long_only:
                reward -= self.invalid_action_penalty
                info["invalid"] = "close_short_disabled"
            elif self._pos == -1 and self._bars_since_entry >= self.min_hold_bars:
                exec_price = slip_buy(next_open) if self.exec_at_next_open else slip_buy(price)
                fee = exec_price * self.lot_size * self.fee_rate
                pnl = (self._entry - exec_price) * self.lot_size
                reward += self.realized_weight * pnl
                reward -= fee
                reward -= self.action_cost_close
                self._pos = 0
                self._entry = 0.0
                self._bars_since_exit = 0
                info["hold_bars"] = int(self._bars_since_entry)
                self._bars_since_entry = 0
                info["trade"] = "BUY_TO_COVER"
                trade_executed = True
            else:
                reward -= self.invalid_action_penalty
                info["invalid"] = "close_short_not_allowed"
        else:
            # hold
            pass

        # Forced exits by max hold
        if self._pos == 1 and self.max_position_bars is not None and self._bars_since_entry >= self.max_position_bars:
            exec_price = slip_sell(next_open) if self.exec_at_next_open else slip_sell(price)
            fee = exec_price * self.lot_size * self.fee_rate
            pnl = (exec_price - self._entry) * self.lot_size
            reward += self.realized_weight * pnl
            reward -= fee
            reward -= self.action_cost_close
            self._pos = 0
            self._entry = 0.0
            self._bars_since_exit = 0
            self._bars_since_entry = 0
            info["trade_forced"] = "SELL_MAX_HOLD"
            trade_executed = True
        if self._pos == -1 and self.max_position_bars is not None and self._bars_since_entry >= self.max_position_bars:
            exec_price = slip_buy(next_open) if self.exec_at_next_open else slip_buy(price)
            fee = exec_price * self.lot_size * self.fee_rate
            pnl = (self._entry - exec_price) * self.lot_size
            reward += self.realized_weight * pnl
            reward -= fee
            reward -= self.action_cost_close
            self._pos = 0
            self._entry = 0.0
            self._bars_since_exit = 0
            self._bars_since_entry = 0
            info["trade_forced"] = "BUY_TO_COVER_MAX_HOLD"
            trade_executed = True

        # Idle penalty: penalize holding flat with action=Hold after grace period
        if (
            self.idle_penalty > 0.0
            and action == 0
            and self._pos == 0
            and not trade_executed
            and self._bars_since_exit >= self.idle_grace_bars
        ):
            extra = max(0, self._bars_since_exit - self.idle_grace_bars)
            scale = 1.0 + self.idle_ramp * float(extra)
            reward -= self.idle_penalty * scale
            info["idle"] = True

        # M2M shaping within the next bar only (open->close), using post-action position
        if self._pos != 0:
            delta = next_price - next_open
            m2m = (delta * self.lot_size) if self._pos == 1 else ((-delta) * self.lot_size)
            if self.reward_atr_norm and self._atr is not None and (self._i + 1) < len(self._atr):
                scale = max(float(self._atr[self._i + 1]), self.atr_eps)
                m2m = m2m / scale
            reward += self.m2m_weight * m2m

        # advance time
        self._i += 1
        if self._pos != 0:
            self._bars_since_entry += 1
        else:
            self._bars_since_exit += 1

        obs = self._obs()
        done = self._i >= (len(self._base_features) - 1) or (self.episode_len is not None and self._i >= self._end_i)
        return StepResult(obs, float(reward), done, info)

    def run_episode(self, policy_fn: Callable[[np.ndarray], np.ndarray], max_steps: Optional[int] = None) -> Dict[str, Any]:
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
