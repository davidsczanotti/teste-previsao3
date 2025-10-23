from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:  # Prefer gym if available
    import gym
    SpacesDiscrete = gym.spaces.Discrete
    SpacesBox = gym.spaces.Box
except ModuleNotFoundError:
    class SpacesDiscrete:
        def __init__(self, n: int):
            self.n = n

        def sample(self) -> int:
            return int(np.random.randint(self.n))

    class SpacesBox:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _DummyEnv:
        def reset(self):  # pragma: no cover
            raise NotImplementedError

        def step(self, action):  # pragma: no cover
            raise NotImplementedError

    class _DummyGym:  # minimal stub for type hints
        Env = _DummyEnv
        spaces = type("spaces", (), {"Discrete": SpacesDiscrete, "Box": SpacesBox})

    gym = _DummyGym()


Action = int


@dataclass
class EnvConfig:
    fee_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.0001  # 0.01%
    position_size: float = 0.1  # BTC
    stop_atr_mult: float = 2.0
    trail_atr_mult: float = 1.0


class BTCMixtureEnv(gym.Env):
    """Custom trading environment for BTCUSDT 1h mixture-of-experts agent."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        config: EnvConfig,
    ) -> None:
        assert len(df) == len(features), "Price and feature data misaligned"
        self.df = df.reset_index(drop=True)
        self.features = features.reset_index(drop=True)
        self.cfg = config
        self.action_space = SpacesDiscrete(3)  # 0 short, 1 flat, 2 long
        self.observation_space = SpacesBox(
            low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32
        )

        self._pos: int = 0
        self._entry_price: float = 0.0
        self._trailing: Optional[float] = None
        self._equity: float = 1000.0
        self._cash: float = 1000.0
        self._step = 0
        self._norm_mean = features.mean()
        self._norm_std = features.std().replace(0.0, 1.0)

    def reset(self) -> np.ndarray:
        self._pos = 0
        self._entry_price = 0.0
        self._trailing = None
        self._equity = 1000.0
        self._cash = 1000.0
        self._step = 0
        return self._get_obs()

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, dict]:
        done = False
        reward = 0.0

        price = float(self.df.loc[self._step, "close"])
        atr = float(self.features.loc[self._step, "atr_14"])

        desired_pos = action - 1  # map {0:-1,1:0,2:+1}

        if desired_pos != self._pos:
            # close current position first
            if self._pos != 0:
                reward += self._close_position(price)
            # open new position
            if desired_pos != 0:
                reward += self._open_position(desired_pos, price, atr)

        # Move to next bar
        next_idx = self._step + 1
        if next_idx >= len(self.df) - 1:
            done = True
        next_price = float(self.df.loc[next_idx, "close"])
        next_atr = float(self.features.loc[next_idx, "atr_14"])
        next_low = float(self.df.loc[next_idx, "low"])
        next_high = float(self.df.loc[next_idx, "high"])

        # mark-to-market PnL
        reward += self._pos * self.cfg.position_size * (next_price - price)

        # Update trailing stop
        reward += self._maybe_apply_trailing(next_price, next_low, next_high, next_atr)

        self._cash += reward
        self._equity = self._cash
        self._step = next_idx
        obs = self._get_obs()
        info = {"equity": self._equity}
        return obs, reward, done, info

    # ------------------------------------------------------------------
    def _transaction_cost(self, price: float) -> float:
        notional = price * self.cfg.position_size
        return notional * (self.cfg.fee_pct + self.cfg.slippage_pct)

    def _close_position(self, price: float) -> float:
        pnl = self._pos * self.cfg.position_size * (price - self._entry_price)
        cost = self._transaction_cost(price)
        self._pos = 0
        self._entry_price = 0.0
        self._trailing = None
        return pnl - cost

    def _open_position(self, pos: int, price: float, atr: float) -> float:
        self._pos = pos
        self._entry_price = price
        if pos > 0:
            self._trailing = price - self.cfg.stop_atr_mult * atr
        else:
            self._trailing = price + self.cfg.stop_atr_mult * atr
        cost = self._transaction_cost(price)
        return -cost

    def _maybe_apply_trailing(
        self, next_price: float, next_low: float, next_high: float, next_atr: float
    ) -> float:
        if self._pos == 0 or self._trailing is None:
            return 0.0
        reward_adj = 0.0
        if self._pos > 0:
            self._trailing = max(self._trailing, next_price - self.cfg.trail_atr_mult * next_atr)
            if next_low <= self._trailing:
                stop_price = self._trailing
                diff = self.cfg.position_size * (stop_price - next_price)
                reward_adj += diff  # adjust PnL to stop execution price
                reward_adj += self._close_position(stop_price)
        else:
            self._trailing = min(self._trailing, next_price + self.cfg.trail_atr_mult * next_atr)
            if next_high >= self._trailing:
                stop_price = self._trailing
                diff = self.cfg.position_size * (next_price - stop_price)
                reward_adj += diff
                reward_adj += self._close_position(stop_price)
        return reward_adj

    def _get_obs(self) -> np.ndarray:
        feats = (self.features.iloc[self._step] - self._norm_mean) / (self._norm_std + 1e-9)
        return feats.values.astype(np.float32)
