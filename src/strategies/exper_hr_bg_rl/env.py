from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd


Action = int  # 0=flat, 1=long, 2=short (se allow_short=True)


@dataclass
class EnvConfig:
    fee_pct: float = 0.001
    slippage_pct: float = 0.0003
    position_size: float = 0.001
    init_equity: float = 1000.0
    window_bars: int = 512
    random_start: bool = True
    allow_short: bool = True
    reward_scale_divisor: float = 10.0
    idle_penalty: float = 0.0
    min_hold_bars: int = 0


class RangeVolEnv:
    """Ambiente RL simples para BTCUSDT 1h baseado em features de range/vol."""

    def __init__(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        cfg: EnvConfig,
        timestamps: Optional[List] = None,
    ) -> None:
        if len(prices) != len(features):
            raise ValueError("prices e features devem ter o mesmo comprimento.")
        self.df = prices.reset_index(drop=True)
        self.features = features.reset_index(drop=True)
        self.cfg = cfg
        self.timestamps = list(timestamps) if timestamps is not None else list(range(len(self.df)))

        self._close = self.df["close"].astype(float).to_numpy()
        self._n = len(self._close)
        if self._n < 3:
            raise ValueError("Série muito curta para o ambiente.")

        self.allow_short = bool(cfg.allow_short)
        self.n_actions = 3 if self.allow_short else 2  # 0=flat,1=long,(2=short)

        self._start_idx = 0
        self._end_idx = self._n
        self._step = 0
        self._pos = 0  # -1,0,1
        self._equity = float(cfg.init_equity)
        self._done = False
        self._entry_step: Optional[int] = None

    def reset(self) -> np.ndarray:
        total_len = self._n
        window = int(self.cfg.window_bars)
        if window <= 0 or window > total_len:
            window = total_len

        if self.cfg.random_start and window < total_len:
            max_start = total_len - window
            self._start_idx = int(np.random.randint(0, max_start + 1))
        else:
            self._start_idx = 0
        self._end_idx = self._start_idx + window

        self._step = 0
        self._pos = 0
        self._equity = float(self.cfg.init_equity)
        self._done = False
        self._entry_step = None

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        idx = self._start_idx + self._step
        idx = min(max(idx, 0), self._end_idx - 1)
        row = self.features.iloc[idx]
        arr = row.to_numpy(dtype=np.float32)
        pos_feat = np.array([float(self._pos)], dtype=np.float32)
        obs = np.concatenate([arr, pos_feat], axis=0)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() chamado após done=True.")

        act = int(action)
        if act < 0 or act >= self.n_actions:
            raise ValueError(f"Ação inválida: {action}")

        if self.allow_short:
            # 0=flat,1=long,2=short
            desired_pos = 0 if act == 0 else (1 if act == 1 else -1)
        else:
            # 0=flat,1=long
            desired_pos = 0 if act == 0 else 1

        cur_idx = self._start_idx + self._step
        cur_idx = min(max(cur_idx, 0), self._end_idx - 2)  # garante próximo candle

        price = float(self._close[cur_idx])
        next_idx = cur_idx + 1
        if next_idx >= self._end_idx:
            next_idx = self._end_idx - 1
        next_price = float(self._close[next_idx])

        prev_equity = float(self._equity)

        # Regra de hold mínimo: impede fechar/virar posição antes de N barras
        min_hold = max(0, int(getattr(self.cfg, "min_hold_bars", 0)))
        if min_hold > 0 and self._pos != 0 and self._entry_step is not None:
            held_bars = int(self._step - self._entry_step)
            if held_bars < min_hold and desired_pos != self._pos:
                desired_pos = self._pos

        # custo de transação ao mudar posição
        delta_pos = float(desired_pos - self._pos)
        cost = 0.0
        if delta_pos != 0.0:
            notional = abs(delta_pos) * self.cfg.position_size * price
            cost = notional * (self.cfg.fee_pct + self.cfg.slippage_pct)

        # PnL mark-to-market com base na posição ATUAL (antes de mudar)
        pnl_mtm = self._pos * self.cfg.position_size * (next_price - price)

        raw_reward = pnl_mtm - cost
        scale = max(1.0, float(self.cfg.reward_scale_divisor))
        reward = float(raw_reward / scale)

        self._equity = prev_equity + reward
        # Atualiza posição e passo de entrada
        new_pos = int(desired_pos)
        if new_pos != self._pos:
            if new_pos == 0:
                self._entry_step = None
            else:
                self._entry_step = int(self._step)
        self._pos = new_pos

        # Penalidade por ficar flat (idle), inspirada em exper_corr_pos
        idle_pen = float(getattr(self.cfg, "idle_penalty", 0.0))
        if idle_pen > 0.0 and self._pos == 0:
            reward -= idle_pen
            self._equity -= idle_pen

        # avanço de tempo
        self._step += 1
        if self._start_idx + self._step >= self._end_idx - 1:
            self._done = True

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "equity": float(self._equity),
            "position": int(self._pos),
            "price": price,
            "next_price": next_price,
        }
        return obs, reward, self._done, info
