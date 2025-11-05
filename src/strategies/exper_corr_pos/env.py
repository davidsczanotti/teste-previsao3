from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

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
    accounting_mode: str = "mtm"  # "mtm" or "legacy"
    init_equity: float = 1000.0
    leverage: float = 1.0
    equity_floor_pct: float = 0.0
    max_drawdown_pct: float = 1.0
    drawdown_kill_bars: int = 0
    turnover_penalty: float = 0.0
    dynamic_position: bool = False
    random_start: bool = False
    window_bars: int = 0
    idle_penalty_factor: float = 0.0
    hold_bonus_alpha: float = 0.0
    # Novos campos de controle de risco/execução
    max_trade_notional: float = 1000.0  # teto em USD por trade
    profit_trail_pct: float = 0.02      # trailing por pico/vale para perseguir lucro
    allow_intrabar_closes: bool = True  # permite fechar/reabrir na mesma barra


class BTCMixtureEnv(gym.Env):
    """Custom trading environment for BTCUSDT 1h mixture-of-experts agent."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        config: EnvConfig,
        *,
        norm_mean: Optional[pd.Series] = None,
        norm_std: Optional[pd.Series] = None,
        timestamps: Optional[Union[pd.Index, np.ndarray]] = None,
    ) -> None:
        assert len(df) == len(features), "Price and feature data misaligned"
        self.df = df.reset_index(drop=True)
        self.features = features.reset_index(drop=True)
        self.cfg = config
        if timestamps is not None:
            self._timestamps = pd.Index(timestamps)
        else:
            idx = getattr(df, "index", None)
            if isinstance(idx, pd.RangeIndex) or idx is None:
                self._timestamps = None
            else:
                self._timestamps = pd.Index(idx)
        mode = (self.cfg.accounting_mode or "mtm").lower()
        if mode not in {"mtm", "legacy"}:
            raise ValueError(f"accounting_mode inválido: {self.cfg.accounting_mode}")
        self.cfg.accounting_mode = mode
        self.action_space = SpacesDiscrete(3)  # 0 short, 1 flat, 2 long
        self.observation_space = SpacesBox(
            low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32
        )

        self._pos: int = 0
        self._pos_size: float = 0.0
        self._entry_price: float = 0.0
        self._trailing: Optional[float] = None
        self._peak_price: Optional[float] = None    # maior preço desde a entrada (long)
        self._trough_price: Optional[float] = None  # menor preço desde a entrada (short)
        self._equity: float = self.cfg.init_equity
        self._cash: float = self.cfg.init_equity
        self._peak_equity: float = self.cfg.init_equity
        self._drawdown_counter: int = 0
        self._step = 0
        # Controle de eventos de trade para relatórios/avaliações
        self._just_closed: bool = False
        self._last_trade_pnl: float = 0.0
        self._last_trade_bars: int = 0
        self._last_trade_reason: str = ""
        self._last_trade_cost: float = 0.0
        self._last_trade_bonus: float = 0.0
        self._last_trade_gross: float = 0.0
        self._last_trade_entry_price: float = 0.0
        self._last_trade_exit_price: float = 0.0
        self._last_trade_entry_idx: int = -1
        self._last_trade_exit_idx: int = -1
        self._last_trade_entry_ts: Optional[pd.Timestamp] = None
        self._last_trade_exit_ts: Optional[pd.Timestamp] = None
        self._last_trade_side: int = 0
        self._last_trade_size: float = 0.0
        self._open_cost: float = 0.0
        self._entry_idx: int = -1
        self._entry_timestamp: Optional[pd.Timestamp] = None
        # Normalização: permite injetar estatísticas pré-ajustadas (ex.: do treino)
        self._norm_mean = features.mean() if norm_mean is None else norm_mean
        std_series = features.std().replace(0.0, 1.0) if norm_std is None else norm_std
        # evita divisões por zero
        self._norm_std = std_series.replace(0.0, 1.0)
        self._start_idx: int = 0
        self._end_idx: int = len(self.df)
        self._idle_penalty_per_step: float = 0.0

    def reset(self) -> np.ndarray:
        total_len = len(self.df)
        window = self.cfg.window_bars
        if window <= 0 or window > total_len:
            window = total_len
        if self.cfg.random_start and window < total_len:
            max_start = total_len - window
            self._start_idx = int(np.random.randint(0, max_start + 1))
        else:
            self._start_idx = 0
        self._end_idx = self._start_idx + window

        self._pos = 0
        self._pos_size = 0.0
        self._entry_price = 0.0
        self._entry_step = 0
        self._trailing = None
        self._peak_price = None
        self._trough_price = None
        self._equity = self.cfg.init_equity
        self._cash = self.cfg.init_equity
        self._peak_equity = self.cfg.init_equity
        self._drawdown_counter = 0
        self._step = 0
        self._just_closed = False
        self._last_trade_pnl = 0.0
        self._last_trade_bars = 0
        self._last_trade_reason = ""
        self._last_trade_cost = 0.0
        self._last_trade_bonus = 0.0
        self._last_trade_gross = 0.0
        self._last_trade_entry_price = 0.0
        self._last_trade_exit_price = 0.0
        self._last_trade_entry_idx = -1
        self._last_trade_exit_idx = -1
        self._last_trade_entry_ts = None
        self._last_trade_exit_ts = None
        self._last_trade_side = 0
        self._last_trade_size = 0.0
        self._open_cost = 0.0
        self._entry_idx = -1
        self._entry_timestamp = None
        self._episode_length = max(1, self._end_idx - self._start_idx)
        if self.cfg.idle_penalty_factor > 0.0:
            self._idle_penalty_per_step = (
                (self.cfg.init_equity * self.cfg.idle_penalty_factor) / float(self._episode_length)
            )
        else:
            self._idle_penalty_per_step = 0.0
        return self._get_obs()

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, dict]:
        done = False
        reward = 0.0
        ruined = False

        cur_idx = self._start_idx + self._step
        price = float(self.df.iloc[cur_idx]["close"])
        atr = float(self.features.iloc[cur_idx]["atr_14"])

        desired_pos = action - 1  # map {0:-1,1:0,2:+1}

        if desired_pos != self._pos:
            if self.cfg.turnover_penalty > 0.0:
                reward -= self.cfg.turnover_penalty
            # close current position first
            if self._pos != 0:
                reward += self._close_position(price, reason="flip")
            # open new position
            if desired_pos != 0:
                reward += self._open_position(desired_pos, price, atr)

        # Move to next bar
        next_idx = cur_idx + 1
        if next_idx >= self._end_idx:
            done = True
            next_idx = min(next_idx, self._end_idx - 1)
        next_price = float(self.df.iloc[next_idx]["close"])
        next_atr = float(self.features.iloc[next_idx]["atr_14"])
        next_low = float(self.df.iloc[next_idx]["low"])
        next_high = float(self.df.iloc[next_idx]["high"])

        # mark-to-market PnL
        reward += self._pos * self._pos_size * (next_price - price)

        # Update trailing stop
        reward += self._maybe_apply_trailing(next_price, next_low, next_high, next_atr)

        # Penalidade por ficar flat (sem posição)
        if self._pos == 0 and self._idle_penalty_per_step > 0.0:
            reward -= self._idle_penalty_per_step

        self._cash += reward
        self._equity = self._cash
        self._peak_equity = max(self._peak_equity, self._equity)

        drawdown = 0.0
        if self._peak_equity > 0.0:
            drawdown = (self._peak_equity - self._equity) / self._peak_equity

        equity_floor = self.cfg.init_equity * self.cfg.equity_floor_pct
        if self.cfg.equity_floor_pct > 0.0 and self._equity <= equity_floor:
            done = True
            ruined = True

        if self.cfg.max_drawdown_pct < 1.0 and drawdown >= self.cfg.max_drawdown_pct:
            self._drawdown_counter += 1
            if self.cfg.drawdown_kill_bars == 0 or self._drawdown_counter >= self.cfg.drawdown_kill_bars:
                done = True
                ruined = True
        else:
            self._drawdown_counter = 0

        self._step = min(self._step + 1, self._end_idx - self._start_idx - 1)
        obs = self._get_obs()
        info = {
            "equity": self._equity,
            "drawdown": drawdown,
            "ruined": ruined,
            "start_idx": self._start_idx,
            "end_idx": self._end_idx,
            # informações adicionais para análise de trades
            "position": self._pos,
            "trade_closed": bool(self._just_closed),
            "trade_pnl": float(self._last_trade_pnl) if self._just_closed else 0.0,
            "trade_bars": int(self._last_trade_bars) if self._just_closed else 0,
            "trade_reason": self._last_trade_reason if self._just_closed else "",
            "trade_cost": float(self._last_trade_cost) if self._just_closed else 0.0,
            "trade_bonus": float(self._last_trade_bonus) if self._just_closed else 0.0,
            "trade_gross": float(self._last_trade_gross) if self._just_closed else 0.0,
            "trade_entry_price": float(self._last_trade_entry_price) if self._just_closed else 0.0,
            "trade_exit_price": float(self._last_trade_exit_price) if self._just_closed else 0.0,
            "trade_entry_idx": int(self._last_trade_entry_idx) if self._just_closed else -1,
            "trade_exit_idx": int(self._last_trade_exit_idx) if self._just_closed else -1,
            "trade_entry_ts": self._format_timestamp(self._last_trade_entry_ts) if self._just_closed else "",
            "trade_exit_ts": self._format_timestamp(self._last_trade_exit_ts) if self._just_closed else "",
            "trade_side": int(self._last_trade_side) if self._just_closed else 0,
            "trade_size": float(self._last_trade_size) if self._just_closed else 0.0,
            "timestamp": self._format_timestamp(self._resolve_timestamp(self._start_idx + self._step)),
        }
        # reset do marcador de fechamento para o próximo passo
        self._just_closed = False
        return obs, reward, done, info

    # ------------------------------------------------------------------
    def _transaction_cost(self, price: float, size: float) -> float:
        notional = price * size
        return notional * (self.cfg.fee_pct + self.cfg.slippage_pct)

    def _close_position(self, price: float, reason: str = "close") -> float:
        entry_price = self._entry_price
        pnl = self._pos * self._pos_size * (price - entry_price)
        side = self._pos
        size = self._pos_size
        cost = self._transaction_cost(price, size)
        # duração do trade em barras desde a entrada
        duration_bars = max(1, self._step - self._entry_step)
        exit_idx = self._start_idx + self._step
        entry_idx = self._entry_idx
        entry_ts = self._entry_timestamp
        self._pos = 0
        self._pos_size = 0.0
        self._entry_price = 0.0
        self._trailing = None
        self._peak_price = None
        self._trough_price = None
        mode = (self.cfg.accounting_mode or "mtm").lower()
        # bônus/malus por duração do trade
        bonus = 0.0
        if self.cfg.hold_bonus_alpha > 0.0:
            bonus = self.cfg.hold_bonus_alpha * duration_bars * pnl
        # PnL total realizado do trade (inclui custos de entrada e saída e bônus)
        trade_pnl_total = pnl - self._open_cost - cost + bonus
        # Sinaliza fechamento para consumo externo (walk-forward/relatórios)
        self._just_closed = True
        self._last_trade_pnl = float(trade_pnl_total)
        self._last_trade_bars = int(duration_bars)
        self._last_trade_reason = reason
        self._last_trade_cost = float(self._open_cost + cost)
        self._last_trade_bonus = float(bonus)
        self._last_trade_gross = float(pnl)
        self._last_trade_entry_price = float(entry_price)
        self._last_trade_exit_price = float(price)
        self._last_trade_entry_idx = int(entry_idx)
        self._last_trade_exit_idx = int(exit_idx)
        self._last_trade_entry_ts = entry_ts
        self._last_trade_exit_ts = self._resolve_timestamp(exit_idx)
        self._last_trade_side = int(side)
        self._last_trade_size = float(size)
        self._entry_idx = -1
        self._entry_timestamp = None
        if mode == "legacy":
            return pnl - cost
        return -cost + pnl + bonus

    def _open_position(self, pos: int, price: float, atr: float) -> float:
        self._pos = pos
        self._entry_price = price
        self._pos_size = self._compute_position_size(price)
        self._entry_step = self._step
        self._entry_idx = self._start_idx + self._step
        self._entry_timestamp = self._resolve_timestamp(self._entry_idx)
        if pos > 0:
            self._trailing = price - self.cfg.stop_atr_mult * atr
            self._peak_price = price
            self._trough_price = None
        else:
            self._trailing = price + self.cfg.stop_atr_mult * atr
            self._trough_price = price
            self._peak_price = None
        cost = self._transaction_cost(price, self._pos_size)
        # registra custo de entrada para contabilizar no PnL do trade no fechamento
        self._open_cost = float(cost)
        return -cost

    def _maybe_apply_trailing(
        self, next_price: float, next_low: float, next_high: float, next_atr: float
    ) -> float:
        if not getattr(self.cfg, "allow_intrabar_closes", True):
            return 0.0
        if self._pos == 0 or self._trailing is None:
            return 0.0
        reward_adj = 0.0
        # Atualiza trailing combinando ATR e trailing por lucro (pico/vale)
        ptp = max(0.0, float(self.cfg.profit_trail_pct))
        if self._pos > 0:
            # Long: stop é um piso; sobe com ATR e com o pico de preço
            atr_floor = next_price - self.cfg.trail_atr_mult * next_atr
            if self._peak_price is None:
                self._peak_price = next_high
            else:
                self._peak_price = max(self._peak_price, next_high)
            profit_floor = self._peak_price * (1.0 - ptp) if ptp > 0.0 else -np.inf
            self._trailing = max(self._trailing, atr_floor, profit_floor)
            # trailing não pode ultrapassar o preço atual para cima (piso < preço)
            self._trailing = min(self._trailing, next_price)
            if next_low <= self._trailing:
                stop_price = float(self._trailing)
                diff = self._pos_size * (stop_price - next_price)
                reward_adj += diff  # ajustar PnL para o preço de stop
                reason = "trail_profit" if ptp > 0.0 and stop_price >= profit_floor else "trail_atr"
                reward_adj += self._close_position(stop_price, reason=reason)
        else:
            # Short: stop é um teto; desce com ATR e com o vale de preço
            atr_ceiling = next_price + self.cfg.trail_atr_mult * next_atr
            if self._trough_price is None:
                self._trough_price = next_low
            else:
                self._trough_price = min(self._trough_price, next_low)
            profit_ceiling = self._trough_price * (1.0 + ptp) if ptp > 0.0 else np.inf
            self._trailing = min(self._trailing, atr_ceiling, profit_ceiling)
            # trailing não pode ficar abaixo do preço atual para baixo (teto > preço)
            self._trailing = max(self._trailing, next_price)
            if next_high >= self._trailing:
                stop_price = float(self._trailing)
                diff = self._pos_size * (next_price - stop_price)
                reward_adj += diff
                reason = "trail_profit" if ptp > 0.0 and stop_price <= profit_ceiling else "trail_atr"
                reward_adj += self._close_position(stop_price, reason=reason)
        return reward_adj

    def _get_obs(self) -> np.ndarray:
        feats = (self.features.iloc[self._start_idx + self._step] - self._norm_mean) / (self._norm_std + 1e-9)
        return feats.values.astype(np.float32)

    def _compute_position_size(self, price: float) -> float:
        max_notional = max(0.0, float(getattr(self.cfg, "max_trade_notional", 1000.0)))
        px = max(price, 1e-9)
        if not self.cfg.dynamic_position:
            # cap por notional: min(position_size, max_notional/preço)
            cap_size = max_notional / px if max_notional > 0 else float("inf")
            return float(min(self.cfg.position_size, cap_size))
        equity = max(self._equity, 0.0)
        notional = equity * max(self.cfg.leverage, 0.0)
        capped = min(notional, max_notional) if max_notional > 0 else notional
        return capped / px

    def _resolve_timestamp(self, idx: int) -> Optional[pd.Timestamp]:
        if self._timestamps is None or len(self._timestamps) == 0:
            return None
        if idx < 0 or idx >= len(self._timestamps):
            return None
        ts = self._timestamps[idx]
        if isinstance(ts, pd.Timestamp):
            return ts
        try:
            return pd.Timestamp(ts)
        except Exception:
            return None

    @staticmethod
    def _format_timestamp(ts: Optional[pd.Timestamp]) -> str:
        if ts is None:
            return ""
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is None:
                return ts.isoformat()
            return ts.tz_convert("UTC").isoformat()
        return str(ts)
