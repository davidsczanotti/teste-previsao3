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
    turnover_penalty_pct: float = 0.0
    flip_exit_penalty: float = 0.0
    flip_exit_penalty_pct: float = 0.0
    dynamic_position: bool = False
    random_start: bool = False
    window_bars: int = 0
    idle_penalty_factor: float = 0.0
    hold_bonus_alpha: float = 0.0
    # Se verdadeiro, o bônus de hold só é aplicado quando o trade tem PnL positivo.
    # Em perdas, não aplica malus adicional.
    hold_bonus_positive_only: bool = False
    # Novos campos de controle de risco/execução
    max_trade_notional: float = 1000.0  # teto em USD por trade
    profit_trail_pct: float = 0.02      # trailing por pico/vale para perseguir lucro
    allow_intrabar_closes: bool = True  # permite fechar/reabrir na mesma barra
    adaptive_stop_window: int = 50      # janela para calcular ATR histórico médio
    kelly_fraction: float = 0.1         # fração Kelly para position sizing
    var_confidence: float = 0.95        # confiança para VaR
    trend_penalty_coef: float = 0.0     # penaliza posição contra tendência HTF
    trend_penalty_coef_pct: float = 0.0
    trend_penalty_entry_mult: float = 1.0  # multiplicador aplicado na penalidade única em novas entradas
    # Bônus de alinhamento com a tendência HTF (opcional, default desligado)
    trend_bonus_coef: float = 0.0
    trend_bonus_coef_pct: float = 0.0
    trend_bonus_entry_mult: float = 1.0
    trend_throttle_threshold: float = 0.0  # |htf_trend_strength| mínimo para bloquear trades contra a tendência
    trend_throttle_cooldown: int = 0       # número de barras em que o bloqueio permanece ativo
    trend_throttle_idle_penalty: float = 0.0  # penalidade aplicada quando o throttle impede uma ação
    trend_throttle_use_divergence_override: bool = False  # permite furar o bloqueio se houver divergência forte
    trend_throttle_spread_thr: float = 1.5  # |spread_z_zscore| mínimo para considerar divergência
    trend_throttle_pattern_thr: float = 0.6  # intensidade mínima de padrão (ex.: hammer/shooting_star roll) para liberar
    reward_scale_divisor: float = 1.0   # divisor aplicado na recompensa para evitar explosões numéricas


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
        self._pending_action: Optional[int] = None
        self._trend_cooldown: int = 0
        self._trend_cooldown_sign: int = 0

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
        self._open_bonus = 0.0
        self._last_trade_entry_price = 0.0
        self._last_trade_exit_price = 0.0
        self._last_trade_entry_idx = -1
        self._last_trade_exit_idx = -1
        self._last_trade_entry_ts = None
        self._last_trade_exit_ts = None
        self._last_trade_side = 0
        self._last_trade_size = 0.0
        self._last_trade_penalty = 0.0
        self._open_cost = 0.0
        self._current_trade_penalty = 0.0
        self._current_trade_bonus = 0.0
        self._entry_idx = -1
        self._entry_timestamp = None
        self._pending_action = None
        self._trend_cooldown = 0
        self._trend_cooldown_sign = 0
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
        trend_state, trend_strength = self._trend_snapshot(cur_idx)

        if not getattr(self.cfg, "allow_intrabar_closes", True):
            if self._pending_action is not None:
                action = self._pending_action
                self._pending_action = None
            else:
                if not (self._step == 0 and self._pos == 0):
                    self._pending_action = action
                    action = self._pos + 1

        desired_pos = action - 1  # map {0:-1,1:0,2:+1}
        idle_throttle_penalty = float(getattr(self.cfg, "trend_throttle_idle_penalty", 0.0))
        reward_adjust = 0.0

        if self._trend_cooldown > 0:
            if desired_pos == -self._trend_cooldown_sign:
                desired_pos = 0
                if idle_throttle_penalty > 0.0:
                    reward_adjust -= idle_throttle_penalty
            self._trend_cooldown -= 1
            if self._trend_cooldown == 0:
                self._trend_cooldown_sign = 0
        else:
            if desired_pos != self._pos and desired_pos != 0:
                if self._should_throttle(desired_pos, trend_state, trend_strength):
                    desired_pos = 0
                    cooldown_len = int(getattr(self.cfg, "trend_throttle_cooldown", 0))
                    if cooldown_len > 0:
                        self._trend_cooldown = cooldown_len
                        self._trend_cooldown_sign = int(np.sign(trend_state))
                    if idle_throttle_penalty > 0.0:
                        reward_adjust -= idle_throttle_penalty

        if desired_pos != self._pos:
            # close current position first
            if self._pos != 0:
                close_reason = "flip" if desired_pos != 0 else "close"
                reward += self._close_position(price, reason=close_reason)
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

        # Removido: penalidade por barra contra tendência. Agora aplicamos um custo único na abertura
        # (registrado em _open_position). Mantemos este trecho sem efeito por compatibilidade.

        # Update trailing stop
        reward += self._maybe_apply_trailing(next_price, next_low, next_high, next_atr)

        # Aplica eventuais ajustes do throttle (idle penalty quando bloqueado)
        reward += reward_adjust

        # Penalidade por ficar flat (sem posição)
        if self._pos == 0 and self._idle_penalty_per_step > 0.0:
            reward -= self._idle_penalty_per_step

        reward_scale = max(1.0, float(getattr(self.cfg, "reward_scale_divisor", 1.0)))
        reward /= reward_scale

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
            "trade_penalty": float(self._last_trade_penalty) if self._just_closed else 0.0,
            "timestamp": self._format_timestamp(self._resolve_timestamp(self._start_idx + self._step)),
        }
        # reset do marcador de fechamento para o próximo passo
        self._just_closed = False
        return obs, reward, done, info

    # ------------------------------------------------------------------
    def _transaction_cost(self, price: float, size: float) -> float:
        notional = price * size
        return notional * (self.cfg.fee_pct + self.cfg.slippage_pct)

    def _turnover_penalty_value(self, notional: float) -> float:
        pct = max(0.0, float(getattr(self.cfg, "turnover_penalty_pct", 0.0)))
        if pct > 0.0 and notional > 0.0:
            return notional * pct
        return max(0.0, float(getattr(self.cfg, "turnover_penalty", 0.0)))

    def _flip_exit_penalty_value(self, notional: float) -> float:
        pct = max(0.0, float(getattr(self.cfg, "flip_exit_penalty_pct", 0.0)))
        if pct > 0.0 and notional > 0.0:
            return notional * pct
        return max(0.0, float(getattr(self.cfg, "flip_exit_penalty", 0.0)))

    def _close_position(self, price: float, reason: str = "close") -> float:
        entry_price = self._entry_price
        pnl = self._pos * self._pos_size * (price - entry_price)
        side = self._pos
        size = self._pos_size
        cost = self._transaction_cost(price, size)
        notional = abs(size * price)
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
        # bônus/malus por duração do trade + bônus de alinhamento na entrada
        bonus = 0.0
        if self.cfg.hold_bonus_alpha > 0.0:
            raw_bonus = self.cfg.hold_bonus_alpha * duration_bars * pnl
            if getattr(self.cfg, "hold_bonus_positive_only", False) and raw_bonus < 0.0:
                bonus = 0.0
            else:
                bonus = raw_bonus
        # adiciona bônus de alinhamento (se configurado) calculado na abertura
        # aplica somente para trades com duração >= 2 barras para desencorajar scalps de 1 barra
        if duration_bars >= 2 and getattr(self, "_open_bonus", 0.0) != 0.0:
            bonus += float(self._open_bonus)
        flip_penalty = self._flip_exit_penalty_value(notional) if reason == "flip" else 0.0
        # PnL total realizado do trade (inclui custos de entrada, saída, penalidades e bônus)
        trade_penalty_total = float(self._current_trade_penalty + flip_penalty)
        trade_pnl_total = pnl - self._open_cost - cost - flip_penalty + bonus
        # Sinaliza fechamento para consumo externo (walk-forward/relatórios)
        self._just_closed = True
        self._last_trade_pnl = float(trade_pnl_total)
        self._last_trade_bars = int(duration_bars)
        self._last_trade_reason = reason
        self._last_trade_cost = float(self._open_cost + cost)
        self._last_trade_bonus = float(bonus)
        self._last_trade_gross = float(pnl)
        self._last_trade_penalty = trade_penalty_total
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
        self._open_bonus = 0.0
        self._current_trade_penalty = 0.0
        if mode == "legacy":
            return pnl - cost - flip_penalty
        return -cost - flip_penalty + pnl + bonus

    def _compute_adaptive_atr_mult(self, atr: float) -> Tuple[float, float]:
        """Calcula multiplicadores ATR adaptativos baseados na volatilidade histórica."""
        window = getattr(self.cfg, "adaptive_stop_window", 50)
        cur_idx = self._start_idx + self._step
        start_idx = max(0, cur_idx - window)
        if start_idx >= cur_idx:
            # sem histórico suficiente, usa valores base
            return self.cfg.stop_atr_mult, self.cfg.trail_atr_mult
        hist_atr = self.features.iloc[start_idx:cur_idx]["atr_14"].mean()
        if hist_atr <= 0:
            return self.cfg.stop_atr_mult, self.cfg.trail_atr_mult
        ratio = atr / hist_atr
        # limita o ajuste para evitar extremos
        ratio = max(0.5, min(2.0, ratio))
        stop_mult = self.cfg.stop_atr_mult * ratio
        trail_mult = self.cfg.trail_atr_mult * ratio
        return stop_mult, trail_mult

    def _trend_snapshot(self, idx: int) -> Tuple[float, float]:
        """Retorna (estado, força) da tendência HTF no índice informado."""
        if idx < 0 or idx >= len(self.features):
            return 0.0, 0.0
        row = self.features.iloc[idx]
        state = row.get("htf_trend_state", 0.0)
        strength = row.get("htf_trend_strength", 0.0)
        try:
            state = float(state)
        except (TypeError, ValueError):
            state = 0.0
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            strength = 0.0
        return state, strength

    def _trend_alignment_penalty(
        self,
        pos: int,
        idx: Optional[int] = None,
        *,
        multiplier: Optional[float] = None,
    ) -> float:
        """Penalidade proporcional à força da tendência superior quando posicionado contra ela."""
        if pos == 0:
            return 0.0
        idx = self._start_idx + self._step if idx is None else idx
        state, strength = self._trend_snapshot(idx)
        if not np.isfinite(state) or state == 0.0:
            return 0.0
        if np.sign(state) == np.sign(pos):
            return 0.0
        if not np.isfinite(strength) or strength == 0.0:
            strength = 1.0
        mult = multiplier if multiplier is not None else 1.0
        coef_pct = abs(float(getattr(self.cfg, "trend_penalty_coef_pct", 0.0)))
        base_coef = abs(float(getattr(self.cfg, "trend_penalty_coef", 0.0)))
        if coef_pct > 0.0:
            price = float(self.df.iloc[idx]["close"])
            notional = abs(self._pos_size * price)
            base = notional * coef_pct if notional > 0.0 else base_coef
        else:
            base = base_coef
        if base <= 0.0:
            return 0.0
        penalty = base * abs(strength) * max(1.0, mult)
        return penalty

    def _trend_alignment_bonus(
        self,
        pos: int,
        idx: Optional[int] = None,
        *,
        multiplier: Optional[float] = None,
    ) -> float:
        """Bônus proporcional quando a posição está ALINHADA à tendência HTF.

        Retorna 0 quando não há alinhamento ou coeficientes desabilitados.
        """
        if pos == 0:
            return 0.0
        idx = self._start_idx + self._step if idx is None else idx
        state, strength = self._trend_snapshot(idx)
        if not np.isfinite(state) or state == 0.0:
            return 0.0
        if np.sign(state) != np.sign(pos):
            return 0.0
        if not np.isfinite(strength) or strength == 0.0:
            strength = 1.0
        mult = multiplier if multiplier is not None else 1.0
        coef_pct = abs(float(getattr(self.cfg, "trend_bonus_coef_pct", 0.0)))
        base_coef = abs(float(getattr(self.cfg, "trend_bonus_coef", 0.0)))
        if coef_pct > 0.0:
            price = float(self.df.iloc[idx]["close"])
            notional = abs(self._pos_size * price)
            base = notional * coef_pct if notional > 0.0 else base_coef
        else:
            base = base_coef
        if base <= 0.0:
            return 0.0
        bonus = base * abs(strength) * max(1.0, mult)
        return bonus

    def _should_throttle(self, desired_pos: int, trend_state: float, trend_strength: float) -> bool:
        threshold = float(getattr(self.cfg, "trend_throttle_threshold", 0.0))
        cooldown = int(getattr(self.cfg, "trend_throttle_cooldown", 0))
        if desired_pos == 0 or threshold <= 0.0 or cooldown <= 0:
            return False
        if not np.isfinite(trend_state) or trend_state == 0.0:
            return False
        if np.sign(trend_state) != -np.sign(desired_pos):
            return False
        if not np.isfinite(trend_strength):
            return False
        # Verifica possibilidade de override por divergência forte (Spread/Pattern)
        if bool(getattr(self.cfg, "trend_throttle_use_divergence_override", False)):
            if self._has_strong_divergence(desired_pos, trend_state):
                return False
        return abs(trend_strength) >= threshold

    def _has_strong_divergence(self, desired_pos: int, trend_state: float) -> bool:
        """Retorna True se houver evidência forte (Spread/Pattern) contra a tendência HTF.

        - desired_pos: +1 para long, -1 para short
        - trend_state: +1 tendência alta, -1 tendência baixa
        """
        idx = self._start_idx + self._step
        if idx < 0 or idx >= len(self.features):
            return False
        row = self.features.iloc[idx]
        spread_thr = float(getattr(self.cfg, "trend_throttle_spread_thr", 1.5))
        patt_thr = float(getattr(self.cfg, "trend_throttle_pattern_thr", 0.6))
        # Indicadores de spread
        spread_z_z = row.get("spread_z_zscore", 0.0)
        try:
            spread_z_z = float(spread_z_z)
        except (TypeError, ValueError):
            spread_z_z = 0.0
        spread_ok = abs(spread_z_z) >= spread_thr

        # Indicadores de padrão
        hammer = float(row.get("hammer_roll3", 0.0) or 0.0)
        shooting = float(row.get("shooting_star_roll3", 0.0) or 0.0)
        engulf_diff = float(row.get("engulf_diff_roll5", 0.0) or 0.0)
        body_atr = float(row.get("body_atr", 0.0) or 0.0)

        pattern_bearish = max(shooting, float(row.get("bearish_engulf_flag", 0.0) or 0.0), max(0.0, -body_atr))
        pattern_bullish = max(hammer, float(row.get("bullish_engulf_flag", 0.0) or 0.0), max(0.0, body_atr))
        # Ajusta por sinal de engolfo médio
        if engulf_diff < 0:
            pattern_bearish = max(pattern_bearish, min(1.0, -engulf_diff))
        elif engulf_diff > 0:
            pattern_bullish = max(pattern_bullish, min(1.0, engulf_diff))

        # Regras de override específicas por lado desejado
        if desired_pos < 0 and trend_state > 0:  # quer short em tendência de alta
            return spread_ok or (pattern_bearish >= patt_thr)
        if desired_pos > 0 and trend_state < 0:  # quer long em tendência de baixa
            return spread_ok or (pattern_bullish >= patt_thr)
        return False

    def _open_position(self, pos: int, price: float, atr: float) -> float:
        self._pos = pos
        self._entry_price = price
        self._pos_size = self._compute_position_size(price)
        self._entry_step = self._step
        self._entry_idx = self._start_idx + self._step
        self._entry_timestamp = self._resolve_timestamp(self._entry_idx)
        # Calcula stops adaptativos
        stop_mult, trail_mult = self._compute_adaptive_atr_mult(atr)
        if pos > 0:
            self._trailing = price - stop_mult * atr
            self._peak_price = price
            self._trough_price = None
        else:
            self._trailing = price + stop_mult * atr
            self._trough_price = price
            self._peak_price = None
        cost = self._transaction_cost(price, self._pos_size)
        # registra custo/bonus de entrada para contabilizar no PnL do trade no fechamento
        entry_penalty_mult = float(getattr(self.cfg, "trend_penalty_entry_mult", 1.0))
        idx = self._start_idx + self._step
        entry_penalty = self._trend_alignment_penalty(pos, idx=idx, multiplier=entry_penalty_mult)
        notional = abs(self._pos_size * price)
        turnover_penalty = self._turnover_penalty_value(notional) if notional > 0.0 else 0.0
        entry_penalty_total = entry_penalty + turnover_penalty
        self._current_trade_penalty = float(entry_penalty_total)
        self._open_cost = float(cost + entry_penalty_total)
        # cálculo do bônus por alinhamento com a tendência (aplicado somente no fechamento)
        entry_bonus_mult = float(getattr(self.cfg, "trend_bonus_entry_mult", getattr(self.cfg, "trend_penalty_entry_mult", 1.0)))
        entry_bonus = self._trend_alignment_bonus(pos, idx=idx, multiplier=entry_bonus_mult)
        self._open_bonus = float(entry_bonus)
        self._current_trade_bonus = float(entry_bonus)
        # Nota: não creditamos o bônus na abertura; será aplicado no fechamento conforme regra de duração
        return -(cost + entry_penalty_total)

    def _maybe_apply_trailing(
        self, next_price: float, next_low: float, next_high: float, next_atr: float
    ) -> float:
        if self._pos == 0 or self._trailing is None:
            return 0.0
        # evita fechar imediatamente após abrir (mesmo candle) quando desativado
        if (
            not getattr(self.cfg, "allow_intrabar_closes", True)
            and self._pos != 0
            and self._step == getattr(self, "_entry_step", 0)
        ):
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
        # Kelly Criterion simplificado: usa volatilidade como proxy para risco
        kelly_fraction = getattr(self.cfg, "kelly_fraction", 0.1)
        var_confidence = getattr(self.cfg, "var_confidence", 0.95)
        # Calcula volatilidade histórica (std de retornos)
        window = getattr(self.cfg, "adaptive_stop_window", 50)
        cur_idx = self._start_idx + self._step
        start_idx = max(0, cur_idx - window)
        if start_idx < cur_idx:
            returns = self.df.iloc[start_idx:cur_idx]["close"].pct_change().dropna()
            if len(returns) > 0:
                vol = returns.std()
                # Kelly: f = (expected_return / variance), mas aqui expected_return ~ 0, então f = kelly_fraction / vol
                kelly_adj = kelly_fraction / max(vol, 0.01)  # evita divisão por zero
                # VaR: limita exposição para não exceder perda máxima
                z_score = np.abs(np.percentile(np.random.normal(0, 1, 1000), (1 - var_confidence) * 100))
                var_limit = equity * 0.1  # assume perda máxima de 10% do equity
                var_size = var_limit / (px * vol * z_score) if vol > 0 else float("inf")
                # Combina Kelly e VaR
                notional = equity * min(kelly_adj, var_size / equity)
            else:
                notional = equity * kelly_fraction
        else:
            notional = equity * kelly_fraction
        # Aplica leverage e cap
        notional = min(notional * max(self.cfg.leverage, 0.0), max_notional) if max_notional > 0 else notional * max(self.cfg.leverage, 0.0)
        return notional / px

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
