from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


@dataclass
class RLConfig:
    fee_pct: float = 0.0004
    slippage_pct: float = 0.0005
    lot_size: float = 0.001
    init_equity: float = 1000.0
    trade_penalty: float = 0.0  # penalidade fixa por trade (costumeiramente 0)
    dd_threshold_pct: float = 0.02  # drawdown intraday permitido antes de penalizar
    dd_penalty: float = 0.0        # malus por cruzar o threshold
    min_hold_bars: int = 0         # penaliza churn se fechar antes disso
    churn_penalty: float = 0.0
    align_bonus: float = 0.0       # bônus por estar alinhado (ema_fast>ema_slow>ref)
    realized_bonus_coef: float = 0.0  # bônus proporcional ao pnl realizado em trades positivos
    atr_risk_scale: float = 0.0    # >0 reduz recompensa em alta vol (divide por 1+k*atr_rel)
    enforce_ref_bias: bool = True  # usa EMA de referência como viés direcional
    reward_scale: float = 1.0      # escala final da recompensa
    consensus_bonus: float = 0.0   # bônus por consenso de experts
    consensus_threshold: float = 0.66  # limiar de consenso (ex.: 2 de 3)
    vol_max_atr_rel: float = 0.0   # se >0, bloqueia/pune entradas com atr_rel acima
    vol_penalty: float = 0.0       # malus ao tentar abrir em vol alta
    gating_penalty: float = 0.0    # malus se ação de entrada for bloqueada pelo gate
    turnover_penalty: float = 0.0  # penaliza cada fechamento (turnover) — pode ficar 0
    living_cost_per_episode: float = 0.0  # custo fixo distribuído ao longo do episódio
    use_monthly_reward: bool = False      # se True, reward é agregado por mês
    allow_short: bool = True              # permite posições short (pos=-1)
    trend_entry_bonus: float = 0.0        # bônus ao abrir trade a favor da tendência EMA
    trend_entry_penalty: float = 0.0      # penalidade ao abrir trade contra a tendência EMA
    trend_flip_penalty: float = 0.0       # penalidade ao inverter posição em plena tendência
    max_long_entry_dist_fast_pct: float = 0.0   # distância máxima acima da ema_fast para abrir long
    max_short_entry_dist_fast_pct: float = 0.0  # distância máxima abaixo da ema_fast para abrir short
    pullback_entry_bonus: float = 0.0     # bônus ao abrir perto/abaixo da ema_fast (long) ou acima (short)
    trend_exit_penalty: float = 0.0       # penalidade ao fechar ainda do lado “bom” da ema_fast
    atr_stop_mult: float = 0.0            # múltiplo de ATR para stop inicial
    atr_trail_mult: float = 0.0           # múltiplo de ATR para trailing stop


class EmaEnv(gym.Env):
    """
    Ambiente RL simples para seguir EMAs (long/short, 1 posição).

    Convenção das ações:
      - 0: manter posição atual (hold)
      - 1: alvo = long (+1)
      - 2: alvo = short (-1)  [se allow_short=True]
      - 3: alvo = flat (0)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        cfg: RLConfig,
        norm_mean: Optional[pd.Series] = None,
        norm_std: Optional[pd.Series] = None,
    ) -> None:
        assert len(df) == len(features), "df e features precisam ter mesmo tamanho"
        self.df = df.reset_index(drop=True)
        self.features = features.reset_index(drop=True)
        self.cfg = cfg
        # 4 ações: hold, long, short, flat
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32
        )
        self._norm_mean = features.mean() if norm_mean is None else norm_mean
        std = features.std().replace(0.0, 1.0) if norm_std is None else norm_std
        self._norm_std = std.replace(0.0, 1.0)
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.idx = 0
        self.position = 0  # -1 short, 0 flat, 1 long
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.equity = self.cfg.init_equity
        self.last_equity = self.equity
        self.trades = 0
        self._entry_idx: int = -1
        self._peak_equity_step: float = self.cfg.init_equity
        # custo fixo distribuído por barra (ex.: contas a pagar)
        total_bars = max(1, len(self.df))
        living_total = float(getattr(self.cfg, "living_cost_per_episode", 0.0))
        self._living_penalty_per_step = living_total / float(total_bars)
        # controle de recompensa mensal
        self._use_monthly = bool(getattr(self.cfg, "use_monthly_reward", False))
        self._month_reward_accum: float = 0.0
        if "Date" in self.df.columns:
            try:
                first_ts = pd.to_datetime(self.df["Date"].iloc[0])
                self._current_month = first_ts.to_period("M")
            except Exception:
                self._current_month = None
        else:
            self._current_month = None
        return self._obs(), {}

    def _obs(self):
        feats = (self.features.iloc[self.idx] - self._norm_mean) / self._norm_std
        return feats.to_numpy(dtype=np.float32)

    def _price(self, i: int) -> float:
        return float(self.df["close"].iloc[i])

    def _ref_price(self, i: int) -> float:
        if "ref_ema" in self.df.columns:
            return float(self.df["ref_ema"].iloc[i])
        return float(self.df["close"].iloc[i])

    def _apply_cost(self, price: float):
        cost = (self.cfg.fee_pct + self.cfg.slippage_pct) * price * self.cfg.lot_size
        self.equity -= cost
        return cost

    def step(self, action: int):
        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        price = self._price(self.idx)
        # ATR absoluto (a partir de atr_rel) para uso em stops
        atr_value = None
        if "atr_rel" in self.features.columns:
            try:
                atr_rel = float(self.features["atr_rel"].iloc[self.idx])
                if np.isfinite(atr_rel):
                    atr_value = atr_rel * price
            except Exception:
                atr_value = None

        # Atualiza trailing stop se já houver posição aberta
        if self.position != 0 and atr_value is not None and self.cfg.atr_trail_mult > 0.0:
            if self.position == 1:
                trail = price - self.cfg.atr_trail_mult * atr_value
                self.stop_price = trail if self.stop_price == 0.0 else max(self.stop_price, trail)
            elif self.position == -1:
                trail = price + self.cfg.atr_trail_mult * atr_value
                self.stop_price = trail if self.stop_price == 0.0 else min(self.stop_price, trail)

        # --- Filtros de viés/ref e consenso ---------------------------------
        ref_price = self._ref_price(self.idx)
        ref_long_ok = True
        ref_short_ok = True
        if self.cfg.enforce_ref_bias:
            # long prefere preço acima da ref; short, abaixo
            ref_long_ok = price >= ref_price
            ref_short_ok = price <= ref_price

        # Vol throttle
        if self.cfg.vol_max_atr_rel > 0 and "atr_rel" in self.features.columns:
            atr_rel = float(self.features["atr_rel"].iloc[self.idx])
            if atr_rel > self.cfg.vol_max_atr_rel:
                ref_long_ok = False
                ref_short_ok = False
                if self.cfg.vol_penalty > 0:
                    reward -= self.cfg.vol_penalty
        cons = 0.5
        if "experts_mean" in self.features.columns:
            cons = float(self.features["experts_mean"].iloc[self.idx])
        thr = float(self.cfg.consensus_threshold)
        cons_long_ok = cons >= thr
        cons_short_ok = cons <= (1.0 - thr)

        # --- Decodifica ação em posição alvo --------------------------------
        desired_pos = self.position
        if action == 1:
            desired_pos = 1
        elif action == 2 and self.cfg.allow_short:
            desired_pos = -1
        elif action == 3:
            desired_pos = 0
        # action == 0 => hold

        # Gate só atua em entradas a partir de cash (position == 0)
        if self.position == 0 and desired_pos != 0:
            # Distância do preço até a ema_fast (para evitar comprar topo/vender fundo)
            dist_fast = None
            if "ema_fast" in self.features.columns:
                try:
                    ef = float(self.features["ema_fast"].iloc[self.idx])
                    if np.isfinite(ef) and ef != 0.0:
                        dist_fast = (price - ef) / ef  # >0: preço acima da ema_fast
                except Exception:
                    dist_fast = None

            max_long = float(getattr(self.cfg, "max_long_entry_dist_fast_pct", 0.0))
            max_short = float(getattr(self.cfg, "max_short_entry_dist_fast_pct", 0.0))

            if desired_pos == 1:
                block = not (ref_long_ok and cons_long_ok)
                if (
                    not block
                    and max_long > 0.0
                    and dist_fast is not None
                    and dist_fast > max_long
                ):
                    # preço está esticado demais acima da ema_fast: evita comprar topo
                    block = True
                if block:
                    reward -= self.cfg.gating_penalty
                    desired_pos = 0
            elif desired_pos == -1:
                block = not (ref_short_ok and cons_short_ok)
                if (
                    not block
                    and max_short > 0.0
                    and dist_fast is not None
                    and dist_fast < -max_short
                ):
                    # preço muito abaixo da ema_fast: evita vender fundo
                    block = True
                if block:
                    reward -= self.cfg.gating_penalty
                    desired_pos = 0

        # Penaliza inversão de posição em plena tendência (1 -> -1 ou -1 -> 1)
        if (
            desired_pos != 0
            and self.position != 0
            and desired_pos != self.position
            and self.cfg.trend_flip_penalty != 0.0
        ):
            try:
                exp_trend = float(self.features.get("exp_trend", pd.Series([np.nan])).iloc[self.idx])
                exp_ref = float(self.features.get("exp_ref", pd.Series([np.nan])).iloc[self.idx])
            except Exception:
                exp_trend = np.nan
                exp_ref = np.nan
            if np.isfinite(exp_trend) and np.isfinite(exp_ref):
                # Regime de alta forte: exp_trend=1, exp_ref=1
                if exp_trend >= 0.5 and exp_ref >= 0.5:
                    # penaliza sair de long para short
                    if self.position == 1 and desired_pos == -1:
                        reward -= float(self.cfg.trend_flip_penalty)
                # Regime de baixa forte: exp_trend=0, exp_ref=0
                elif exp_trend < 0.5 and exp_ref < 0.5:
                    # penaliza sair de short para long
                    if self.position == -1 and desired_pos == 1:
                        reward -= float(self.cfg.trend_flip_penalty)

        # Stop ATR: força fechamento se o preço cruzar o stop_price
        if self.position == 1 and self.stop_price not in (0.0, np.nan):
            if price <= self.stop_price:
                desired_pos = 0
        elif self.position == -1 and self.stop_price not in (0.0, np.nan):
            if price >= self.stop_price:
                desired_pos = 0

        # --- Fecha posição atual, se necessário -----------------------------
        if desired_pos != self.position and self.position != 0:
            side = self.position
            if side == 1:
                pnl = (price - self.entry_price) * self.cfg.lot_size
            else:  # side == -1
                pnl = (self.entry_price - price) * self.cfg.lot_size
            self.equity += pnl
            cost = self._apply_cost(price)
            # custo fixo de turnover/trade (normalmente 0 neste setup)
            self.equity -= self.cfg.trade_penalty + self.cfg.turnover_penalty
            if pnl > 0 and self.cfg.realized_bonus_coef > 0:
                reward += self.cfg.realized_bonus_coef * pnl
            # Penaliza sair “cedo demais” enquanto o preço ainda está do lado bom da ema_fast
            if self.cfg.trend_exit_penalty != 0.0 and "ema_fast" in self.features.columns:
                try:
                    ef_close = float(self.features["ema_fast"].iloc[self.idx])
                    if np.isfinite(ef_close) and ef_close != 0.0:
                        dist_fast_close = (price - ef_close) / ef_close
                        # long: preço acima da ema_fast ainda favorável; short: abaixo
                        if side == 1 and dist_fast_close > 0.0:
                            reward -= float(self.cfg.trend_exit_penalty)
                        elif side == -1 and dist_fast_close < 0.0:
                            reward -= float(self.cfg.trend_exit_penalty)
                except Exception:
                    pass
            self.position = 0
            self.entry_price = 0.0
            self.stop_price = 0.0
            self._entry_idx = -1

        # --- Abre nova posição, se desejado ---------------------------------
        if desired_pos != self.position and desired_pos != 0:
            self.position = desired_pos
            self.entry_price = price
            # Stop ATR inicial para o novo trade
            if atr_value is not None and self.cfg.atr_stop_mult > 0.0:
                if self.position == 1:
                    self.stop_price = price - self.cfg.atr_stop_mult * atr_value
                elif self.position == -1:
                    self.stop_price = price + self.cfg.atr_stop_mult * atr_value
            else:
                self.stop_price = 0.0
            self._entry_idx = self.idx
            self.trades += 1
            cost = self._apply_cost(price)
            self.equity -= self.cfg.trade_penalty
            # Shaping: bônus/penalidade por entrar a favor/contra a tendência EMA
            ef = np.nan
            es = np.nan
            try:
                ef = float(self.features["ema_fast"].iloc[self.idx])
                es = float(self.features["ema_slow"].iloc[self.idx])
                if np.isfinite(ef) and np.isfinite(es):
                    trend = np.sign(ef - es)
                else:
                    trend = 0.0
            except Exception:
                trend = 0.0
            pos_sign = float(self.position)
            if trend != 0.0 and pos_sign != 0.0:
                align = trend * pos_sign  # >0: com a tendência, <0: contra
                if align > 0.0 and self.cfg.trend_entry_bonus != 0.0:
                    reward += float(self.cfg.trend_entry_bonus)
                elif align < 0.0 and self.cfg.trend_entry_penalty != 0.0:
                    reward -= float(self.cfg.trend_entry_penalty)
            # Bônus por entrar em pullback (perto/abaixo da ema_fast para long; acima para short)
            if self.cfg.pullback_entry_bonus != 0.0 and np.isfinite(ef) and ef != 0.0:
                dist_fast_entry = (price - ef) / ef
                if self.position == 1 and dist_fast_entry <= 0.0:
                    reward += float(self.cfg.pullback_entry_bonus)
                elif self.position == -1 and dist_fast_entry >= 0.0:
                    reward += float(self.cfg.pullback_entry_bonus)

        # mark-to-market unrealized
        if self.position == 1:
            unreal = (price - self.entry_price) * self.cfg.lot_size
        elif self.position == -1:
            unreal = (self.entry_price - price) * self.cfg.lot_size
        else:
            unreal = 0.0
        mtm_equity = self.equity + unreal
        step_reward = mtm_equity - self.last_equity
        # Drawdown penalty intraday (independe de long/short; só olha equity)
        if self.position != 0:
            peak = getattr(self, "_peak_equity_step", self.last_equity)
            peak = max(peak, mtm_equity)
            self._peak_equity_step = peak
            dd = (mtm_equity - peak) / peak if peak > 0 else 0.0
            if dd < -self.cfg.dd_threshold_pct:
                step_reward -= self.cfg.dd_penalty
        # ATR scaling (requires atr_rel feature at same index)
        if "atr_rel" in self.features.columns and self.cfg.atr_risk_scale > 0:
            atr_rel = float(self.features["atr_rel"].iloc[self.idx])
            step_reward /= (1.0 + self.cfg.atr_risk_scale * atr_rel)
        reward += step_reward

        # Custo fixo por barra (custo de vida / operar)
        if getattr(self, "_living_penalty_per_step", 0.0) != 0.0:
            reward -= self._living_penalty_per_step

        # Alignment bonus (ema_fast > ema_slow > ref) — só faz sentido para long
        if self.position == 1 and self.cfg.align_bonus > 0:
            ef = float(self.features["ema_fast"].iloc[self.idx])
            es = float(self.features["ema_slow"].iloc[self.idx])
            ref = float(self.features["ref_ema"].iloc[self.idx]) if "ref_ema" in self.features.columns else 0.0
            if ef > es and es > ref:
                reward += self.cfg.align_bonus

        # Consensus bonus
        if self.position == 1 and self.cfg.consensus_bonus > 0 and "experts_mean" in self.features.columns:
            cons = float(self.features["experts_mean"].iloc[self.idx])
            if cons >= self.cfg.consensus_threshold:
                reward += self.cfg.consensus_bonus

        # Churn penalty: fechar antes de min_hold_bars sem lucro
        if action == 2 and self.position == 1 and self.cfg.min_hold_bars > 0:
            bars_held = self.idx - self._entry_idx if self._entry_idx >= 0 else 0
            if bars_held < self.cfg.min_hold_bars:
                reward -= self.cfg.churn_penalty

        # Atualiza equity anterior
        self.last_equity = mtm_equity

        # Agregação mensal opcional: acumula reward por mês e só libera no fim de cada mês.
        out_reward: float
        if self._use_monthly:
            self._month_reward_accum += reward
            out_reward = 0.0
            cur_month = self._current_month
            # olha o próximo índice para detectar mudança de mês
            next_idx = self.idx + 1
            if "Date" in self.df.columns and cur_month is not None:
                if next_idx >= len(self.df) - 1:
                    month_changed = True
                    next_month = cur_month
                else:
                    try:
                        next_ts = pd.to_datetime(self.df["Date"].iloc[next_idx])
                        next_month = next_ts.to_period("M")
                    except Exception:
                        next_month = cur_month
                    month_changed = next_month != cur_month
            else:
                month_changed = False
                next_month = cur_month

            if month_changed or terminated:
                out_reward = self._month_reward_accum
                self._month_reward_accum = 0.0
                self._current_month = next_month
        else:
            out_reward = reward

        out_reward *= self.cfg.reward_scale

        self.idx += 1
        if self.idx >= len(self.df) - 1:
            terminated = True

        obs = self._obs() if not terminated else np.zeros_like(self._obs())
        return obs, float(out_reward), terminated, truncated, info
