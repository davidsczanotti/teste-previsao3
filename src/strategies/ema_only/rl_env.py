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
    entry_bonus_fast_over_slow: float = 0.0  # bônus ao abrir com ema_fast > ema_slow (ou < para short)
    entry_bonus_full_trend: float = 0.0      # bônus extra ao abrir com fast>slow>ref (ou invertido para short)
    override_long_gate: bool = False         # se True, não bloqueia entradas long por consenso/tendência/distância
    override_short_gate: bool = False        # se True, não bloqueia entradas short por consenso/tendência/distância
    bear_regime_threshold: float = 0.45      # limiar para detectar regime de baixa (exp_trend/ref < threshold)
    block_long_in_bear: bool = False         # se True, bloqueia novas entradas long em regime de baixa
    bear_consensus_short_threshold: float = 0.3  # limiar especial de consenso para short em regime de baixa
    exit_on_fast_slow_cross: bool = True     # se True, fecha posição quando ema_fast cruza ema_slow contra a posição
    risk_per_trade_pct: float = 0.0          # sizing dinâmico: % do capital a arriscar por trade (0 = desliga)
    max_position_pct: float = 0.95           # limite superior da posição em relação ao capital (para pagar taxas)
    # Recompensa por metas mensais (tiered)
    monthly_target_tiers: Optional[list[tuple[float, float]]] = None  # lista de (ret_min, bonus)
    monthly_shortfall_penalty: float = 0.0   # penalidade se retorno mensal < 0


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
        self.position_size = 0.0  # tamanho do lote em BTC para o trade atual
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

    def _apply_cost(self, price: float, lot_size: float):
        cost = (self.cfg.fee_pct + self.cfg.slippage_pct) * price * lot_size
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
        # Confirmação extra de tendência: exige que exp_trend e exp_ref concordem
        # e que o consenso esteja minimamente alto para considerar entradas.
        trend_long_ok = True
        trend_short_ok = True
        if "exp_trend" in self.features.columns and "exp_ref" in self.features.columns:
            try:
                exp_trend_val = float(self.features["exp_trend"].iloc[self.idx])
                exp_ref_val = float(self.features["exp_ref"].iloc[self.idx])
            except Exception:
                exp_trend_val = np.nan
                exp_ref_val = np.nan
            if not (np.isfinite(exp_trend_val) and np.isfinite(exp_ref_val)):
                trend_long_ok = False
                trend_short_ok = False
            else:
                # Simetria: long se ambos >= 0.5; short se ambos < 0.5
                trend_long_ok = (exp_trend_val >= 0.5) and (exp_ref_val >= 0.5)
                trend_short_ok = (exp_trend_val < 0.5) and (exp_ref_val < 0.5)
        # Flags de regime de baixa
        in_bear_regime = False
        in_bear_strict = False
        try:
            if "exp_trend" in self.features.columns and "exp_ref" in self.features.columns:
                exp_tr = float(self.features["exp_trend"].iloc[self.idx])
                exp_rf = float(self.features["exp_ref"].iloc[self.idx])
                if np.isfinite(exp_tr) and np.isfinite(exp_rf):
                    in_bear_regime = (exp_tr < 0.4) or (exp_rf < 0.4)
                    bear_thr = float(getattr(self.cfg, "bear_regime_threshold", 0.45))
                    in_bear_strict = (exp_tr < bear_thr) and (exp_rf < bear_thr)
        except Exception:
            in_bear_regime = False
            in_bear_strict = False
        # Slope da ref_ema (EMA longa) para filtrar tendências flat
        slope_ref = 0.0
        if "ref_ema" in self.features.columns and self.idx > 0:
            try:
                prev_ref = float(self.features["ref_ema"].iloc[self.idx - 1])
                curr_ref = float(self.features["ref_ema"].iloc[self.idx])
                if np.isfinite(prev_ref) and np.isfinite(curr_ref):
                    slope_ref = curr_ref - prev_ref
            except Exception:
                slope_ref = 0.0

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
                # Long exige tendência de alta: fast > slow e slope_ref > 0
                fast = float(self.features["ema_fast"].iloc[self.idx]) if "ema_fast" in self.features.columns else np.nan
                slow = float(self.features["ema_slow"].iloc[self.idx]) if "ema_slow" in self.features.columns else np.nan
                fast_above_slow = np.isfinite(fast) and np.isfinite(slow) and fast > slow
                # Bloqueio extra: regime de baixa estrito
                if in_bear_strict and getattr(self.cfg, "block_long_in_bear", False):
                    reward -= self.cfg.gating_penalty
                    desired_pos = 0
                else:
                    block = False if self.cfg.override_long_gate else not (ref_long_ok and cons_long_ok and trend_long_ok and fast_above_slow and slope_ref > 0.0)
                    if (
                        not block
                        and max_long > 0.0
                        and dist_fast is not None
                        and dist_fast > max_long
                    ):
                        block = True
                    if block:
                        reward -= self.cfg.gating_penalty
                        desired_pos = 0
            elif desired_pos == -1:
                # Short exige tendência de baixa: fast < slow e slope_ref < 0
                fast = float(self.features["ema_fast"].iloc[self.idx]) if "ema_fast" in self.features.columns else np.nan
                slow = float(self.features["ema_slow"].iloc[self.idx]) if "ema_slow" in self.features.columns else np.nan
                fast_below_slow = np.isfinite(fast) and np.isfinite(slow) and fast < slow
                if in_bear_strict and getattr(self.cfg, "bear_consensus_short_threshold", 0.0) > 0.0:
                    cons_short_ok = cons <= float(self.cfg.bear_consensus_short_threshold)
                block = False if self.cfg.override_short_gate else not (ref_short_ok and cons_short_ok and trend_short_ok and fast_below_slow and slope_ref < 0.0)
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
        # Stop de regime: se estamos long e entramos em regime de baixa, forçamos zerar
        if self.position == 1 and in_bear_regime:
            desired_pos = 0
        # Stop técnico: cruza fast/slow contra a posição
        if self.cfg.exit_on_fast_slow_cross and "ema_fast" in self.features.columns and "ema_slow" in self.features.columns:
            try:
                ef_exit = float(self.features["ema_fast"].iloc[self.idx])
                es_exit = float(self.features["ema_slow"].iloc[self.idx])
                if np.isfinite(ef_exit) and np.isfinite(es_exit):
                    if self.position == 1 and ef_exit < es_exit:
                        desired_pos = 0
                    elif self.position == -1 and ef_exit > es_exit:
                        desired_pos = 0
            except Exception:
                pass

        # --- Fecha posição atual, se necessário -----------------------------
        if desired_pos != self.position and self.position != 0:
            side = self.position
            if side == 1:
                pnl = (price - self.entry_price) * self.position_size
            else:  # side == -1
                pnl = (self.entry_price - price) * self.position_size
            self.equity += pnl
            cost = self._apply_cost(price, self.position_size)
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
            self.position_size = 0.0
            self.stop_price = 0.0
            self._entry_idx = -1

        # --- Abre nova posição, se desejado ---------------------------------
        if desired_pos != self.position and desired_pos != 0:
            self.position = desired_pos
            # Sizing dinâmico opcional baseado no risco por trade e stop ATR
            lot = self.cfg.lot_size
            if atr_value is not None and self.cfg.risk_per_trade_pct > 0 and self.cfg.atr_stop_mult > 0:
                stop_dist = self.cfg.atr_stop_mult * atr_value
                if stop_dist > 0:
                    risk_amount = self.equity * self.cfg.risk_per_trade_pct
                    max_pos_usd = self.equity * float(getattr(self.cfg, "max_position_pct", 0.95))
                    lot_est = risk_amount / stop_dist
                    # converte usd -> lot em BTC dividindo pelo preço
                    lot = lot_est / price
                    lot_usd = lot * price
                    if lot_usd > max_pos_usd:
                        lot = max_pos_usd / price
            self.position_size = lot
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
            cost = self._apply_cost(price, self.position_size)
            self.equity -= self.cfg.trade_penalty
            # Shaping de entrada por degraus de alinhamento de EMAs
            ef = np.nan
            es = np.nan
            ref = np.nan
            try:
                ef = float(self.features["ema_fast"].iloc[self.idx])
                es = float(self.features["ema_slow"].iloc[self.idx])
                ref = float(self.features["ref_ema"].iloc[self.idx]) if "ref_ema" in self.features.columns else np.nan
            except Exception:
                pass
            if np.isfinite(ef) and np.isfinite(es):
                trend_level = 0
                if self.position == 1:
                    if ef > es:
                        trend_level = 1
                        if np.isfinite(ref) and es > ref:
                            trend_level = 2
                elif self.position == -1:
                    if ef < es:
                        trend_level = 1
                        if np.isfinite(ref) and es < ref:
                            trend_level = 2
                if trend_level == 1 and self.cfg.entry_bonus_fast_over_slow != 0.0:
                    reward += float(self.cfg.entry_bonus_fast_over_slow)
                elif trend_level == 2:
                    # pode optar por somar ou substituir; aqui substituímos pelo valor do patamar cheio
                    bonus = float(self.cfg.entry_bonus_full_trend)
                    if bonus == 0.0 and self.cfg.entry_bonus_fast_over_slow != 0.0:
                        # fallback: soma se full_trend não foi configurado
                        bonus = float(self.cfg.entry_bonus_fast_over_slow)
                    reward += bonus
            # Bônus por entrar em pullback (perto/abaixo da ema_fast para long; acima para short)
            if self.cfg.pullback_entry_bonus != 0.0 and np.isfinite(ef) and ef != 0.0:
                dist_fast_entry = (price - ef) / ef
                if self.position == 1 and dist_fast_entry <= 0.0:
                    reward += float(self.cfg.pullback_entry_bonus)
                elif self.position == -1 and dist_fast_entry >= 0.0:
                    reward += float(self.cfg.pullback_entry_bonus)

        # mark-to-market unrealized
        if self.position == 1:
            unreal = (price - self.entry_price) * self.position_size
        elif self.position == -1:
            unreal = (self.entry_price - price) * self.position_size
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
        if self.cfg.consensus_bonus > 0 and "experts_mean" in self.features.columns:
            cons = float(self.features["experts_mean"].iloc[self.idx])
            if self.position == 1 and cons >= self.cfg.consensus_threshold:
                reward += self.cfg.consensus_bonus
            elif self.position == -1 and cons <= (1.0 - self.cfg.consensus_threshold):
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
                # Aplica bônus/penalidade por metas mensais (alvo de retorno)
                if hasattr(self.cfg, "monthly_target_tiers") and self.cfg.monthly_target_tiers:
                    month_ret = (self.last_equity - self.cfg.init_equity) / self.cfg.init_equity
                    bonus = 0.0
                    # tiers devem estar ordenados por ret_min crescente
                    for ret_min, tier_bonus in self.cfg.monthly_target_tiers:
                        if month_ret >= ret_min:
                            bonus = tier_bonus
                    out_reward += bonus
                    if month_ret < 0 and getattr(self.cfg, "monthly_shortfall_penalty", 0.0) != 0.0:
                        out_reward -= float(self.cfg.monthly_shortfall_penalty)
                self._current_month = next_month
        else:
            out_reward = reward

        out_reward *= self.cfg.reward_scale

        self.idx += 1
        if self.idx >= len(self.df) - 1:
            terminated = True

        obs = self._obs() if not terminated else np.zeros_like(self._obs())
        return obs, float(out_reward), terminated, truncated, info
