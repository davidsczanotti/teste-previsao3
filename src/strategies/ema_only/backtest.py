from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class EmaOnlyParams:
    """
    Parâmetros da estratégia EMA-only.

    Obs.: os campos ema_period / slow_ema_period são mantidos para compatibilidade
    com testes e configs antigos; internamente eles representam a EMA rápida e lenta
    usadas no modo `ema_cross`.
    """

    # Parâmetros básicos (usados diretamente nos testes)
    ema_period: int
    slow_ema_period: Optional[int] = None
    signal_mode: str = "ema_cross"
    pullback_pct: float = 0.0
    use_trend_filter: bool = False
    trend_filter_period: Optional[int] = None
    use_cross: bool = False
    ref_filter_enabled: bool = False
    ref_ema_period: Optional[int] = None
    ref_buffer_pct: float = 0.0
    lot_size: float = 0.001
    fee_rate: float = 0.0004

    # Bloco de 6 MAs (3 SMAs + 3 EMAs)
    sma_fast_period: Optional[int] = None
    sma_mid_period: Optional[int] = None
    sma_slow_period: Optional[int] = None
    ema_fast_period: Optional[int] = None
    ema_mid_period: Optional[int] = None
    ema_slow_period: Optional[int] = None

    # Stops móveis
    trailing_stop_type: str = "none"  # "none", "atr_trailing", "percent_trailing", "ma_trailing"
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    atr_trail_mult: float = 1.0
    breakeven_rr: float = 1.0
    percent_trailing_pct: float = 0.01
    ma_trail_source: str = "ema_slow"  # uma de {sma_fast, sma_mid, sma_slow, ema_fast, ema_mid, ema_slow}
    ma_trail_offset_atr_mult: float = 1.0
    allow_short: bool = False

    def __post_init__(self) -> None:
        # Compatibilidade: se apenas ema_period/slow_ema_period forem passados (como nos testes),
        # usamos esses valores também como ema_fast_period/ema_slow_period.
        if self.ema_fast_period is None:
            self.ema_fast_period = self.ema_period
        if self.ema_slow_period is None and self.slow_ema_period is not None:
            self.ema_slow_period = self.slow_ema_period

        # Garante que ema_period/slow_ema_period reflitam os campos fast/slow, se estes forem definidos.
        if self.ema_fast_period is not None:
            self.ema_period = int(self.ema_fast_period)
        if self.ema_slow_period is not None:
            self.slow_ema_period = int(self.ema_slow_period)


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """EMA padrão (idêntica usada nos testes)."""
    return series.astype(float).ewm(span=period, adjust=False).mean()


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).rolling(window=period, min_periods=period).mean()


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    ATR clássico baseado em high/low/close.
    Se high/low não existirem, usa close como aproximação.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def _select_ma_for_trailing(
    df: pd.DataFrame,
    idx: int,
    params: EmaOnlyParams,
    ema_fast: pd.Series,
    ema_mid: Optional[pd.Series],
    ema_slow: pd.Series,
    sma_fast: Optional[pd.Series],
    sma_mid: Optional[pd.Series],
    sma_slow: Optional[pd.Series],
) -> Optional[float]:
    """Retorna o valor da média escolhida em ma_trail_source."""
    src = params.ma_trail_source
    if src == "ema_fast":
        return float(ema_fast.iloc[idx])
    if src == "ema_mid" and ema_mid is not None:
        return float(ema_mid.iloc[idx])
    if src == "ema_slow":
        return float(ema_slow.iloc[idx])
    if src == "sma_fast" and sma_fast is not None:
        return float(sma_fast.iloc[idx])
    if src == "sma_mid" and sma_mid is not None:
        return float(sma_mid.iloc[idx])
    if src == "sma_slow" and sma_slow is not None:
        return float(sma_slow.iloc[idx])
    return None


def backtest_ema_only(
    df: pd.DataFrame,
    params: EmaOnlyParams,
    initial_capital: float,
    monthly_target_pct: float = 0.01,
) -> Tuple[List[Dict], float, Dict[str, float]]:
    """
    Backtest simples baseado em EMAs, com suporte a stop móvel.

    Modos de sinal suportados:
    - `ema_cross`: cruzamento clássico ema_fast vs ema_slow.
    - `ema_asym`: entradas condicionadas a alinhamento fast/mid/slow, com
      gatilhos de cruzamento e saídas mais sensíveis à perda de alinhamento.

    Retorna lista de trades (BUY/SELL), PnL total e dict de métricas agregadas.
    """
    if params.signal_mode not in {
        "ema_cross",
        "ema_asym",
        "ema_pullback",
        "ema_trend",
        "stack_fast_sma",
        "simple_ref_ema",
    }:
        raise NotImplementedError(
            "signal_mode deve ser 'ema_cross', 'ema_asym', 'ema_pullback', 'ema_trend', 'stack_fast_sma' ou 'simple_ref_ema'."
        )

    if "Date" not in df.columns or "close" not in df.columns:
        raise ValueError("DataFrame deve conter colunas 'Date' e 'close'.")

    df = df.copy().reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])

    close = df["close"].astype(float)

    # EMAs principal (fast/slow) usadas tanto no backtest quanto nos testes.
    ema_fast = compute_ema(close, params.ema_period)
    if params.slow_ema_period is None:
        raise ValueError("slow_ema_period é obrigatório para signal_mode='ema_cross'.")
    ema_slow = compute_ema(close, params.slow_ema_period)

    # EMAs/SMAs adicionais (6 MAs). Se períodos não forem informados, usamos defaults simples.
    ema_mid: Optional[pd.Series] = None
    sma_fast: Optional[pd.Series] = None
    sma_mid: Optional[pd.Series] = None
    sma_slow: Optional[pd.Series] = None

    if params.ema_mid_period:
        ema_mid = compute_ema(close, params.ema_mid_period)
    if params.sma_fast_period:
        sma_fast = compute_sma(close, params.sma_fast_period)
    if params.sma_mid_period:
        sma_mid = compute_sma(close, params.sma_mid_period)
    if params.sma_slow_period:
        sma_slow = compute_sma(close, params.sma_slow_period)

    atr: Optional[pd.Series] = None
    if params.trailing_stop_type in {"atr_trailing", "ma_trailing"} and params.atr_period > 0:
        atr = compute_atr(df, params.atr_period)

    # Índice inicial: garante período suficiente para EMAs/ATR e filtros.
    start = max(
        params.ema_period + 1,
        (params.slow_ema_period or 0) + 1,
        (params.trend_filter_period or 0) + 1,
        (params.atr_period if atr is not None else 0) + 1,
        (params.sma_fast_period or 0) + 1,
        (params.sma_mid_period or 0) + 1,
        2,
    )

    trades: List[Dict] = []
    trade_pnls: List[float] = []

    # Estado financeiro agregado: equity = capital_inicial + pnl_realizado + pnl_não_realizado - taxas
    position = 0.0  # >0 long, <0 short
    entry_price: Optional[float] = None
    entry_fee: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    risk_per_unit: Optional[float] = None
    stop_price: Optional[float] = None
    peak_price: Optional[float] = None  # para percent_trailing

    equity_curve: List[Tuple[pd.Timestamp, float]] = []

    for i in range(start, len(df)):
        price = float(close.iloc[i])
        date = df["Date"].iloc[i]
        high = float(df["high"].iloc[i]) if "high" in df.columns else price
        low = float(df["low"].iloc[i]) if "low" in df.columns else price
        ref_val: Optional[float] = None
        if "ref_ema" in df.columns:
            ref_val_raw = df["ref_ema"].iloc[i]
            if pd.notna(ref_val_raw):
                ref_val = float(ref_val_raw)

        # Cruzamento fast/slow (usa mesma lógica dos testes).
        e_prev, e = float(ema_fast.iloc[i - 1]), float(ema_fast.iloc[i])
        s_prev, s = float(ema_slow.iloc[i - 1]), float(ema_slow.iloc[i])
        cross_up = (e_prev <= s_prev) and (e > s)
        cross_down = (e_prev >= s_prev) and (e < s)

        # Filtro de viés por timeframe de referência (desligado, exceto no modo ema_trend).
        allow_long_entry = True
        allow_short_entry = params.allow_short
        if params.signal_mode == "ema_trend":
            ref_ok_long = price > ref_val if ref_val is not None else True
            ref_ok_short = price < ref_val if ref_val is not None else True
            allow_long_entry = ref_ok_long and price > s
            allow_short_entry = params.allow_short and ref_ok_short and price < s
        elif params.signal_mode == "stack_fast_sma":
            allow_long_entry = True
            allow_short_entry = params.allow_short
        elif params.signal_mode == "simple_ref_ema":
            allow_long_entry = True
            allow_short_entry = params.allow_short

        exit_reason: Optional[str] = None
        exit_price: Optional[float] = None

        # Atualiza trailing stop se houver posição (ignorado no modo ema_trend).
        if position != 0 and params.trailing_stop_type != "none" and params.signal_mode != "ema_trend":
            if params.trailing_stop_type == "atr_trailing" and atr is not None:
                atr_val = float(atr.iloc[i])
                if np.isfinite(atr_val) and atr_val > 0:
                    if position > 0:
                        candidate = price - params.atr_trail_mult * atr_val
                        if stop_price is None:
                            stop_price = price - params.atr_stop_mult * atr_val
                        stop_price = max(stop_price, candidate)
                    else:
                        candidate = price + params.atr_trail_mult * atr_val
                        if stop_price is None:
                            stop_price = price + params.atr_stop_mult * atr_val
                        stop_price = min(stop_price, candidate)

            elif params.trailing_stop_type == "percent_trailing":
                if peak_price is None:
                    peak_price = price
                if position > 0:
                    peak_price = max(peak_price, high)
                    candidate = peak_price * (1.0 - params.percent_trailing_pct)
                    if stop_price is None:
                        stop_price = candidate
                    stop_price = max(stop_price, candidate)
                else:
                    peak_price = min(peak_price, low)
                    candidate = peak_price * (1.0 + params.percent_trailing_pct)
                    if stop_price is None:
                        stop_price = candidate
                    stop_price = min(stop_price, candidate)

            elif params.trailing_stop_type == "ma_trailing":
                if atr is not None:
                    atr_val = float(atr.iloc[i])
                else:
                    atr_val = 0.0
                ma_val = _select_ma_for_trailing(
                    df,
                    idx=i,
                    params=params,
                    ema_fast=ema_fast,
                    ema_mid=ema_mid,
                    ema_slow=ema_slow,
                    sma_fast=sma_fast,
                    sma_mid=sma_mid,
                    sma_slow=sma_slow,
                )
                if ma_val is not None:
                    if position > 0:
                        candidate = ma_val - params.ma_trail_offset_atr_mult * atr_val
                        if stop_price is None:
                            stop_price = candidate
                        stop_price = max(stop_price, candidate)
                    else:
                        candidate = ma_val + params.ma_trail_offset_atr_mult * atr_val
                        if stop_price is None:
                            stop_price = candidate
                        stop_price = min(stop_price, candidate)

            # Lógica de breakeven em R, se soubermos o risco inicial.
            if (
                position != 0
                and risk_per_unit
                and risk_per_unit > 0
                and stop_price is not None
                and params.breakeven_rr > 0
            ):
                if position > 0:
                    r_multiple = (price - entry_price) / risk_per_unit  # type: ignore[operator]
                    if r_multiple >= params.breakeven_rr and stop_price < entry_price:  # type: ignore[operator]
                        stop_price = entry_price
                else:
                    r_multiple = (entry_price - price) / risk_per_unit  # type: ignore[operator]
                    if r_multiple >= params.breakeven_rr and stop_price > entry_price:  # type: ignore[operator]
                        stop_price = entry_price

        # Saída por stop ou cruzamento.
        if position > 0 and stop_price is not None and low <= stop_price:
            exit_reason = "stop_long"
            exit_price = stop_price
        elif position < 0 and stop_price is not None and high >= stop_price:
            exit_reason = "stop_short"
            exit_price = stop_price
        else:
            # Saída por perda de alinhamento/sinal.
            if params.signal_mode == "ema_cross":
                if position > 0 and cross_down:
                    exit_reason = "signal_cross_down"
                    exit_price = price
                elif position < 0 and cross_up:
                    exit_reason = "signal_cross_up"
                    exit_price = price
            elif params.signal_mode == "ema_asym":
                if ema_mid is not None:
                    m_val = float(ema_mid.iloc[i])
                else:
                    m_val = s

                if position > 0:
                    lose_stack = not (e > m_val > s)
                    if cross_down or lose_stack:
                        exit_reason = "loss_of_alignment_long"
                        exit_price = price
                elif position < 0:
                    lose_stack = not (e < m_val < s)
                    if cross_up or lose_stack:
                        exit_reason = "loss_of_alignment_short"
                        exit_price = price
            elif params.signal_mode == "ema_pullback":
                sma_fast_val = float(sma_fast.iloc[i]) if sma_fast is not None else None
                sma_fast_prev = float(sma_fast.iloc[i - 1]) if sma_fast is not None else None
                ema_fast_prev = float(ema_fast.iloc[i - 1])

                if position > 0 and sma_fast_val is not None and sma_fast_prev is not None:
                    cross_down_fast_sma = (ema_fast_prev >= sma_fast_prev) and (e < sma_fast_val)
                    if cross_down_fast_sma:
                        exit_reason = "pullback_exit_long"
                        exit_price = price
                elif position < 0 and sma_fast_val is not None and sma_fast_prev is not None:
                    cross_up_fast_sma = (ema_fast_prev <= sma_fast_prev) and (e > sma_fast_val)
                    if cross_up_fast_sma:
                        exit_reason = "pullback_exit_short"
                        exit_price = price
            elif params.signal_mode == "ema_trend":
                long_cond = (price > ref_val if ref_val is not None else True) and price > s
                short_cond = (price < ref_val if ref_val is not None else True) and price < s
                if position > 0 and not long_cond:
                    exit_reason = "trend_exit_long"
                    exit_price = price
                elif position < 0 and not short_cond:
                    exit_reason = "trend_exit_short"
                    exit_price = price
            elif params.signal_mode == "simple_ref_ema":
                if ema_mid is not None:
                    m_val = float(ema_mid.iloc[i])
                else:
                    m_val = s
                long_cond = e > m_val
                short_cond = e < m_val
                if position > 0 and not long_cond:
                    exit_reason = "simple_exit_long"
                    exit_price = price
                elif position < 0 and not short_cond:
                    exit_reason = "simple_exit_short"
                    exit_price = price
            else:  # stack_fast_sma
                if sma_fast is None or sma_mid is None:
                    raise ValueError("sma_fast_period e sma_mid_period são obrigatórios para signal_mode='stack_fast_sma'.")
                sf = float(sma_fast.iloc[i])
                sm = float(sma_mid.iloc[i])
                long_cond = (e > sf) and (sf > sm)
                short_cond = (e < sf) and (sf < sm)

                if position > 0:
                    # Só saímos (e flipamos) quando aparece o empilhamento oposto.
                    if params.allow_short and short_cond:
                        exit_reason = "stack_flip_to_short"
                        exit_price = price
                    elif not params.allow_short and not long_cond:
                        # Sem short permitido, saímos quando o empilhamento long quebra.
                        exit_reason = "stack_exit_long"
                        exit_price = price
                elif position < 0:
                    if long_cond:
                        exit_reason = "stack_flip_to_long"
                        exit_price = price

        if position != 0 and exit_price is not None:
            # Realiza a posição.
            pos_before = position
            notional = exit_price * abs(pos_before)
            exit_fee = notional * params.fee_rate
            total_fees += exit_fee

            # Guard rail: em teoria entry_price nunca deveria ser None aqui,
            # mas se for, consideramos PnL zero para preservar consistência.
            if entry_price is None:
                pnl = 0.0
                exit_action = "SELL" if pos_before > 0 else "BUY"
            else:
                if pos_before > 0:
                    pnl = (exit_price - entry_price) * abs(pos_before)  # type: ignore[operator]
                    exit_action = "SELL"
                else:
                    pnl = (entry_price - exit_price) * abs(pos_before)  # type: ignore[operator]
                    exit_action = "BUY"

            realized_pnl += float(pnl)
            trade_pnls.append(float(pnl))
            trades.append(
                {
                    "date": date,
                    "action": exit_action,
                    "price": exit_price,
                    "size": abs(params.lot_size),
                    "fee": entry_fee + exit_fee,
                    "reason": exit_reason,
                    "pnl": float(pnl),
                }
            )
            position = 0.0
            entry_price = None
            entry_fee = 0.0
            stop_price = None
            risk_per_unit = None
            peak_price = None

        # Entrada de posição (apenas se estamos flat).
        if position == 0:
            # Sinais brutos dependentes do modo.
            open_long_signal = False
            open_short_signal = False

            if params.signal_mode == "ema_cross":
                open_long_signal = cross_up
                open_short_signal = cross_down
            elif params.signal_mode == "ema_asym":
                stack_up = True
                stack_down = True
                if ema_mid is not None:
                    m_val = float(ema_mid.iloc[i])
                    stack_up = (e > m_val > s)
                    stack_down = (e < m_val < s)
                else:
                    stack_up = e > s
                    stack_down = e < s

                open_long_signal = cross_up and stack_up
                open_short_signal = cross_down and stack_down
            elif params.signal_mode == "ema_pullback":
                sma_fast_val = float(sma_fast.iloc[i]) if sma_fast is not None else None
                sma_fast_prev = float(sma_fast.iloc[i - 1]) if sma_fast is not None else None
                ema_fast_prev = float(ema_fast.iloc[i - 1])

                if sma_fast_val is not None and sma_fast_prev is not None:
                    open_long_signal = (ema_fast_prev <= sma_fast_prev) and (e > sma_fast_val)
                    open_short_signal = (ema_fast_prev >= sma_fast_prev) and (e < sma_fast_val)
            elif params.signal_mode == "ema_trend":
                open_long_signal = allow_long_entry
                open_short_signal = allow_short_entry
            elif params.signal_mode == "simple_ref_ema":
                if ema_mid is not None:
                    m_val = float(ema_mid.iloc[i])
                else:
                    m_val = s
                open_long_signal = e > m_val
                open_short_signal = e < m_val
            else:  # stack_fast_sma
                if sma_fast is None or sma_mid is None:
                    raise ValueError("sma_fast_period e sma_mid_period são obrigatórios para signal_mode='stack_fast_sma'.")
                sf = float(sma_fast.iloc[i])
                sm = float(sma_mid.iloc[i])
                open_long_signal = (e > sf) and (sf > sm)
                open_short_signal = (e < sf) and (sf < sm)

            open_long = open_long_signal and allow_long_entry
            open_short = params.allow_short and open_short_signal and allow_short_entry

            if open_long or open_short:
                side = 1.0 if open_long else -1.0
                entry_price = price
                notional = entry_price * params.lot_size
                entry_fee = notional * params.fee_rate
                total_fees += entry_fee

                position = side * params.lot_size

                # Inicializa stop móvel (se configurado).
                stop_price = None
                risk_per_unit = None
                peak_price = price

                if params.trailing_stop_type == "atr_trailing" and atr is not None:
                    atr_val = float(atr.iloc[i])
                    if np.isfinite(atr_val) and atr_val > 0:
                        if side > 0:
                            stop_price = entry_price - params.atr_stop_mult * atr_val
                            risk_per_unit = entry_price - stop_price
                        else:
                            stop_price = entry_price + params.atr_stop_mult * atr_val
                            risk_per_unit = stop_price - entry_price
                elif params.trailing_stop_type == "percent_trailing":
                    if side > 0:
                        stop_price = entry_price * (1.0 - params.percent_trailing_pct)
                        risk_per_unit = entry_price - stop_price
                    else:
                        stop_price = entry_price * (1.0 + params.percent_trailing_pct)
                        risk_per_unit = stop_price - entry_price
                elif params.trailing_stop_type == "ma_trailing":
                    ma_val = _select_ma_for_trailing(
                        df,
                        idx=i,
                        params=params,
                        ema_fast=ema_fast,
                        ema_mid=ema_mid,
                        ema_slow=ema_slow,
                        sma_fast=sma_fast,
                        sma_mid=sma_mid,
                        sma_slow=sma_slow,
                    )
                    atr_val = float(atr.iloc[i]) if atr is not None else 0.0
                    if ma_val is not None:
                        if side > 0:
                            stop_price = ma_val - params.ma_trail_offset_atr_mult * atr_val
                            risk_per_unit = entry_price - stop_price
                        else:
                            stop_price = ma_val + params.ma_trail_offset_atr_mult * atr_val
                            risk_per_unit = stop_price - entry_price

                if params.signal_mode == "ema_cross":
                    entry_reason = "signal_cross_up" if side > 0 else "signal_cross_down"
                elif params.signal_mode == "ema_asym":
                    entry_reason = "signal_asym_long" if side > 0 else "signal_asym_short"
                elif params.signal_mode == "ema_pullback":
                    entry_reason = "signal_pullback_long" if side > 0 else "signal_pullback_short"
                elif params.signal_mode == "ema_trend":
                    entry_reason = "signal_trend_long" if side > 0 else "signal_trend_short"
                elif params.signal_mode == "simple_ref_ema":
                    entry_reason = "simple_long" if side > 0 else "simple_short"
                else:
                    entry_reason = "signal_stack_long" if side > 0 else "signal_stack_short"

                trades.append(
                    {
                        "date": date,
                        "action": "BUY" if side > 0 else "SELL",
                        "price": entry_price,
                        "size": abs(params.lot_size),
                        "fee": entry_fee,
                        "reason": entry_reason,
                    }
                )

        # Atualiza equity (capital inicial + PnL realizado + PnL não realizado - taxas).
        unrealized = 0.0
        if position > 0 and entry_price is not None:
            unrealized = (price - entry_price) * abs(position)
        elif position < 0 and entry_price is not None:
            unrealized = (entry_price - price) * abs(position)

        equity = initial_capital + realized_pnl + unrealized - total_fees
        equity_curve.append((date, float(equity)))

    # Se ainda houver posição aberta no final, fecha no último preço.
    if position != 0 and entry_price is not None:
        price = float(close.iloc[-1])
        date = df["Date"].iloc[-1]
        notional = price * abs(position)
        exit_fee = notional * params.fee_rate
        total_fees += exit_fee
        if position > 0:
            pnl = (price - entry_price) * abs(position)  # type: ignore[operator]
        else:
            pnl = (entry_price - price) * abs(position)  # type: ignore[operator]

        realized_pnl += float(pnl)
        trade_pnls.append(float(pnl))
        trades.append(
            {
                "date": date,
                "action": "SELL" if position > 0 else "BUY",
                "price": price,
                "size": abs(params.lot_size),
                "fee": entry_fee + exit_fee,
                "reason": "forced_liquidation_at_end",
                "pnl": float(pnl),
            }
        )
        position = 0.0
        entry_price = None
        entry_fee = 0.0
        equity = initial_capital + realized_pnl - total_fees
        equity_curve.append((date, float(equity)))

    # Métricas agregadas.
    equity_series = pd.Series(
        data=[e for _, e in equity_curve],
        index=[d for d, _ in equity_curve],
        dtype=float,
    )
    if equity_series.empty:
        total_pnl = 0.0
        stats = {
            "num_trades": 0,
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "avg_monthly_return_pct": 0.0,
            "monthly_target_pct": monthly_target_pct,
            "monthly_target_hit_ratio": 0.0,
            "num_months": 0,
        }
        return trades, total_pnl, stats

    total_pnl = float(equity_series.iloc[-1] - initial_capital)
    total_return_pct = total_pnl / float(initial_capital)

    # Max drawdown.
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    max_dd_pct = float(abs(dd.min())) if not dd.empty else 0.0

    # Sharpe simples (por candle, não anualizado).
    ret = equity_series.pct_change().dropna()
    if not ret.empty and ret.std(ddof=1) > 0:
        sharpe = float(ret.mean() / ret.std(ddof=1))
    else:
        sharpe = 0.0

    calmar = float(total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

    # Estatísticas de trades.
    num_trades = len(trade_pnls)
    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = float(wins / num_trades) if num_trades > 0 else 0.0

    # Retorno mensal vs meta + breakdown por mês.
    monthly_equity = equity_series.resample("M").last().dropna()
    monthly_returns = monthly_equity.pct_change().dropna()
    monthly_breakdown = {}

    if not monthly_equity.empty:
        prev_eq = float(initial_capital)
        for ts, eq in monthly_equity.items():
            end_eq = float(eq)
            start_eq = prev_eq
            pnl_m = end_eq - start_eq
            ret_m = pnl_m / start_eq if start_eq != 0 else 0.0
            key = ts.strftime("%Y-%m")
            monthly_breakdown[key] = {
                "start_equity": start_eq,
                "end_equity": end_eq,
                "pnl": pnl_m,
                "return_pct": ret_m,
            }
            prev_eq = end_eq

    if not monthly_returns.empty:
        avg_monthly_return = float(monthly_returns.mean())
        num_months = int(len(monthly_returns))
        hit_ratio = float((monthly_returns >= monthly_target_pct).mean())
    else:
        avg_monthly_return = 0.0
        num_months = 0
        hit_ratio = 0.0

    stats = {
        "num_trades": num_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct,
        "sharpe": sharpe,
        "calmar": calmar,
        "avg_monthly_return_pct": avg_monthly_return,
        "monthly_target_pct": monthly_target_pct,
        "monthly_target_hit_ratio": hit_ratio,
        "num_months": num_months,
        "monthly_breakdown": monthly_breakdown,
    }

    return trades, total_pnl, stats
