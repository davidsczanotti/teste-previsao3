import numpy as np
import pandas as pd
import vectorbt as vbt


def run_backtest(
    price_data: pd.DataFrame,
    rsi_slow_period: int,
    rsi_medium_period: int,
    rsi_fast_period: int,
    rsi_pullback_level_long: int,
    rsi_pullback_level_short: int,
    adx_period: int,
    adx_threshold: float,
    atr_period: int,
    min_atr_pct: float,
    rr_ratio: float,
    stop_loss_multiplier: float,
    fee: float,
    initial_capital: float,
    size: float,
):
    """
    Executa um backtest vetorizado da estratégia Triple RSI.
    """
    price = price_data["close"]

    # 1. Calcular Indicadores
    rsi_slow = vbt.RSI.run(price, window=rsi_slow_period).rsi
    rsi_medium = vbt.RSI.run(price, window=rsi_medium_period).rsi
    rsi_fast = vbt.RSI.run(price, window=rsi_fast_period).rsi
    adx = vbt.ADX.run(price_data["high"], price_data["low"], price, window=adx_period).adx
    atr = vbt.ATR.run(price_data["high"], price_data["low"], price, window=atr_period).real

    # 2. Definir Condições de Regime e Filtros
    is_uptrend_regime = rsi_slow > 50
    is_downtrend_regime = rsi_slow < 50
    is_trending = adx > adx_threshold
    is_volatile = (atr / price) * 100 > min_atr_pct

    # 3. Definir Sinais de Entrada
    # Sinal de pullback de compra: em regime de alta, RSI médio cai abaixo do nível
    pullback_long = rsi_medium < rsi_pullback_level_long
    # Gatilho de compra: RSI rápido cruza para cima de 50
    trigger_long = rsi_fast.vbt.crossed_above(50)
    # Entrada de compra: todas as condições devem ser verdadeiras
    entries_long = (
        is_uptrend_regime & is_trending & is_volatile & pullback_long & trigger_long
    )

    # Sinal de pullback de venda: em regime de baixa, RSI médio sobe acima do nível
    pullback_short = rsi_medium > rsi_pullback_level_short
    # Gatilho de venda: RSI rápido cruza para baixo de 50
    trigger_short = rsi_fast.vbt.crossed_below(50)
    # Entrada de venda: todas as condições devem ser verdadeiras
    entries_short = (
        is_downtrend_regime & is_trending & is_volatile & pullback_short & trigger_short
    )

    # 4. Executar o Backtest com vectorbt
    pf = vbt.Portfolio.from_signals(
        close=price,
        entries=entries_long,
        exits=entries_short,  # Usando exits opostos para simplificar
        sl_stop=stop_loss_multiplier * atr / price,
        tp_stop=rr_ratio * (stop_loss_multiplier * atr / price),
        size=size,
        init_cash=initial_capital,
        fees=fee,
        freq="1T",  # Frequência de 1 minuto
    )
    return pf
