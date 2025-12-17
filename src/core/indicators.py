import pandas as pd
import numpy as np
from typing import Dict

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calcula SMA com o mesmo comportamento do Pine (NaN até completar a janela)."""
    return series.rolling(window=period, min_periods=period).mean()


def calculate_ema_tv(series: pd.Series, period: int) -> pd.Series:
    """Calcula EMA compatível com TradingView (ta.ema).

    Implementa a recursão com alpha=2/(period+1) e seed via SMA(period),
    retornando NaN até existir janela completa (comportamento de `ta.ema`).
    """
    period = int(period)
    if period <= 0:
        raise ValueError("EMA period must be > 0")

    values = pd.Series(series, copy=False).astype(float)
    out = pd.Series(index=values.index, dtype=float)

    if len(values) < period:
        return out  # tudo NaN

    alpha = 2.0 / (period + 1.0)

    seed = float(values.iloc[:period].mean())
    out.iloc[: period - 1] = np.nan
    out.iloc[period - 1] = seed

    for i in range(period, len(values)):
        prev = out.iloc[i - 1]
        cur = values.iloc[i]
        out.iloc[i] = (alpha * cur) + ((1.0 - alpha) * prev)

    return out


def calculate_cci_from_series(source: pd.Series, period: int) -> pd.Series:
    """Calcula CCI compatível com TradingView (ta.cci(source, period))."""
    period = int(period)
    if period <= 0:
        raise ValueError("CCI period must be > 0")

    src = pd.Series(source, copy=False).astype(float)
    ma = calculate_sma(src, period)

    def _mean_dev(x):
        m = np.mean(x)
        return np.mean(np.abs(x - m))

    mean_dev = src.rolling(window=period, min_periods=period).apply(_mean_dev, raw=True)
    denom = 0.015 * mean_dev
    return (src - ma) / denom


def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Calcula ATR (Average True Range)."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_cci(df: pd.DataFrame, period: int) -> pd.Series:
    """Calcula CCI (Commodity Channel Index) usando HLC3 (legado do projeto)."""
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    ma = calculate_sma(tp, int(period))

    def _mean_dev(x):
        return np.mean(np.abs(x - np.mean(x)))

    # Otimização: usar rolling().apply pode ser lento, mas para backtest é aceitável.
    # Se precisar de performance, vetorizar o mean_dev.
    mean_dev = tp.rolling(window=int(period), min_periods=int(period)).apply(_mean_dev, raw=True)
    cci = (tp - ma) / (0.015 * mean_dev)
    return cci

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcula RSI (Relative Strength Index)."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcula ADX (Average Directional Index)."""
    # True Range
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Directional Movement
    up_move = df['high'].diff()
    down_move = -df['low'].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smoothing (Wilder's Smoothing)
    # TR, +DM, -DM
    tr_s = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / tr_s)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / tr_s)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

def add_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Adiciona todos os indicadores técnicos necessários ao DataFrame
    baseado na configuração fornecida.
    """
    df = df.copy()
    strategy = config['strategy']

    # --- 1. Médias Móveis Genéricas (para visualização ou lógica base) ---
    if 'sma_fast_period' in strategy:
        df['sma_fast'] = calculate_sma(df['close'], int(strategy['sma_fast_period']))
    if 'sma_mid_period' in strategy:
        df['sma_mid'] = calculate_sma(df['close'], int(strategy['sma_mid_period']))
    if 'sma_slow_period' in strategy:
        df['sma_slow'] = calculate_sma(df['close'], int(strategy['sma_slow_period']))

    if 'ema_fast_period' in strategy:
        df['ema_fast'] = df['close'].ewm(span=strategy['ema_fast_period']).mean()
    if 'ema_mid_period' in strategy:
        df['ema_mid'] = df['close'].ewm(span=strategy['ema_mid_period']).mean()
    if 'ema_slow_period' in strategy:
        df['ema_slow'] = df['close'].ewm(span=strategy['ema_slow_period']).mean()

    # --- 2. ATR Global ---
    if 'atr_period' in strategy:
        df['atr'] = calculate_atr(df, strategy['atr_period'])

    # --- 3. CCI Global ---
    if 'cci_period' in strategy:
        df['cci'] = calculate_cci(df, int(strategy['cci_period']))

    # --- 4. ADX (Novo) ---
    if 'adx_period' in strategy:
        df['adx'] = calculate_adx(df, strategy['adx_period'])

    # --- 5. Indicadores Específicos do Modo Custom (Identity Fix) ---
    if strategy.get('signal_mode') == 'custom_cci_ma':
        # Médias específicas da lógica customizada
        df['custom_ma_fast'] = calculate_sma(df['close'], int(strategy['custom_ma_fast']))
        df['custom_ma_slow'] = calculate_sma(df['close'], int(strategy['custom_ma_slow']))
        
        # CCI específico
        df['custom_cci'] = calculate_cci(df, strategy['custom_cci_period'])
        
        # ATR específico
        custom_atr_period = strategy.get('custom_atr_period', 10)
        df['custom_atr'] = calculate_atr(df, custom_atr_period)
        
        # Filtro Macro (EMA 200 hardcoded na lógica original, agora explícito)
        df['ema_200'] = df['close'].ewm(span=200).mean()

    # --- 6. Indicadores Específicos do Modo Trend Surfer v4 ---
    if strategy.get('signal_mode') == 'trend_surfer_v4':
        fast_p = strategy.get('ts_fast_period', 9)
        slow_p = strategy.get('ts_slow_period', 21)
        macro_p = strategy.get('ts_ema_macro_period', 200)
        cci_p = strategy.get('ts_cci_period', 14)

        df['ts_fast_ma'] = calculate_sma(df['close'], int(fast_p))
        df['ts_slow_ma'] = calculate_sma(df['close'], int(slow_p))
        df['ts_ema_macro'] = calculate_ema_tv(df['close'], int(macro_p))
        df['ts_cci'] = calculate_cci_from_series(df['close'], int(cci_p))

    # --- 7. Indicadores Específicos do Modo EMA Strategy v5.2 ---
    if strategy.get('signal_mode') == 'ema_strategy_v5_2':
        fast_p = strategy.get('ts_fast_period', 9)
        slow_p = strategy.get('ts_slow_period', 21)
        macro_p = strategy.get('ts_ema_macro_period', 200)
        
        # User default: EMA (maType="EMA")
        df['ts_fast_ma'] = calculate_ema_tv(df['close'], int(fast_p))
        df['ts_slow_ma'] = calculate_ema_tv(df['close'], int(slow_p))
        df['ts_ema_macro'] = calculate_ema_tv(df['close'], int(macro_p))

    # --- 8. Indicadores Específicos do Modo Dynamic Volatility v6 ---
    if strategy.get('signal_mode') == 'dynamic_volatility_v6':
        fast_p = strategy.get('ts_fast_period', 9)
        slow_p = strategy.get('ts_slow_period', 21)
        macro_p = strategy.get('ts_ema_macro_period', 200)
        adx_p = strategy.get('adx_period', 14)
        atr_p = strategy.get('atr_period', 14)
        
        df['ts_fast_ma'] = calculate_ema_tv(df['close'], int(fast_p))
        df['ts_slow_ma'] = calculate_ema_tv(df['close'], int(slow_p))
        df['ts_ema_macro'] = calculate_ema_tv(df['close'], int(macro_p))
        df['adx'] = calculate_adx(df, int(adx_p))
        df['atr'] = calculate_atr(df, int(atr_p))

    # --- 9. Indicadores Específicos do Modo SuperTrend AI ---
    if strategy.get('signal_mode') == 'supertrend_ai':
        # Importação local para evitar ciclo
        try:
            from src.strategies.supertrend_ai import calculate_supertrend_ai
            df = calculate_supertrend_ai(df, config)
        except ImportError:
            # Fallback se executado de local diferente
            from strategies.supertrend_ai import calculate_supertrend_ai
            df = calculate_supertrend_ai(df, config)

    return df
