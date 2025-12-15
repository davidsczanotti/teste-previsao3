import pandas as pd
import numpy as np
from typing import Dict

def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Calcula ATR (Average True Range)."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_cci(df: pd.DataFrame, period: int) -> pd.Series:
    """Calcula CCI (Commodity Channel Index)."""
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    ma = tp.rolling(period).mean()

    def _mean_dev(x):
        return np.mean(np.abs(x - np.mean(x)))

    # Otimização: usar rolling().apply pode ser lento, mas para backtest é aceitável.
    # Se precisar de performance, vetorizar o mean_dev.
    mean_dev = tp.rolling(period).apply(_mean_dev, raw=True)
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
        df['sma_fast'] = df['close'].rolling(strategy['sma_fast_period']).mean()
    if 'sma_mid_period' in strategy:
        df['sma_mid'] = df['close'].rolling(strategy['sma_mid_period']).mean()
    if 'sma_slow_period' in strategy:
        df['sma_slow'] = df['close'].rolling(strategy['sma_slow_period']).mean()

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
        df['custom_ma_fast'] = df['close'].rolling(strategy['custom_ma_fast']).mean()
        df['custom_ma_slow'] = df['close'].rolling(strategy['custom_ma_slow']).mean()
        
        # CCI específico
        df['custom_cci'] = calculate_cci(df, strategy['custom_cci_period'])
        
        # ATR específico
        custom_atr_period = strategy.get('custom_atr_period', 10)
        df['custom_atr'] = calculate_atr(df, custom_atr_period)
        
        # Filtro Macro (EMA 200 hardcoded na lógica original, agora explícito)
        df['ema_200'] = df['close'].ewm(span=200).mean()

    return df
