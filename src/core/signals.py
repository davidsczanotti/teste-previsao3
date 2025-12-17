import pandas as pd
import numpy as np
from typing import Dict

def apply_signals(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Roteador de estratégias: Aplica a lógica de sinal baseada no 'signal_mode'
    definido na configuração.
    """
    df = df.copy()
    strategy = config['strategy']
    mode = strategy.get('signal_mode', 'ema_cross')

    # Inicializar sinal neutro
    df['signal'] = 0

    # 1. Aplicar Viés de Referência (Timeframe Superior) - Comum a todos
    if strategy.get('ref_filter_enabled'):
        ref_buffer = strategy['ref_buffer_pct']
        df['ref_bias'] = np.where(df['close'] > df['ref_ema'] * (1 + ref_buffer), 1,
                                  np.where(df['close'] < df['ref_ema'] * (1 - ref_buffer), -1, 0))
    
    # 2. Roteamento de Lógica Específica
    if mode == 'custom_cci_ma':
        df = _signal_custom_cci_ma(df, strategy)
    elif mode == 'trend_surfer_v4':
        df = _signal_trend_surfer_v4(df, strategy)
    elif mode == 'ema_strategy_v5_2':
        df = _signal_ema_strategy_v5_2(df, strategy)
    elif mode == 'dynamic_volatility_v6':
        df = _signal_dynamic_volatility_v6(df, strategy)
    elif mode == 'supertrend_ai':
        df = _signal_supertrend_ai(df, strategy)
    elif mode == 'ema_cross':
        df = _signal_ema_cross(df, strategy)
    else:
        pass

    # 3. Filtragem Final
    allow_short = bool(strategy.get('allow_short', False))
    
    if strategy.get('ref_filter_enabled') and 'ref_bias' in df.columns:
        if allow_short:
            df['signal'] = np.where(
                (df['signal'] == 1) & (df['ref_bias'] == 1), 1,
                np.where((df['signal'] == -1) & (df['ref_bias'] == -1), -1, 0)
            )
        else:
            df['signal'] = np.where((df['signal'] == 1) & (df['ref_bias'] == 1), 1, 0)
    
    if not allow_short:
        df['signal'] = np.where(df['signal'] == -1, 0, df['signal'])

    return df

# ... (Previous functions custom_cci_ma, ema_cross, trend_surfer_v4, ema_strategy_v5_2, dynamic_volatility_v6 remain same)

def _signal_custom_cci_ma(df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
    mc = df['custom_ma_fast']
    ml = df['custom_ma_slow']
    cci = df['custom_cci']
    atr = df.get('custom_atr', df['close'] * 0.01)
    ema_200 = df.get('ema_200', df['close'])
    level = strategy['custom_cci_level']
    dist_mult = strategy.get('custom_dist_atr_mult', 0.5)
    min_dist = atr * dist_mult
    diff = (mc - ml).abs()
    cond_long = (mc > ml) & (cci > level) & (diff > min_dist) & (df['close'] > ema_200)
    cond_short = (mc < ml) & (cci < -level) & (diff > min_dist) & (df['close'] < ema_200)
    df.loc[cond_long, 'signal'] = 1
    df.loc[cond_short, 'signal'] = -1
    return df

def _signal_ema_cross(df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
    ema_fast = df['ema_fast']
    ema_slow = df['ema_slow']
    df['ema_cross'] = np.where(ema_fast > ema_slow, 1, -1)
    df['ema_cross_prev'] = df['ema_cross'].shift(1)
    df['signal'] = np.where((df['ema_cross'] == 1) & (df['ema_cross_prev'] == -1), 1,
                            np.where((df['ema_cross'] == -1) & (df['ema_cross_prev'] == 1), -1, 0))
    if strategy.get('max_long_entry_dist_fast_pct'):
        max_dist = strategy.get('max_long_entry_dist_fast_pct')
        dist_fast = (df['close'] - ema_fast).abs() / ema_fast
        df.loc[(df['signal'] == 1) & (dist_fast > max_dist), 'signal'] = 0
    return df

def _signal_trend_surfer_v4(df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
    fast = df['ts_fast_ma']
    slow = df['ts_slow_ma']
    macro = df['ts_ema_macro']
    cci = df['ts_cci']
    cci_min = strategy.get('ts_cci_min', 0)
    use_date_filter = bool(strategy.get("ts_use_date_filter", False))
    start_year = int(strategy.get("ts_start_year", 2016))
    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    trend_ok = df['close'] > macro
    mom_ok = cci > cci_min
    if use_date_filter and 'Date' in df.columns:
        years = pd.to_datetime(df['Date']).dt.year
        date_ok = years >= start_year
    else:
        date_ok = True
    cond_long = cross_up & trend_ok & mom_ok & date_ok
    df.loc[cond_long, 'signal'] = 1
    if strategy.get('allow_short', False):
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        trend_bear = df['close'] < macro
        mom_bear = cci < -cci_min
        cond_short = cross_down & trend_bear & mom_bear
        df.loc[cond_short, 'signal'] = -1
    return df

def _signal_ema_strategy_v5_2(df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
    fast = df['ts_fast_ma']
    slow = df['ts_slow_ma']
    macro = df['ts_ema_macro']
    start_year = int(strategy.get("ts_start_year", 2010))
    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    trend_ok = df['close'] > macro
    if 'Date' in df.columns:
        years = pd.to_datetime(df['Date']).dt.year
        date_ok = years >= start_year
    else:
        date_ok = True
    cond_long = cross_up & trend_ok & date_ok
    df.loc[cond_long, 'signal'] = 1
    exit_trigger = strategy.get("exit_trigger", "cross_ma") 
    if exit_trigger == "close_under_slow":
        cond_exit = (df['close'] < slow) & (df['close'].shift(1) >= slow.shift(1))
    else:
        cond_exit = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    df['exit_signal'] = np.where(cond_exit, 1, 0)
    return df

def _signal_dynamic_volatility_v6(df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
    fast = df['ts_fast_ma']
    slow = df['ts_slow_ma']
    macro = df['ts_ema_macro']
    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    trend_ok = df['close'] > macro
    cond_long = cross_up & trend_ok
    df.loc[cond_long, 'signal'] = 1
    cond_exit = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    df['exit_signal'] = np.where(cond_exit, 1, 0)
    return df

def _signal_supertrend_ai(df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
    """
    Estratégia: SuperTrend AI (LuxAlgo)
    Entrada: Trend muda para Bull (1).
    Saída: Trend muda para Bear (0).
    """
    trend = df['supertrend_ai_trend']
    trend_prev = trend.shift(1).fillna(0)
    
    cond_long = (trend == 1) & (trend_prev == 0)
    df.loc[cond_long, 'signal'] = 1
    
    cond_exit = (trend == 0) & (trend_prev == 1)
    df['exit_signal'] = np.where(cond_exit, 1, 0)
    
    return df