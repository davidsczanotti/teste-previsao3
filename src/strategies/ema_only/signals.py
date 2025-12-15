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
    # Isso define um 'filtro' global antes dos sinais específicos
    if strategy.get('ref_filter_enabled'):
        ref_buffer = strategy['ref_buffer_pct']
        # 1 = Bullish Bias, -1 = Bearish Bias, 0 = Neutro
        df['ref_bias'] = np.where(df['close'] > df['ref_ema'] * (1 + ref_buffer), 1,
                                  np.where(df['close'] < df['ref_ema'] * (1 - ref_buffer), -1, 0))
    
    # 2. Roteamento de Lógica Específica
    if mode == 'custom_cci_ma':
        df = _signal_custom_cci_ma(df, strategy)
    elif mode == 'ema_cross':
        df = _signal_ema_cross(df, strategy)
    else:
        # Fallback ou outros modos
        pass

    # 3. Filtragem Final (Direção permitida vs Viés)
    allow_short = bool(strategy.get('allow_short', False))
    
    if strategy.get('ref_filter_enabled') and 'ref_bias' in df.columns:
        if allow_short:
            # Só compra se bias for bull, só vende se bias for bear
            df['signal'] = np.where(
                (df['signal'] == 1) & (df['ref_bias'] == 1), 1,
                np.where((df['signal'] == -1) & (df['ref_bias'] == -1), -1, 0)
            )
        else:
            # Long-only: só compra em bull bias
            df['signal'] = np.where((df['signal'] == 1) & (df['ref_bias'] == 1), 1, 0)
    
    if not allow_short:
        df['signal'] = np.where(df['signal'] == -1, 0, df['signal'])

    return df

def _signal_custom_cci_ma(df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
    """
    Lógica 'Custom CCI + MA'.
    Combina momentum (CCI), tendência (MAs) e volatilidade (ATR).
    """
    # Recupera colunas pré-calculadas em indicators.py
    mc = df['custom_ma_fast']
    ml = df['custom_ma_slow']
    cci = df['custom_cci']
    atr = df.get('custom_atr', df['close'] * 0.01)
    ema_200 = df.get('ema_200', df['close'])
    
    level = strategy['custom_cci_level']
    dist_mult = strategy.get('custom_dist_atr_mult', 0.5)
    
    # Lógica de Separação Mínima
    min_dist = atr * dist_mult
    diff = (mc - ml).abs()
    
    # Condições Booleanas
    # Long: Rápida > Lenta AND CCI > Nível AND Separação OK AND Acima da Média Macro
    cond_long = (mc > ml) & (cci > level) & (diff > min_dist) & (df['close'] > ema_200)
    
    # Short: Rápida < Lenta AND CCI < -Nível AND Separação OK AND Abaixo da Média Macro
    cond_short = (mc < ml) & (cci < -level) & (diff > min_dist) & (df['close'] < ema_200)
    
    df.loc[cond_long, 'signal'] = 1
    df.loc[cond_short, 'signal'] = -1
    
    return df

def _signal_ema_cross(df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
    """
    Lógica Clássica 'EMA Cross'.
    Cruzamento simples de médias exponenciais.
    """
    # Usa as médias 'padrão' definidas no config
    ema_fast = df['ema_fast']
    ema_slow = df['ema_slow']
    
    df['ema_cross'] = np.where(ema_fast > ema_slow, 1, -1)
    df['ema_cross_prev'] = df['ema_cross'].shift(1)
    
    # Crossover (1) e Crossunder (-1)
    df['signal'] = np.where((df['ema_cross'] == 1) & (df['ema_cross_prev'] == -1), 1,
                            np.where((df['ema_cross'] == -1) & (df['ema_cross_prev'] == 1), -1, 0))

    # Filtros Opcionais do modo clássico
    if strategy.get('max_long_entry_dist_fast_pct'):
        max_dist = strategy.get('max_long_entry_dist_fast_pct')
        dist_fast = (df['close'] - ema_fast).abs() / ema_fast
        df.loc[(df['signal'] == 1) & (dist_fast > max_dist), 'signal'] = 0

    return df
