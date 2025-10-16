"""
Funções utilitárias para carregamento de dados
"""

import pandas as pd
from typing import Optional
from ..binance_client import get_historical_klines, get_cached_klines


def load_data(symbol: str, timeframe: str, days: int = 365, use_cache_only: bool = False) -> pd.DataFrame:
    """
    Carrega dados históricos para um símbolo e timeframe específicos
    
    Args:
        symbol: Símbolo do ativo (ex: BTCUSDT)
        timeframe: Timeframe das velas (ex: 15m, 1h)
        days: Número de dias de dados históricos
        
    Returns:
        DataFrame com os dados históricos
    """
    from datetime import datetime, timedelta, UTC
    
    start_dt = datetime.now(UTC) - timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    if use_cache_only:
        df = get_cached_klines(symbol, timeframe, start_str)
    else:
        df = get_historical_klines(symbol, timeframe, start_str)
    
    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {symbol} @ {timeframe}")
    
    return df.sort_values("Date").reset_index(drop=True)


def load_data_range(symbol: str, timeframe: str, start_date: str, end_date: str, use_cache_only: bool = False) -> pd.DataFrame:
    """
    Carrega dados históricos para um período específico
    
    Args:
        symbol: Símbolo do ativo
        timeframe: Timeframe das velas
        start_date: Data inicial (formato: YYYY-MM-DD HH:MM:SS)
        end_date: Data final (formato: YYYY-MM-DD HH:MM:SS)
        
    Returns:
        DataFrame com os dados históricos
    """
    if use_cache_only or symbol.upper().startswith("FAKE"):
        df = get_cached_klines(symbol, timeframe, start_date, end_date)
    else:
        df = get_historical_klines(symbol, timeframe, start_date, end_date)
    
    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {symbol} @ {timeframe} no período {start_date} a {end_date}")
    
    return df.sort_values("Date").reset_index(drop=True)
