"""
Funções utilitárias para carregamento de dados
"""

import pandas as pd
from typing import Optional
import yfinance as yf
from datetime import datetime, timedelta, UTC
from ..binance_client import get_historical_klines, get_cached_klines

def load_data(symbol: str, timeframe: str, days: int = 365, use_cache_only: bool = False) -> pd.DataFrame:
    """
    Carrega dados de mercado. Suporta B3 (.SA) via Yahoo Finance e Cripto via Binance.
    """
    # Lógica B3 (Yahoo Finance)
    if symbol.endswith(".SA"):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        
        if df.empty:
            raise ValueError(f"Nenhum dado encontrado no Yahoo Finance para {symbol}")

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except: pass
        
        df = df.reset_index()
        cols_map = {"Date": "Date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        df.rename(columns=cols_map, inplace=True)
        df.columns = [c.lower() if c in ['Open','High','Low','Close','Volume'] else c for c in df.columns]
        if 'date' in df.columns: df.rename(columns={'date': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values("Date").reset_index(drop=True)

    # Lógica Cripto (Binance)
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
    """
    # Lógica B3 (Yahoo Finance)
    if symbol.endswith(".SA"):
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if df.empty:
            raise ValueError(f"Nenhum dado encontrado no Yahoo Finance para {symbol}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        cols_map = {"Date": "Date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        df.rename(columns=cols_map, inplace=True)
        df.columns = [c.lower() if c in ['Open','High','Low','Close','Volume'] else c for c in df.columns]
        if 'date' in df.columns: df.rename(columns={'date': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values("Date").reset_index(drop=True)

    # Lógica Cripto (Binance)
    if use_cache_only or symbol.upper().startswith("FAKE"):
        df = get_cached_klines(symbol, timeframe, start_date, end_date)
    else:
        df = get_historical_klines(symbol, timeframe, start_date, end_date)
    
    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {symbol} @ {timeframe} no período {start_date} a {end_date}")
    
    return df.sort_values("Date").reset_index(drop=True)