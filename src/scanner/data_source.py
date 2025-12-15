import pandas as pd
import yfinance as yf
from src.binance_client import get_historical_klines

def download_b3_data(ticker: str, days: int = 3000) -> pd.DataFrame:
    """Baixa dados do Yahoo Finance e padroniza."""
    try:
        from datetime import datetime, timedelta
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Download
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        
        if df.empty:
            return pd.DataFrame()

        # Ajuste de MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if ticker in df.columns.get_level_values(1):
                        df.columns = df.columns.get_level_values(0)
                else:
                        df.columns = df.columns.get_level_values(0)
            except:
                df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        cols_map = {
            "Date": "Date", "Open": "open", "High": "high", 
            "Low": "low", "Close": "close", "Volume": "volume"
        }
        df = df.rename(columns=cols_map)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.dropna()
    except Exception as e:
        print(f"Erro YFinance {ticker}: {e}")
        return pd.DataFrame()

def download_crypto_data(symbol: str, timeframe: str = "4h", days: int = 1500) -> pd.DataFrame:
    """Baixa dados da Binance e padroniza."""
    try:
        from datetime import datetime, timedelta, UTC
        start_dt = datetime.now(UTC) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Usa o client existente do projeto (já tem cache!)
        df = get_historical_klines(symbol, timeframe, start_str)
        
        if df.empty:
            return pd.DataFrame()
            
        return df.sort_values("Date").reset_index(drop=True)
    except Exception as e:
        print(f"Erro Binance {symbol}: {e}")
        return pd.DataFrame()
