import time
from typing import Optional

import pandas as pd
from binance.client import Client
import urllib3

from .cache.klines_cache import cached_klines, to_timestamp_ms

# É uma boa prática não colocar chaves de API diretamente no código.
# Para dados públicos como histórico de preços, elas não são necessárias.
# A opção verify=False pode gerar avisos. Vamos desabilitá-los.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

client = Client("", "", requests_params={"timeout": 20, "verify": False})


def _download_klines(symbol: str, interval: str, start_ms: int, end_ms: int):
    if start_ms > end_ms:
        return []
    raw = client.get_historical_klines(symbol, interval, start_ms, end_ms)
    rows = []
    for k in raw:
        rows.append(
            (
                int(k[0]),
                float(k[1]),
                float(k[2]),
                float(k[3]),
                float(k[4]),
                float(k[5]),
            )
        )
    return rows


def _cached_dataframe(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    df = cached_klines(
        symbol,
        interval,
        start_ms,
        end_ms,
        lambda s, e: _download_klines(symbol, interval, s, e),
    )
    return df


def _format_output(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=False)
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
    return df[["Date", "open", "high", "low", "close", "volume"]]


def _direct_download(symbol: str, interval: str, start_str, end_str=None) -> pd.DataFrame:
    raw = client.get_historical_klines(symbol, interval, start_str, end_str)
    if not raw:
        return pd.DataFrame()
    columns = [
        "Open time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)
    df["Date"] = pd.to_datetime(df["Open time"], unit="ms")
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    return df[["Date", "open", "high", "low", "close", "volume"]]


def get_historical_klines(symbol, interval, start_str, end_str=None):
    """Busca dados históricos de klines com cache local."""
    start_ms: Optional[int] = to_timestamp_ms(start_str)
    end_ms: Optional[int] = to_timestamp_ms(end_str) if end_str else None

    if start_ms is None:
        return _format_output(_direct_download(symbol, interval, start_str, end_str))

    if end_ms is None:
        end_ms = int(time.time() * 1000)

    df = _cached_dataframe(symbol, interval, start_ms, end_ms)
    return _format_output(df)
