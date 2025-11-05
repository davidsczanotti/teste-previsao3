import time
from typing import Optional

import pandas as pd
import os
from binance.client import Client
import urllib3

from .cache.klines_cache import cached_klines, to_timestamp_ms
from .cache.klines_cache import load_range as _cache_load_range
from .cache.klines_cache import latest_open_time as _cache_latest_open_time

# É uma boa prática não colocar chaves de API diretamente no código.
# Para dados públicos como histórico de preços, elas não são necessárias.
# A opção verify=False pode gerar avisos. Vamos desabilitá-los.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

"""
Tune timeouts to reduce read timeouts and allow per-call retries from wrappers.
Using a (connect, read) tuple keeps the connection snappy while allowing
slightly longer reads quando a Binance está lenta.
"""

_client_cache: Client | None = None


def _is_offline() -> bool:
    return os.environ.get("BINANCE_OFFLINE", "0") == "1"


def _get_client() -> Client | None:
    global _client_cache
    if _is_offline():
        return None
    if _client_cache is None:
        _client_cache = Client("", "", requests_params={"timeout": (5, 30), "verify": False})
    return _client_cache


def get_current_price(symbol: str, retries: int = 3, backoff: float = 0.5) -> float:
    """Busca o preço de mercado mais recente para um símbolo com retries exponenciais.

    Args:
        symbol: par (ex: BTCUSDT)
        retries: número de tentativas
        backoff: base do tempo de espera entre tentativas (segundos)
    """
    client = _get_client()
    if client is None:
        raise RuntimeError("BINANCE_OFFLINE=1: não é possível buscar preço atual sem rede.")
    last_exc: Exception | None = None
    for i in range(max(1, retries)):
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:  # network hiccup, API timeout etc.
            last_exc = e
            if i == retries - 1:
                break
            time.sleep(backoff * (2 ** i))
    # Propaga a última exceção para o chamador poder decidir o fallback
    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to fetch current price without specific error")


def _download_klines(symbol: str, interval: str, start_ms: int, end_ms: int):
    client = _get_client()
    if start_ms > end_ms or client is None:
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
    client = _get_client()
    if client is None:
        return pd.DataFrame()
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


def get_cached_klines(symbol: str, interval: str, start_str, end_str=None) -> pd.DataFrame:
    """Retorna somente dados do cache local, sem chamadas à Binance.

    - Ajusta o end_ms para o último candle disponível no cache.
    - Se não houver dados, retorna DataFrame vazio.
    """
    start_ms: Optional[int] = to_timestamp_ms(start_str)
    end_ms: Optional[int] = to_timestamp_ms(end_str) if end_str else None

    if start_ms is None:
        # Se o start não é timestamp parseável, não tentamos rede; devolvemos vazio
        return pd.DataFrame(columns=["Date", "open", "high", "low", "close", "volume"])

    latest = _cache_latest_open_time(symbol, interval)
    if latest is None:
        return pd.DataFrame(columns=["Date", "open", "high", "low", "close", "volume"])

    if end_ms is None or end_ms > latest:
        end_ms = latest

    df = _cache_load_range(symbol, interval, start_ms, end_ms)
    return _format_output(df)
