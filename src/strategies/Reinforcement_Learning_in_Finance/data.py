from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ...cache.klines_cache import (
    earliest_open_time,
    latest_open_time,
    load_range,
    to_timestamp_ms,
)


@dataclass(frozen=True)
class TimeBounds:
    """Container with start and end timestamps (milliseconds since epoch)."""

    start_ms: int
    end_ms: int


class CacheUnavailableError(RuntimeError):
    """Raised when no cached data is available for the requested query."""


def _resolve_time_bounds(
    symbol: str,
    interval: str,
    start: Optional[str],
    end: Optional[str],
) -> TimeBounds:
    """Compute the millisecond interval that must be pulled from cache.

    Args:
        symbol: Trading pair ticker (e.g. ``"BTCUSDT"``).
        interval: Kline timeframe (e.g. ``"5m"``).
        start: Optional lower bound as an ISO date string.
        end: Optional upper bound as an ISO date string.

    Returns:
        A ``TimeBounds`` instance with normalized millisecond timestamps.

    Raises:
        CacheUnavailableError: If the cache has no klines for the request.
        ValueError: If the resulting time window would be empty.
    """

    earliest = earliest_open_time(symbol, interval)
    latest = latest_open_time(symbol, interval)
    if earliest is None or latest is None:
        raise CacheUnavailableError(
            "Nenhum candle encontrado no cache local. Execute "
            "`poetry run python -m scripts.populate_cache` antes de treinar."
        )

    start_ms = to_timestamp_ms(start) if start else earliest
    if start_ms is None:
        raise ValueError(f"Não foi possível interpretar a data inicial: {start}")
    start_ms = max(start_ms, earliest)

    end_ms = to_timestamp_ms(end) if end else latest
    if end_ms is None:
        raise ValueError(f"Não foi possível interpretar a data final: {end}")
    end_ms = min(end_ms, latest)

    if start_ms >= end_ms:
        raise ValueError("Intervalo temporal inválido: início é maior ou igual ao fim.")

    return TimeBounds(start_ms=start_ms, end_ms=end_ms)


def load_price_history(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch cached candlestick data and add daily returns for RL usage.

    Args:
        symbol: Trading pair ticker (default: ``"BTCUSDT"``).
        interval: Kline timeframe (default: ``"5m"``).
        start: Optional ISO timestamp string to trim the beginning.
        end: Optional ISO timestamp string to trim the end.

    Returns:
        A ``pandas.DataFrame`` ordered by time with the columns:
        ``["Date", "open", "high", "low", "close", "volume", "return"]``.
        The ``return`` column contains simple percentage change of the close.

    Raises:
        CacheUnavailableError: If the local cache has no rows to serve.
        ValueError: If the requested interval is empty or malformed.
    """

    bounds = _resolve_time_bounds(symbol, interval, start, end)
    df = load_range(symbol, interval, bounds.start_ms, bounds.end_ms)
    if df.empty:
        raise CacheUnavailableError(
            "O cache retornou vazio. Confirme o símbolo/intervalo e rode o script "
            "de `populate_cache` para atualizar os dados."
        )

    df = df.sort_values("Date").reset_index(drop=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["return"] = df["close"].pct_change().fillna(0.0)
    return df

