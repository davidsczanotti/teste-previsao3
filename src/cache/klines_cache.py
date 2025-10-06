from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Iterable, Tuple

import pandas as pd

CACHE_PATH = Path("data/klines_cache.db")


def _get_connection() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS klines (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            open_time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (symbol, interval, open_time)
        )
        """
    )
    return conn


def _interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    if unit == "d":
        return value * 86_400_000
    if unit == "w":
        return value * 604_800_000
    if unit == "M":
        return value * 2_592_000_000
    raise ValueError(f"Unsupported interval: {interval}")


def to_timestamp_ms(value: str) -> Optional[int]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)


def store_rows(symbol: str, interval: str, rows: Iterable[Tuple[int, float, float, float, float, float]]) -> None:
    conn = _get_connection()
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO klines(symbol, interval, open_time, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ((symbol, interval, *row) for row in rows),
        )
    conn.close()


def load_range(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    conn = _get_connection()
    cursor = conn.execute(
        "SELECT open_time, open, high, low, close, volume FROM klines WHERE symbol=? AND interval=? AND open_time BETWEEN ? AND ? ORDER BY open_time",
        (symbol, interval, start_ms, end_ms),
    )
    data = cursor.fetchall()
    conn.close()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.drop(columns=["open_time"])
    return df[["Date", "open", "high", "low", "close", "volume"]]


def latest_open_time(symbol: str, interval: str) -> Optional[int]:
    conn = _get_connection()
    cur = conn.execute(
        "SELECT MAX(open_time) FROM klines WHERE symbol=? AND interval=?",
        (symbol, interval),
    )
    value = cur.fetchone()[0]
    conn.close()
    return value


def earliest_open_time(symbol: str, interval: str) -> Optional[int]:
    conn = _get_connection()
    cur = conn.execute(
        "SELECT MIN(open_time) FROM klines WHERE symbol=? AND interval=?",
        (symbol, interval),
    )
    value = cur.fetchone()[0]
    conn.close()
    return value


def ensure_cached(symbol: str, interval: str, start_ms: int, end_ms: int, downloader) -> None:
    interval_ms = _interval_to_ms(interval)

    # fetch newest data if needed
    latest = latest_open_time(symbol, interval)
    fetch_start = None
    if latest is None:
        fetch_start = start_ms
    elif latest < end_ms - interval_ms:
        fetch_start = max(start_ms, latest + interval_ms)

    if fetch_start is not None and fetch_start <= end_ms:
        new_rows = downloader(fetch_start, end_ms)
        if new_rows:
            store_rows(symbol, interval, new_rows)

    # fetch older data if needed
    earliest = earliest_open_time(symbol, interval)
    if earliest is None or earliest > start_ms:
        fetch_end = earliest - interval_ms if earliest is not None else end_ms
        if fetch_end >= start_ms:
            new_rows = downloader(start_ms, fetch_end)
            if new_rows:
                store_rows(symbol, interval, new_rows)


def cached_klines(symbol: str, interval: str, start_ms: int, end_ms: int, downloader) -> pd.DataFrame:
    ensure_cached(symbol, interval, start_ms, end_ms, downloader)
    return load_range(symbol, interval, start_ms, end_ms)
