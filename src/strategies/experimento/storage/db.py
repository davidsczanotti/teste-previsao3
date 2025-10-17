from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE,
    started_at TEXT,
    finished_at TEXT,
    config_json TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS bars (
    run_id TEXT,
    idx INTEGER,
    close_time TEXT,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    ema_fast_30m REAL, ema_slow_30m REAL,
    atr_30m REAL,
    ema_fast_15m REAL, ema_slow_15m REAL,
    signal INTEGER,
    trend_ok INTEGER,
    atr_ok INTEGER,
    vol_ok INTEGER
);

CREATE TABLE IF NOT EXISTS signals (
    run_id TEXT,
    idx INTEGER,
    close_time TEXT,
    signal INTEGER
);

CREATE TABLE IF NOT EXISTS trades (
    run_id TEXT,
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_idx INTEGER,
    exit_idx INTEGER,
    entry_time TEXT,
    exit_time TEXT,
    side TEXT,
    qty REAL,
    entry_price REAL,
    exit_price REAL,
    pnl REAL
);

CREATE TABLE IF NOT EXISTS fills (
    run_id TEXT,
    idx INTEGER,
    time TEXT,
    side TEXT,
    qty REAL,
    price REAL,
    fee REAL
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT,
    key TEXT,
    value REAL
);

CREATE TABLE IF NOT EXISTS params (
    run_id TEXT,
    key TEXT,
    value TEXT
);
"""


def init_db(db_path: str | Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as cx:
        cx.executescript(SCHEMA)


def insert_run(cx: sqlite3.Connection, run_id: str, started_at: str, config: Dict[str, Any]) -> None:
    cx.execute(
        "INSERT OR REPLACE INTO runs (run_id, started_at, config_json) VALUES (?, ?, ?)",
        (run_id, started_at, json.dumps(config)),
    )


def finish_run(cx: sqlite3.Connection, run_id: str, finished_at: str, notes: str | None = None) -> None:
    cx.execute(
        "UPDATE runs SET finished_at = ?, notes = ? WHERE run_id = ?",
        (finished_at, notes, run_id),
    )


def insert_bars(
    cx: sqlite3.Connection,
    run_id: str,
    rows: Iterable[Dict[str, Any]],
) -> None:
    cx.executemany(
        (
            "INSERT INTO bars (run_id, idx, close_time, open, high, low, close, volume, "
            "ema_fast_30m, ema_slow_30m, atr_30m, ema_fast_15m, ema_slow_15m, signal, trend_ok, atr_ok, vol_ok) "
            "VALUES (:run_id, :idx, :close_time, :open, :high, :low, :close, :volume, :ema_fast_30m, :ema_slow_30m, :atr_30m, :ema_fast_15m, :ema_slow_15m, :signal, :trend_ok, :atr_ok, :vol_ok)"
        ),
        list(rows),
    )


def insert_signals(cx: sqlite3.Connection, run_id: str, rows: Iterable[Dict[str, Any]]) -> None:
    cx.executemany(
        "INSERT INTO signals (run_id, idx, close_time, signal) VALUES (:run_id, :idx, :close_time, :signal)",
        list(rows),
    )


def insert_trade(
    cx: sqlite3.Connection,
    run_id: str,
    entry_idx: int,
    exit_idx: int,
    entry_time: str,
    exit_time: str,
    side: str,
    qty: float,
    entry_price: float,
    exit_price: float,
    pnl: float,
) -> None:
    cx.execute(
        (
            "INSERT INTO trades (run_id, entry_idx, exit_idx, entry_time, exit_time, side, qty, entry_price, exit_price, pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        (run_id, entry_idx, exit_idx, entry_time, exit_time, side, qty, entry_price, exit_price, pnl),
    )


def insert_fill(
    cx: sqlite3.Connection,
    run_id: str,
    idx: int,
    time: str,
    side: str,
    qty: float,
    price: float,
    fee: float,
) -> None:
    cx.execute(
        "INSERT INTO fills (run_id, idx, time, side, qty, price, fee) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, idx, time, side, qty, price, fee),
    )


def insert_metrics(cx: sqlite3.Connection, run_id: str, metrics: Dict[str, float]) -> None:
    rows = [(run_id, k, float(v)) for k, v in metrics.items()]
    cx.executemany("INSERT INTO metrics (run_id, key, value) VALUES (?, ?, ?)", rows)
