from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from ..data.synth import generate_ohlcv
from ..data.utils import tf_to_minutes
from ..data.align import merge_context
from ..data.loader import load_mtf_from_cache_or_binance, update_cache_for_mtf
from ..indicators.common import ema, atr
from ..signals.ema_cross import generate_signals
from ..filters.apply import apply_all_filters
from ..engine.backtest import BacktestConfig, backtest_ema_cross
from ..storage.db import init_db, insert_run, finish_run, insert_bars, insert_signals, insert_trade, insert_metrics


def load_config() -> Dict[str, Any]:
    config_path = Path("src/strategies/experimento/config/config_active.json")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_artifacts_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = load_config()
    storage = cfg["storage"]
    init_db(storage["results_db"])

    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    started_at = datetime.now(timezone.utc).isoformat()

    import sqlite3

    with sqlite3.connect(storage["results_db"]) as cx:
        insert_run(cx, run_id, started_at, cfg)
        cx.commit()

    base_tf = cfg["base_timeframe"]
    ctx_tfs = cfg["context_timeframes"]
    days = int(cfg["data"].get("days", 30))
    source = cfg["data"].get("source", "cache").lower()
    auto_update = bool(cfg["data"].get("update_cache", True))

    if source == "synthetic":
        seed = int(cfg["data"].get("seed", 42))
        df_base = generate_ohlcv(base_tf, days=days, seed=seed)
        ctx_dfs = {tf: generate_ohlcv(tf, days=days, seed=seed + i + 1) for i, tf in enumerate(ctx_tfs)}
    else:
        # Atualiza o cache antes de consumir
        if auto_update:
            update_cache_for_mtf(symbol=cfg["symbol"], base_tf=base_tf, ctx_tfs=ctx_tfs, days=days)
        use_cache_only = True  # sempre consumir do cache
        df_base, ctx_dfs = load_mtf_from_cache_or_binance(
            symbol=cfg["symbol"], base_tf=base_tf, ctx_tfs=ctx_tfs, days=days, use_cache_only=use_cache_only
        )

    # Indicators per TF
    # Base TF: EMA fast/slow and ATR
    base_ema_fast = cfg["indicators"][0]["params"]["fast"]
    base_ema_slow = cfg["indicators"][0]["params"]["slow"]
    df_base["ema_fast_30m"] = ema(df_base["close"], base_ema_fast)
    df_base["ema_slow_30m"] = ema(df_base["close"], base_ema_slow)
    df_base["atr_30m"] = atr(df_base, length=cfg["indicators"][2]["params"]["length"])

    # 15m trend EMA
    if "15m" in ctx_dfs:
        df15 = ctx_dfs["15m"].copy()
        df15["ema_fast_15m"] = ema(df15["close"], cfg["filters"]["trend_tf"]["ema_fast"])
        df15["ema_slow_15m"] = ema(df15["close"], cfg["filters"]["trend_tf"]["ema_slow"])
        df_base = merge_context(df_base, df15[["close_time", "ema_fast_15m", "ema_slow_15m"]], suffix="")

    # 5m can be used for future extension; skip for minimal pipeline

    # Signals
    df_signals = generate_signals(
        df_base,
        fast_col="ema_fast_30m",
        slow_col="ema_slow_30m",
        side=cfg["signal_generators"][0]["params"].get("side", "long"),
        exit_on_cross=bool(cfg["signal_generators"][0]["params"].get("exit_on_cross", False)),
    )

    # Filters
    df_signals = apply_all_filters(df_signals, cfg["filters"]) 

    # Backtest
    bt_cfg = BacktestConfig(
        capital=cfg["risk"]["capital"],
        fee_bp=cfg["risk"]["costs"]["fee_bp"],
        slippage_ticks=cfg["risk"]["costs"]["slippage_ticks"],
        tick_size=cfg["risk"]["costs"]["tick_size"],
        stop_mult=cfg["risk"]["stop"]["mult"],
        trailing_mult=cfg["risk"]["trailing"]["mult"],
        side=cfg["signal_generators"][0]["params"].get("side", "long"),
        exit_on_cross=bool(cfg["signal_generators"][0]["params"].get("exit_on_cross", False)),
    )

    trades, total_pnl, df_bt = backtest_ema_cross(df_signals, run_id, bt_cfg)

    # Persist results
    with sqlite3.connect(storage["results_db"]) as cx:
        # Bars
        rows = []
        for i, r in df_bt.reset_index(drop=True).iterrows():
            rows.append(
                {
                    "run_id": run_id,
                    "idx": i,
                    "close_time": str(r["close_time"]),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": float(r["volume"]),
                    "ema_fast_30m": float(r.get("ema_fast_30m", float("nan"))),
                    "ema_slow_30m": float(r.get("ema_slow_30m", float("nan"))),
                    "atr_30m": float(r.get("atr_30m", float("nan"))),
                    "ema_fast_15m": float(r.get("ema_fast_15m", float("nan"))),
                    "ema_slow_15m": float(r.get("ema_slow_15m", float("nan"))),
                    "signal": int(r.get("signal", 0)),
                    "trend_ok": int(r.get("trend_ok", 0)),
                    "atr_ok": int(r.get("atr_ok", 0)),
                    "vol_ok": int(r.get("vol_ok", 0)),
                }
            )
        insert_bars(cx, run_id, rows)

        # Signals table (simple projection)
        sig_rows = [
            {
                "run_id": run_id,
                "idx": i,
                "close_time": str(r["close_time"]),
                "signal": int(r.get("signal", 0)),
            }
            for i, r in df_bt.reset_index(drop=True).iterrows()
        ]
        insert_signals(cx, run_id, sig_rows)

        # Trades
        for t in trades:
            insert_trade(
                cx,
                run_id,
                entry_idx=int(t["entry_idx"]),
                exit_idx=int(t["exit_idx"]),
                entry_time=str(t["entry_time"]),
                exit_time=str(t["exit_time"]),
                side=str(t["side"]),
                qty=float(t["qty"]),
                entry_price=float(t["entry_price"]),
                exit_price=float(t["exit_price"]),
                pnl=float(t["pnl"]),
            )

        # Basic metrics
        closed = trades
        wins = [x for x in closed if x["pnl"] > 0]
        total_profit = sum(x["pnl"] for x in wins)
        total_loss = abs(sum(x["pnl"] for x in closed if x["pnl"] < 0))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float("inf")
        metrics = {
            "pnl_total": float(total_pnl),
            "trades": float(len(closed)),
            "profit_factor": float(profit_factor),
        }
        insert_metrics(cx, run_id, metrics)

        finish_run(cx, run_id, datetime.now(timezone.utc).isoformat())
        cx.commit()

    print(f"Run {run_id} finished. PnL: {total_pnl:.2f}, Trades: {len(trades)}, PF: {metrics['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
