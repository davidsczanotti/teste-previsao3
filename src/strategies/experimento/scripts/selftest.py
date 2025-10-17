from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
import sqlite3

import pandas as pd

from ..data.synth_trend import generate_ohlcv_trend
from ..indicators.common import ema, atr
from ..signals.ema_cross import generate_signals
from ..filters.trend_mtf import apply_trend_gate
from ..filters.atr_threshold import apply_atr_threshold
from ..filters.volume import apply_volume_min
from ..engine.backtest import BacktestConfig, backtest_ema_cross
from ..storage.db import init_db, insert_run, finish_run, insert_bars, insert_signals, insert_trade, insert_metrics


def load_config():
    path = Path("src/strategies/experimento/config/config_active.json")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    cfg = load_config()
    storage = cfg["storage"]
    init_db(storage["results_db"])

    run_id = f"selftest-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    with sqlite3.connect(storage["results_db"]) as cx:
        insert_run(cx, run_id, datetime.now(timezone.utc).isoformat(), cfg)
        cx.commit()

    base_tf = cfg["base_timeframe"]
    days = int(cfg["data"].get("days", 30))
    tests = cfg.get("tests", {}).get("selftest", {})
    direction = tests.get("direction", "up")
    drift = float(tests.get("drift_per_bar", 0.001))
    seed = int(cfg["data"].get("seed", 123))

    # Generate trending base and context (15m) if needed
    df_base = generate_ohlcv_trend(base_tf, days=days, seed=seed, drift_per_bar=drift, direction=direction)

    # Indicators
    base_ema_fast = cfg["indicators"][0]["params"]["fast"]
    base_ema_slow = cfg["indicators"][0]["params"]["slow"]
    df_base["ema_fast_30m"] = ema(df_base["close"], base_ema_fast)
    df_base["ema_slow_30m"] = ema(df_base["close"], base_ema_slow)
    # ATR: detecta dinamicamente no JSON ou usa 14 por padrão
    atr_len = 14
    for ind in cfg.get("indicators", []):
        if ind.get("name") == "atr" and ind.get("tf") == base_tf:
            atr_len = int(ind.get("params", {}).get("length", atr_len))
            break
    df_base["atr_30m"] = atr(df_base, length=atr_len)

    # Signals + Filters
    df_signals = generate_signals(
        df_base,
        fast_col="ema_fast_30m",
        slow_col="ema_slow_30m",
        side=cfg["signal_generators"][0]["params"].get("side", "long"),
    )
    # In selftest, trend_ok sempre 1 para não bloquear tendenciosidade
    df_signals["trend_ok"] = 1
    df_signals["atr_ok"] = apply_atr_threshold(df_signals, "atr_30m", cfg["filters"]["atr_min"]["min_atr_frac"]).astype(int)
    df_signals["vol_ok"] = apply_volume_min(df_signals, cfg["filters"]["volume_min"]["percentile"]).astype(int)

    # Backtest
    # Para o selftest, desligamos stops/trailing para favorecer o efeito da tendência
    bt_cfg = BacktestConfig(
        capital=cfg["risk"]["capital"],
        fee_bp=cfg["risk"]["costs"]["fee_bp"],
        slippage_ticks=cfg["risk"]["costs"]["slippage_ticks"],
        tick_size=cfg["risk"]["costs"]["tick_size"],
        stop_mult=0.0,
        trailing_mult=0.0,
        side=cfg["signal_generators"][0]["params"].get("side", "long"),
    )
    trades, total_pnl, df_bt = backtest_ema_cross(df_signals, run_id, bt_cfg)

    with sqlite3.connect(storage["results_db"]) as cx:
        # Persist
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
                    "trend_ok": int(r.get("trend_ok", 1)),
                    "atr_ok": int(r.get("atr_ok", 1)),
                    "vol_ok": int(r.get("vol_ok", 1)),
                }
            )
        insert_bars(cx, run_id, rows)

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

        wins = [x for x in trades if x["pnl"] > 0]
        total_profit = sum(x["pnl"] for x in wins)
        total_loss = abs(sum(x["pnl"] for x in trades if x["pnl"] < 0))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float("inf")
        metrics = {
            "pnl_total": float(total_pnl),
            "trades": float(len(trades)),
            "profit_factor": float(profit_factor),
        }
        insert_metrics(cx, run_id, metrics)
        finish_run(cx, run_id, datetime.now(timezone.utc).isoformat())
        cx.commit()

    # Expectation: in uptrend, PF should be > 1 typically
    print(f"Selftest {run_id} finished. PnL: {total_pnl:.2f}, Trades: {len(trades)}, PF: {metrics['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
