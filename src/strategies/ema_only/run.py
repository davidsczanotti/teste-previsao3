from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ...utils.data_loader import load_data
from .backtest import EmaOnlyParams, backtest_ema_only


CFG_PATH = Path("src/strategies/ema_only/config.json")


def _default(o: Any) -> Any:
    try:
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        import pandas as pd  # type: ignore

        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def main() -> None:
    cfg = json.loads(CFG_PATH.read_text())
    data_cfg: Dict[str, Any] = cfg.get("data", {})
    strat_cfg: Dict[str, Any] = cfg.get("strategy", {})
    bt_cfg: Dict[str, Any] = cfg.get("backtest", {})

    symbol = str(data_cfg.get("symbol", "BTCUSDT"))
    timeframe = str(data_cfg.get("timeframe", "1h"))
    days = int(data_cfg.get("days", 3650))

    # Carrega dados exclusivamente do cache local
    df = load_data(symbol, timeframe, days=days, use_cache_only=True)

    params = EmaOnlyParams(
        ema_period=int(strat_cfg.get("ema_period", 8)),
        lot_size=float(strat_cfg.get("lot_size", 0.001)),
        fee_rate=float(strat_cfg.get("fee_pct", 0.001)),
        use_cross=bool(strat_cfg.get("use_cross", True)),
    )

    initial_capital = float(bt_cfg.get("initial_capital", 1000.0))
    trades, total_pnl, stats = backtest_ema_only(df, params=params, initial_capital=initial_capital)

    print("EMA-only Backtest Summary (config-driven):")
    print(
        f"Symbol={symbol} Interval={timeframe} Days={days} EMA={params.ema_period} "
        f"UseCross={params.use_cross} Lot={params.lot_size} Fee={params.fee_rate}"
    )
    print(
        "PnL: ${:.2f} | Return: {:.2f}% | Trades: {} | Win rate: {:.2f}% | MDD: {:.2f}%".format(
            stats["pnl"],
            stats["return_pct"],
            stats["num_trades"],
            stats["win_rate"],
            stats["max_drawdown_pct"],
        )
    )

    outdir = Path(bt_cfg.get("outdir", "src/strategies/ema_only/reports/backtest"))
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / f"ema_only_{symbol}_{timeframe}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"params": params.__dict__, "stats": stats, "trades": trades},
            f,
            ensure_ascii=False,
            indent=2,
            default=_default,
        )
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()

