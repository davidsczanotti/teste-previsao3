from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...utils.data_loader import load_data
from .backtest import EmaOnlyParams, backtest_ema_only, compute_ema


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


def prepare_dataset_with_reference(
    symbol: str,
    timeframe: str,
    days: int,
    use_cache_only: bool,
    ref_timeframe: Optional[str],
    ref_days: Optional[int],
    ref_ema_period: Optional[int],
) -> pd.DataFrame:
    df = load_data(symbol, timeframe, days=days, use_cache_only=use_cache_only).sort_values("Date").reset_index(drop=True)

    if ref_timeframe and ref_ema_period:
        ref_df = load_data(symbol, ref_timeframe, days=ref_days or days, use_cache_only=use_cache_only)
        ref_df = ref_df.sort_values("Date").reset_index(drop=True).copy()
        ref_df["ref_ema"] = compute_ema(ref_df["close"].astype(float), int(ref_ema_period))
        ref_df = ref_df[["Date", "ref_ema"]]
        df = pd.merge_asof(df, ref_df, on="Date", direction="backward")

    return df


def main() -> None:
    cfg = json.loads(CFG_PATH.read_text())
    data_cfg: Dict[str, Any] = cfg.get("data", {})
    strat_cfg: Dict[str, Any] = cfg.get("strategy", {})
    bt_cfg: Dict[str, Any] = cfg.get("backtest", {})

    symbol = str(data_cfg.get("symbol", "BTCUSDT"))
    timeframe = str(data_cfg.get("timeframe", "1h"))
    days = int(data_cfg.get("days", 3650))
    ref_timeframe = data_cfg.get("ref_timeframe")
    ref_days = data_cfg.get("ref_days")
    ref_ema_period = strat_cfg.get("ref_ema_period")
    slow_period_raw = strat_cfg.get("slow_ema_period")
    trend_period_raw = strat_cfg.get("trend_filter_period")

    # Carrega dados exclusivamente do cache local, opcionalmente com EMA de referência de TF superior
    df = prepare_dataset_with_reference(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        use_cache_only=True,
        ref_timeframe=ref_timeframe,
        ref_days=ref_days,
        ref_ema_period=ref_ema_period,
    )

    params = EmaOnlyParams(
        ema_period=int(strat_cfg.get("ema_period", 8)),
        slow_ema_period=int(slow_period_raw) if slow_period_raw is not None else None,
        trend_filter_period=int(trend_period_raw) if trend_period_raw is not None else None,
        use_trend_filter=bool(strat_cfg.get("use_trend_filter", False)),
        pullback_pct=float(strat_cfg.get("pullback_pct", 0.0)),
        ref_filter_enabled=bool(strat_cfg.get("ref_filter_enabled", False)),
        ref_ema_period=int(ref_ema_period) if ref_ema_period is not None else None,
        ref_buffer_pct=float(strat_cfg.get("ref_buffer_pct", 0.0)),
        ref_timeframe=str(ref_timeframe) if ref_timeframe else None,
        lot_size=float(strat_cfg.get("lot_size", 0.001)),
        fee_rate=float(strat_cfg.get("fee_pct", 0.001)),
        use_cross=bool(strat_cfg.get("use_cross", False)),
        signal_mode=str(strat_cfg.get("signal_mode", "price_reversion")),
    )

    initial_capital = float(bt_cfg.get("initial_capital", 1000.0))
    trades, total_pnl, stats = backtest_ema_only(df, params=params, initial_capital=initial_capital)

    print("EMA-only Backtest Summary (config-driven):")
    print(
        f"Symbol={symbol} Interval={timeframe} Days={days} Mode={params.signal_mode} "
        f"EMA={params.ema_period} SlowEMA={params.slow_ema_period} "
        f"TrendFilter={params.use_trend_filter}:{params.trend_filter_period} "
        f"RefTF={params.ref_timeframe} RefEMA={params.ref_ema_period} RefBuf={params.ref_buffer_pct} "
        f"Pullback={params.pullback_pct} UseCross={params.use_cross} "
        f"Lot={params.lot_size} Fee={params.fee_rate}"
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
