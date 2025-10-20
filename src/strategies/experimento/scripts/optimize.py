from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
import sqlite3

from ..data.loader import update_cache_for_mtf, load_mtf_from_cache_or_binance
from ..data.align import merge_context
from ..indicators.common import ema, atr, compute_ma, vwap_daily
from ..signals.ema_cross import generate_signals
from ..filters.apply import apply_all_filters
from ..engine.backtest import BacktestConfig, backtest_ema_cross
from ..storage.db import init_db, insert_run, finish_run, insert_bars, insert_signals, insert_trade, insert_metrics, insert_params


def load_config() -> Dict[str, Any]:
    path = Path("src/strategies/experimento/config/config_active.json")
    return json.loads(path.read_text(encoding="utf-8"))


def compute_metrics(trades) -> Dict[str, float]:
    wins = [t for t in trades if t["pnl"] > 0]
    total_profit = sum(t["pnl"] for t in wins)
    total_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = (total_profit / total_loss) if total_loss > 0 else float("inf")

    rets = []
    for t in trades:
        denom = t["entry_price"] * max(t["qty"], 1e-12)
        if denom > 0:
            rets.append(t["pnl"] / denom)
    if len(rets) >= 2:
        sharpe = float(np.mean(rets) / (np.std(rets, ddof=1) + 1e-12))
    else:
        sharpe = 0.0
    return {"profit_factor": float(pf), "sharpe": float(sharpe), "trades": float(len(trades))}


def build_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    base_tf = cfg["base_timeframe"]
    ctx_tfs = cfg["context_timeframes"]
    days = int(cfg["data"].get("days", 180))
    auto_update = bool(cfg.get("data", {}).get("update_cache", True))
    if auto_update:
        update_cache_for_mtf(symbol=cfg["symbol"], base_tf=base_tf, ctx_tfs=ctx_tfs, days=days)
        use_cache_only = False
    else:
        use_cache_only = True
    df_base, ctx_dfs = load_mtf_from_cache_or_binance(
        symbol=cfg["symbol"], base_tf=base_tf, ctx_tfs=ctx_tfs, days=days, use_cache_only=use_cache_only
    )

    # Base indicators (will be re-computed per trial if params vary)
    return df_base, ctx_dfs


def apply_indicators(df_base: pd.DataFrame, ctx_dfs: Dict[str, pd.DataFrame], params: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df_base.copy()
    # Base EMA & ATR
    df["ema_fast_30m"] = ema(df["close"], int(params.get("ema_fast", cfg["indicators"][0]["params"]["fast"])) )
    df["ema_slow_30m"] = ema(df["close"], int(params.get("ema_slow", cfg["indicators"][0]["params"]["slow"])) )
    # Dynamic ATR length from config or fallback 14
    atr_default = 14
    for ind in cfg.get("indicators", []):
        if ind.get("name") == "atr" and ind.get("tf") == cfg["base_timeframe"]:
            atr_default = int(ind.get("params", {}).get("length", atr_default))
            break
    df["atr_30m"] = atr(df, length=int(params.get("atr_len", atr_default)) )

    # Legacy 15m EMA trend
    if "trend_tf" in cfg.get("filters", {}) and "15m" in ctx_dfs:
        d15 = ctx_dfs["15m"].copy()
        ema_f = int(params.get("trend_ema_fast", cfg["filters"]["trend_tf"]["ema_fast"]))
        ema_s = int(params.get("trend_ema_slow", cfg["filters"]["trend_tf"]["ema_slow"]))
        d15["ema_fast_15m"] = ema(d15["close"], ema_f)
        d15["ema_slow_15m"] = ema(d15["close"], ema_s)
        df = merge_context(df, d15[["close_time", "ema_fast_15m", "ema_slow_15m"]], suffix="")

    # Generic MA trend (ma_trend) — usa params se presentes
    if "ma_trend" in cfg.get("filters", {}):
        mt = cfg["filters"]["ma_trend"]
        ma_type = mt.get("ma_type", "ema")
        tf_mt = mt.get("tf", "15m")
        fast = int(params.get("ma_fast", mt.get("fast", 9)))
        slow = int(params.get("ma_slow", mt.get("slow", 20)))
        if tf_mt == cfg["base_timeframe"]:
            df[f"ma_fast_{tf_mt}"] = compute_ma(df["close"], ma_type, fast)
            df[f"ma_slow_{tf_mt}"] = compute_ma(df["close"], ma_type, slow)
        else:
            if tf_mt in ctx_dfs:
                d = ctx_dfs[tf_mt].copy()
                d[f"ma_fast_{tf_mt}"] = compute_ma(d["close"], ma_type, fast)
                d[f"ma_slow_{tf_mt}"] = compute_ma(d["close"], ma_type, slow)
                df = merge_context(df, d[["close_time", f"ma_fast_{tf_mt}", f"ma_slow_{tf_mt}"]], suffix="")

    # VWAP bias support (compute vwap_<tf> if configured)
    if "vwap_bias" in cfg.get("filters", {}):
        vb = cfg["filters"]["vwap_bias"]
        tf_v = vb.get("tf", cfg["base_timeframe"])
        if tf_v == cfg["base_timeframe"]:
            df[f"vwap_{tf_v}"] = vwap_daily(df)
        else:
            if tf_v in ctx_dfs:
                d = ctx_dfs[tf_v].copy()
                d[f"vwap_{tf_v}"] = vwap_daily(d)
                df = merge_context(df, d[["close_time", f"vwap_{tf_v}"]], suffix="")
    return df


def objective_factory(df_base: pd.DataFrame, ctx_dfs: Dict[str, pd.DataFrame], cfg: Dict[str, Any]):
    opt = cfg["optimization"]
    min_trades = int(opt.get("min_trades", 10))
    target_trades = float(opt.get("target_trades", 30))
    w_pf = float(opt.get("w_pf", 0.7))
    w_sharpe = float(opt.get("w_sharpe", 0.3))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "ema_fast": trial.suggest_int("ema_fast", 5, 30),
            "ema_slow": trial.suggest_int("ema_slow", 10, 60),
            "atr_len": trial.suggest_int("atr_len", 7, 28),
            "atr_min_frac": trial.suggest_float("atr_min_frac", 0.0005, 0.01, log=True),
            "trend_ema_fast": trial.suggest_int("trend_ema_fast", 20, 80),
            "trend_ema_slow": trial.suggest_int("trend_ema_slow", 100, 250),
            "stop_mult": trial.suggest_float("stop_mult", 0.0, 4.0),
            "trailing_mult": trial.suggest_float("trailing_mult", 0.0, 4.0),
            # novos knobs
            "ma_fast": trial.suggest_int("ma_fast", 8, 30),
            "ma_slow": trial.suggest_int("ma_slow", 20, 80),
            "vol_pct": trial.suggest_float("vol_pct", 0.4, 0.8),
        }
        if params["ema_fast"] >= params["ema_slow"]:
            raise optuna.TrialPruned()
        if params["trend_ema_fast"] >= params["trend_ema_slow"]:
            raise optuna.TrialPruned()
        if params["ma_fast"] >= params["ma_slow"]:
            raise optuna.TrialPruned()

        df = apply_indicators(df_base, ctx_dfs, params, cfg)
        df_sig = generate_signals(
            df,
            fast_col="ema_fast_30m",
            slow_col="ema_slow_30m",
            side=cfg["signal_generators"][0]["params"].get("side", "long"),
            exit_on_cross=bool(cfg["signal_generators"][0]["params"].get("exit_on_cross", False)),
        )

        # Override filter params with trial
        filters_cfg = dict(cfg["filters"])  # copy
        filters_cfg.setdefault("atr_min", {})
        filters_cfg["atr_min"]["min_atr_frac"] = params["atr_min_frac"]
        # volume min
        filters_cfg.setdefault("volume_min", {})
        filters_cfg["volume_min"]["percentile"] = params["vol_pct"]
        df_sig = apply_all_filters(df_sig, filters_cfg)

        bt_cfg = BacktestConfig(
            capital=cfg["risk"]["capital"],
            fee_bp=cfg["risk"]["costs"]["fee_bp"],
            slippage_ticks=cfg["risk"]["costs"]["slippage_ticks"],
            tick_size=cfg["risk"]["costs"]["tick_size"],
            stop_mult=float(params["stop_mult"]),
            trailing_mult=float(params["trailing_mult"]),
            side=cfg["signal_generators"][0]["params"].get("side", "long"),
            exit_on_cross=bool(cfg["signal_generators"][0]["params"].get("exit_on_cross", False)),
        )
        trades, pnl, _ = backtest_ema_cross(df_sig, run_id="opt", cfg=bt_cfg)
        m = compute_metrics(trades)
        n_trades = m["trades"]
        if n_trades < min_trades:
            return -1e6
        trade_sat = min(1.0, n_trades / target_trades)
        score = w_pf * m["profit_factor"] * trade_sat + w_sharpe * m["sharpe"]
        return float(score)

    return objective


def main() -> None:
    cfg = load_config()
    storage = cfg["storage"]
    init_db(storage["results_db"])

    # Load dataset once
    df_base, ctx_dfs = build_dataset(cfg)

    # Split train/valid by fraction of bars based on optimization.walk_forward? For optimize only, use entire data.
    # Simplify: use entire data for objective (user can run walk_forward for OOS).

    study_name = f"experimento-{cfg['symbol']}-{cfg['base_timeframe']}"
    sampler = optuna.samplers.TPESampler(seed=int(cfg["optimization"].get("seed", 42)))
    storage_url = "sqlite:///data/optuna_studies.db"
    study = optuna.create_study(study_name=study_name, storage=storage_url, direction="maximize", sampler=sampler, load_if_exists=True)

    objective = objective_factory(df_base, ctx_dfs, cfg)
    trials = int(cfg["optimization"].get("trials", 30))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best = study.best_params
    print("Best score:", study.best_value)
    print("Best params:", best)

    # Run backtest with best params and persist
    df = apply_indicators(df_base, ctx_dfs, best, cfg)
    df_sig = generate_signals(
        df,
        fast_col="ema_fast_30m",
        slow_col="ema_slow_30m",
        side=cfg["signal_generators"][0]["params"].get("side", "long"),
        exit_on_cross=bool(cfg["signal_generators"][0]["params"].get("exit_on_cross", False)),
    )
    filters_cfg = dict(cfg["filters"])  # copy
    filters_cfg.setdefault("atr_min", {})
    filters_cfg["atr_min"]["min_atr_frac"] = float(best.get("atr_min_frac", filters_cfg.get("atr_min", {}).get("min_atr_frac", 0.001)))
    df_sig = apply_all_filters(df_sig, filters_cfg)

    bt_cfg = BacktestConfig(
        capital=cfg["risk"]["capital"],
        fee_bp=cfg["risk"]["costs"]["fee_bp"],
        slippage_ticks=cfg["risk"]["costs"]["slippage_ticks"],
        tick_size=cfg["risk"]["costs"]["tick_size"],
        stop_mult=float(best.get("stop_mult", 0.0)),
        trailing_mult=float(best.get("trailing_mult", 0.0)),
        side=cfg["signal_generators"][0]["params"].get("side", "long"),
        exit_on_cross=bool(cfg["signal_generators"][0]["params"].get("exit_on_cross", False)),
    )

    run_id = f"opt-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    with sqlite3.connect(storage["results_db"]) as cx:
        insert_run(cx, run_id, datetime.now(timezone.utc).isoformat(), cfg)
        cx.commit()

    trades, pnl, df_bt = backtest_ema_cross(df_sig, run_id, bt_cfg)
    m = compute_metrics(trades)

    # Persist
    with sqlite3.connect(storage["results_db"]) as cx:
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
        insert_metrics(cx, run_id, {**m, "pnl_total": float(pnl)})
        # Persist best params in DB to avoid relying on files only
        insert_params(cx, run_id, {f"best.{k}": v for k, v in best.items()})
        finish_run(cx, run_id, datetime.now(timezone.utc).isoformat())
        cx.commit()

    # Save best params artifact
    if bool(cfg["storage"].get("write_artifacts", True)):
        artifacts = Path(cfg["storage"]["artifacts_dir"]) / run_id
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Optimization finished. Best score={study.best_value:.4f}. Run {run_id} saved.")


if __name__ == "__main__":
    main()
