from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import optuna
import pandas as pd
import sqlite3

from ..data.loader import update_cache_for_mtf, load_mtf_from_cache_or_binance
from ..data.align import merge_context
from ..indicators.common import ema, atr
from ..signals.ema_cross import generate_signals
from ..filters.apply import apply_all_filters
from ..engine.backtest import BacktestConfig, backtest_ema_cross
from ..storage.db import init_db, insert_run, finish_run, insert_bars, insert_trade, insert_metrics
from ..analysis.monte_carlo import save_artifact as save_json_artifact


def load_config() -> Dict[str, Any]:
    return json.loads(Path("src/strategies/experimento/config/config_active.json").read_text(encoding="utf-8"))


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
    sharpe = float(np.mean(rets) / (np.std(rets, ddof=1) + 1e-12)) if len(rets) >= 2 else 0.0
    return {"profit_factor": float(pf), "sharpe": float(sharpe), "trades": float(len(trades))}


def apply_indicators(df_base: pd.DataFrame, ctx_dfs: Dict[str, pd.DataFrame], params: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df_base.copy()
    df["ema_fast_30m"] = ema(df["close"], int(params.get("ema_fast", cfg["indicators"][0]["params"]["fast"])) )
    df["ema_slow_30m"] = ema(df["close"], int(params.get("ema_slow", cfg["indicators"][0]["params"]["slow"])) )
    df["atr_30m"] = atr(df, length=int(params.get("atr_len", cfg["indicators"][2]["params"]["length"])) )
    if "15m" in ctx_dfs:
        d15 = ctx_dfs["15m"].copy()
        ema_f = int(params.get("trend_ema_fast", cfg["filters"]["trend_tf"]["ema_fast"]))
        ema_s = int(params.get("trend_ema_slow", cfg["filters"]["trend_tf"]["ema_slow"]))
        d15["ema_fast_15m"] = ema(d15["close"], ema_f)
        d15["ema_slow_15m"] = ema(d15["close"], ema_s)
        df = merge_context(df, d15[["close_time", "ema_fast_15m", "ema_slow_15m"]], suffix="")
    return df


def objective_factory(train_df: pd.DataFrame, cfg: Dict[str, Any]):
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
        }
        if params["ema_fast"] >= params["ema_slow"] or params["trend_ema_fast"] >= params["trend_ema_slow"]:
            raise optuna.TrialPruned()

        df_sig = train_df.copy()
        filters_cfg = dict(cfg["filters"])  # copy
        filters_cfg.setdefault("atr_min", {})
        filters_cfg["atr_min"]["min_atr_frac"] = params["atr_min_frac"]
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

    # Prepare dataset
    base_tf = cfg["base_timeframe"]
    ctx_tfs = cfg["context_timeframes"]
    days = int(cfg["data"].get("days", 180))
    update_cache_for_mtf(cfg["symbol"], base_tf, ctx_tfs, days)
    df_base, ctx_dfs = load_mtf_from_cache_or_binance(cfg["symbol"], base_tf, ctx_tfs, days, use_cache_only=True)

    # Compute default indicators for base; per-trial faster to adjust base EMAs? For WFO we compute per window with fixed params.
    # We'll recompute per window with fixed params for validation.

    # Build merged df with default EMAs for signal generation; trial still uses EMA params in objective via train_df precomputed
    # Instead, compute signals per trial? Simpler: compute signals from params; so train_df must include EMAs from params.

    # Create windows by days on close_time
    df_base = df_base.sort_values("close_time").reset_index(drop=True)
    start_dt = pd.to_datetime(df_base["close_time"].iloc[0])
    end_dt = pd.to_datetime(df_base["close_time"].iloc[-1])
    opt_days = int(cfg["walk_forward"].get("opt_days", 60))
    val_days = int(cfg["walk_forward"].get("val_days", 20))
    step_days = int(cfg["walk_forward"].get("step_days", 20))

    windows: List[Dict[str, Any]] = []
    cur = start_dt
    while True:
        train_end = cur + timedelta(days=opt_days)
        val_end = train_end + timedelta(days=val_days)
        if val_end > end_dt:
            break
        windows.append({"train_start": cur, "train_end": train_end, "val_start": train_end, "val_end": val_end})
        cur = cur + timedelta(days=step_days)

    sampler = optuna.samplers.TPESampler(seed=int(cfg["optimization"].get("seed", 42)))

    # Prepare artifacts dir for this WFO run
    ts = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    wfo_dir = Path(cfg["storage"]["artifacts_dir"]) / f"wfo-{ts}"
    wfo_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = wfo_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for wi, w in enumerate(windows):
        # Build window datasets with indicators as defaults first
        # We'll compute indicators per trial inside objective by requiring EMAs/ATR already present? Simpler: apply indicators now with default, then override? For consistency, build params within objective, thus need EMAs present; We'll recompute inside apply_indicators and then signals.
        df_train_base = df_base[(df_base["close_time"] > w["train_start"]) & (df_base["close_time"] <= w["train_end"])].copy()
        df_val_base = df_base[(df_base["close_time"] > w["val_start"]) & (df_base["close_time"] <= w["val_end"])].copy()
        # Recompute indicators context for this window
        train_df = apply_indicators(df_train_base, ctx_dfs, params={}, cfg=cfg)
        # Generate signals based on default params, but objective will use trial signals; to keep simple, include signals computed with params in objective; Provide df having base ohlc and close_time; We'll compute inside objective for speed? We'll pass a precomputed 'train_df' with price+context EMAs? Already have context EMAs from apply_indicators with default values; But objective's EMA params differ; Accept small mismatch; Better: recompute EMAs using trial inside objective; For that, provide raw base and ctx dfs for the train window.
        # To keep it consistent, rebuild a closure using source slices.

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
            }
            if params["ema_fast"] >= params["ema_slow"] or params["trend_ema_fast"] >= params["trend_ema_slow"]:
                raise optuna.TrialPruned()
            df_tr = apply_indicators(df_train_base, ctx_dfs, params, cfg)
            df_sig = generate_signals(
                df_tr,
                fast_col="ema_fast_30m",
                slow_col="ema_slow_30m",
                side=cfg["signal_generators"][0]["params"].get("side", "long"),
                exit_on_cross=bool(cfg["signal_generators"][0]["params"].get("exit_on_cross", False)),
            )
            filters_cfg = dict(cfg["filters"])  # copy
            filters_cfg.setdefault("atr_min", {})
            filters_cfg["atr_min"]["min_atr_frac"] = params["atr_min_frac"]
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
            trades, _, _ = backtest_ema_cross(df_sig, run_id="opt", cfg=bt_cfg)
            m = compute_metrics(trades)
            if m["trades"] < int(cfg["optimization"].get("min_trades", 10)):
                return -1e6
            trade_sat = min(1.0, m["trades"] / float(cfg["optimization"].get("target_trades", 30)))
            score = float(cfg["optimization"].get("w_pf", 0.7)) * m["profit_factor"] * trade_sat + float(cfg["optimization"].get("w_sharpe", 0.3)) * m["sharpe"]
            return score

        study = optuna.create_study(
            study_name=f"wfo-{cfg['symbol']}-{cfg['base_timeframe']}-{wi}",
            storage="sqlite:///data/optuna_studies.db",
            direction="maximize",
            sampler=sampler,
            load_if_exists=True,
        )
        trials = int(cfg["optimization"].get("trials", 20))
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        best = study.best_params
        # Persist per-window best params
        (windows_dir / f"window_{wi:02d}_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

        # Validation with best params
        df_val = apply_indicators(df_val_base, ctx_dfs, best, cfg)
        df_val = generate_signals(
            df_val,
            fast_col="ema_fast_30m",
            slow_col="ema_slow_30m",
            side=cfg["signal_generators"][0]["params"].get("side", "long"),
            exit_on_cross=bool(cfg["signal_generators"][0]["params"].get("exit_on_cross", False)),
        )
        filters_cfg = dict(cfg["filters"])  # copy
        filters_cfg.setdefault("atr_min", {})
        filters_cfg["atr_min"]["min_atr_frac"] = float(best.get("atr_min_frac", filters_cfg["atr_min"].get("min_atr_frac", 0.001)))
        df_val = apply_all_filters(df_val, filters_cfg)

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

        run_id = f"wfo-{wi}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        with sqlite3.connect(storage["results_db"]) as cx:
            insert_run(cx, run_id, datetime.now(timezone.utc).isoformat(), cfg)
            cx.commit()

        trades, pnl, df_bt = backtest_ema_cross(df_val, run_id, bt_cfg)
        m = compute_metrics(trades)
        summary.append({"window": wi, "run_id": run_id, **m, "pnl_total": float(pnl), "best_params": best})

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
            finish_run(cx, run_id, datetime.now(timezone.utc).isoformat())
            cx.commit()

        print(f"WFO window {wi} done. PF={m['profit_factor']:.2f}, Sharpe={m['sharpe']:.2f}, Trades={m['trades']:.0f}")

    # Aggregated OOS metrics
    total_trades = sum(int(s["trades"]) for s in summary) if summary else 0
    total_pnl = sum(float(s["pnl_total"]) for s in summary) if summary else 0.0
    # Aggregate PF: sum profits / sum losses across windows
    agg_profit = 0.0
    agg_loss = 0.0
    # Load trades from DB to compute PF precisely
    with sqlite3.connect(storage["results_db"]) as cx:
        for s in summary:
            rid = s["run_id"]
            rows = pd.read_sql_query("SELECT pnl FROM trades WHERE run_id=?", cx, params=(rid,))
            p = rows[rows["pnl"] > 0]["pnl"].sum()
            l = -rows[rows["pnl"] < 0]["pnl"].sum()
            agg_profit += float(p)
            agg_loss += float(l)
    agg_pf = (agg_profit / agg_loss) if agg_loss > 0 else float("inf")

    # Combined equity curve (step at trade exits) across runs
    equity = []
    with sqlite3.connect(storage["results_db"]) as cx:
        capital0 = float(cfg["risk"]["capital"]) if summary else 0.0
        cur_equity = capital0
        # Collect all exits with timestamps
        exits = []
        for s in summary:
            rid = s["run_id"]
            df_tr = pd.read_sql_query("SELECT exit_time, pnl FROM trades WHERE run_id=? ORDER BY exit_time", cx, params=(rid,))
            for _, r in df_tr.iterrows():
                exits.append((pd.to_datetime(r["exit_time"]), float(r["pnl"])) )
        exits.sort(key=lambda x: x[0])
        for t, pnl in exits:
            cur_equity += pnl
            equity.append({"time": t.isoformat(), "equity": cur_equity})

    # Save summary + equity artifact
    summary_obj = {"windows": summary, "agg": {"trades": total_trades, "pnl_total": total_pnl, "profit_factor": agg_pf}}
    (wfo_dir / "wfo_summary.json").write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
    pd.DataFrame(equity).to_csv(wfo_dir / "equity_curve.csv", index=False)
    print("WFO completed with", len(summary), "windows.", "Agg PF=", f"{agg_pf:.2f}", "Trades=", total_trades)


if __name__ == "__main__":
    main()
