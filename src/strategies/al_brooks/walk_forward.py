#!/usr/bin/env python3
"""
No-flags Walk-Forward runner for the Al Brooks strategy.
Reads windows from src/strategies/al_brooks/config.json (key: walk_forward).
Saves an audit-friendly JSON summary and moves the chart into the strategy's
reports folder.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict

from ...utils.walk_forward import WalkForwardValidator
from .backtest import backtest_al_brooks_inside_bar
from .config import load_active_config
from ...utils.data_loader import load_data
from ...utils.metrics import calculate_metrics
from datetime import datetime, UTC


def main() -> None:
    cfg = load_active_config("BTCUSDT", "1m")
    if not cfg:
        print("Nenhuma configuração encontrada em config.json.")
        return

    # Defaults for WF windows
    # You can add these into config.json as a `walk_forward` block to customize
    wf_defaults: Dict[str, Any] = {
        "opt_window": 90,  # days
        "val_window": 30,  # days
        "step_size": 30,   # days
        "min_trades": 10,
        "cache_only": True,
    }
    # Try to read optional block
    cfg_path = Path(__file__).resolve().parent / "config.json"
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        wf = dict(raw.get("walk_forward", {}))
        wf_cfg = {**wf_defaults, **wf}
    except Exception:
        wf_cfg = wf_defaults

    # Decide locks for Inside Bar from config
    use_ib = bool(getattr(cfg, "use_inside_bar", True))
    ib_inclusive = bool(getattr(cfg, "inside_bar_inclusive", False))
    lock = wf_cfg.get("lock") or {}
    if isinstance(lock, dict):
        if "use_inside_bar" in lock:
            use_ib = bool(lock["use_inside_bar"])
        if "inside_bar_inclusive" in lock:
            ib_inclusive = bool(lock["inside_bar_inclusive"])

    # Objective creator aligned with WF min_trades and IB lock (no IB search)
    def obj_creator(df, lot_size):
        threshold = int(wf_cfg["min_trades"]) if "min_trades" in wf_cfg else 10

        def objective(trial):
            # Parameter search space (excluding IB toggles; locked by config)
            ema_fast = trial.suggest_int("ema_fast_period", 5, 20)
            ema_medium = trial.suggest_int("ema_medium_period", ema_fast + 3, ema_fast + 25)
            ema_slow = trial.suggest_int("ema_slow_period", ema_medium + 5, ema_medium + 80)

            risk_reward_ratio = trial.suggest_float("risk_reward_ratio", 1.2, 2.0, step=0.1)
            max_avg_deviation_pct = trial.suggest_float("max_avg_deviation_pct", 0.2, 1.0, step=0.05)
            adx_threshold = trial.suggest_float("adx_threshold", 18.0, 28.0, step=1.0)
            atr_stop_multiplier = trial.suggest_float("atr_stop_multiplier", 1.0, 3.0, step=0.1)
            atr_trail_multiplier = trial.suggest_float("atr_trail_multiplier", 0.0, 1.0, step=0.1)
            htf_lookback = trial.suggest_int("htf_lookback", 10, 40)
            min_atr = trial.suggest_float("min_atr", 5.0, 25.0, step=0.5)
            pullback_lookback = trial.suggest_int("pullback_lookback", 6, 15)

            try:
                trades, pnl, _ = backtest_al_brooks_inside_bar(
                    df.copy(),
                    ema_fast_period=ema_fast,
                    ema_medium_period=ema_medium,
                    ema_slow_period=ema_slow,
                    risk_reward_ratio=risk_reward_ratio,
                    max_avg_deviation_pct=max_avg_deviation_pct,
                    lot_size=lot_size,
                    adx_threshold=adx_threshold,
                    atr_stop_multiplier=atr_stop_multiplier,
                    atr_trail_multiplier=atr_trail_multiplier,
                    htf_lookback=htf_lookback,
                    use_inside_bar=use_ib,
                    inside_bar_inclusive=ib_inclusive,
                    min_atr=min_atr,
                    pullback_lookback=pullback_lookback,
                )
            except Exception:
                return -1e9

            m = calculate_metrics(trades)
            trade_count = m.get("total_trades", 0)
            total_pnl = m.get("total_pnl", 0.0)
            pf = m.get("profit_factor", 0.0)

            if trade_count < max(1, threshold):
                return -1e9
            if total_pnl <= 0:
                return -1e9

            if pf == float("inf") or pf == float("nan"):
                pf = 10.0
            return float(pf) + (total_pnl / 200.0)

        return objective

    validator = WalkForwardValidator(
        strategy_name="ALBROOKS",
        symbol=cfg.ticker,
        timeframe=cfg.interval,
        days=cfg.days,
        lot_size=cfg.lot_size,
        min_trades_per_window=int(wf_cfg["min_trades"]),
        objective_func_creator=obj_creator,
        backtest_func=backtest_al_brooks_inside_bar,
        use_cache_only=wf_cfg["cache_only"],
        n_trials=int(wf_cfg.get("trials", 50)),
    )

    validator.run_walk_forward(
        optimization_window=int(wf_cfg["opt_window"]),
        validation_window=int(wf_cfg["val_window"]),
        step_size=int(wf_cfg["step_size"]),
    )

    # Save and move chart to strategy's reports
    # The validator saved the chart at reports/charts/...; move/copy to local folder
    charts_root = Path("reports/charts")
    src_chart = charts_root / f"walk_forward_ALBROOKS_{cfg.ticker}_{cfg.interval}.png"
    local_dir = Path(__file__).resolve().parent / "reports" / "charts"
    local_dir.mkdir(parents=True, exist_ok=True)
    if src_chart.exists():
        dst_chart = local_dir / src_chart.name
        try:
            shutil.copy2(src_chart, dst_chart)
            print(f"WF chart copiado para: {dst_chart}")
        except Exception:
            pass

    # Save JSON summary locally
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary_path = Path(__file__).resolve().parent / "reports" / "snapshots" / f"wf_summary_{ts}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(validator.summary_stats, indent=2), encoding="utf-8")
    print(f"WF summary salvo em: {summary_path}")

    # Save full results for audit (per-period best params and metrics)
    results_path = summary_path.parent / f"wf_results_{ts}.json"
    results_payload = {"results": validator.results, "summary": validator.summary_stats}
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"WF results salvos em: {results_path}")

    # Copy global report JSON if present
    global_report = Path("reports") / "walk_forward" / f"ALBROOKS_{cfg.ticker}_{cfg.interval}_report.json"
    if global_report.exists():
        try:
            dst = summary_path.parent / f"wf_report_{ts}.json"
            __import__("shutil").copy2(global_report, dst)
            print(f"WF report copiado para: {dst}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
