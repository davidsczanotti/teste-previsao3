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
from .optimize import make_objective
from .backtest import backtest_al_brooks_inside_bar
from .config import load_active_config
from ...utils.data_loader import load_data


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

    # Objective creator aligned with WF min_trades
    def obj_creator(df, lot_size):
        return make_objective(df, lot_size, min_trade_threshold=int(wf_cfg["min_trades"]))

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
    ts = __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M%S")
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
