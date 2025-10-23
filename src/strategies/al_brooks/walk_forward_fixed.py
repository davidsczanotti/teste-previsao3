from __future__ import annotations

"""
Walk-Forward (anchored) runner for Al Brooks strategy — parameters fixed after the
first optimization window, then validated across subsequent windows without re-optimizing.

Outputs:
- Chart per-period P&L and Win Rate saved under strategy reports
- JSON summary and per-period results with best params and metrics
"""

from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, UTC

import json
import optuna
import numpy as np
import matplotlib.pyplot as plt

from ...utils.data_loader import load_data
from ...utils.metrics import calculate_metrics
from ...utils.walk_forward import WalkForwardValidator
from .config import load_active_config
from .backtest import backtest_al_brooks_inside_bar


def _make_objective_locked_ib(df, lot_size: float, min_trades: int, use_ib: bool, ib_inclusive: bool):
    def objective(trial):
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
            trades, _, _ = backtest_al_brooks_inside_bar(
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

        if trade_count < max(1, int(min_trades)):
            return -1e9
        if total_pnl <= 0:
            return -1e9

        if not np.isfinite(pf):
            pf = 10.0
        return float(pf) + (total_pnl / 200.0)

    return objective


def main() -> None:
    cfg = load_active_config("BTCUSDT", "1h")
    if not cfg:
        print("Nenhuma configuração encontrada em config.json.")
        return

    # Read WF config (optional)
    cfg_path = Path(__file__).resolve().parent / "config.json"
    wf = {}
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        wf = dict(raw.get("walk_forward", {}))
    except Exception:
        wf = {}

    opt_window = int(wf.get("opt_window", 180))
    val_window = int(wf.get("val_window", 45))
    step_size = int(wf.get("step_size", 30))
    min_trades = int(wf.get("min_trades", 6))
    trials = int(wf.get("trials", 120))
    cache_only = bool(wf.get("cache_only", True))
    lock = dict(wf.get("lock", {})) if isinstance(wf.get("lock"), dict) else {}

    use_ib = lock.get("use_inside_bar", getattr(cfg, "use_inside_bar", True))
    ib_inclusive = lock.get("inside_bar_inclusive", getattr(cfg, "inside_bar_inclusive", False))

    # Load data and build periods using helper
    data = load_data(cfg.ticker, cfg.interval, days=cfg.days, use_cache_only=cache_only)
    validator = WalkForwardValidator(
        strategy_name="ALBROOKS",
        symbol=cfg.ticker,
        timeframe=cfg.interval,
        days=cfg.days,
        lot_size=cfg.lot_size,
        min_trades_per_window=min_trades,
        objective_func_creator=lambda *_: None,  # not used here
        backtest_func=backtest_al_brooks_inside_bar,
        use_cache_only=cache_only,
        n_trials=trials,
    )
    periods = validator.create_periods(data, opt_window, val_window, step_size)
    if not periods:
        print("Sem períodos suficientes para WF-ancorado.")
        return

    # Optimize on the first optimization slice only
    first = periods[0]
    opt_data = data.iloc[first["optimization_start"] : first["optimization_end"]].reset_index(drop=True)
    study = optuna.create_study(direction="maximize")
    objective = _make_objective_locked_ib(opt_data, cfg.lot_size, min_trades, use_ib, ib_inclusive)
    study.optimize(objective, n_trials=trials, show_progress_bar=False, gc_after_trial=True)
    best_params = study.best_params
    best_score = study.best_value
    print(f"Anchored optimization: best_score={best_score:.4f}, params={best_params}")

    # Validate across all periods with fixed params
    results: List[Dict[str, Any]] = []
    for per in periods:
        val_data = data.iloc[per["validation_start"] : per["validation_end"]].reset_index(drop=True)
        trades, total_pnl, _ = backtest_al_brooks_inside_bar(
            val_data.copy(), lot_size=cfg.lot_size, use_inside_bar=use_ib, inside_bar_inclusive=ib_inclusive, **best_params
        )
        m = calculate_metrics(trades)
        results.append(
            {
                "period": per,
                "validation_pnl": m.get("total_pnl", 0.0),
                "validation_trades": m.get("total_trades", 0),
                "validation_win_rate": m.get("win_rate", 0.0),
                "validation_profit_factor": m.get("profit_factor", 0.0),
            }
        )

    # Aggregate and save
    pnls = [r["validation_pnl"] for r in results]
    win_rates = [r["validation_win_rate"] for r in results]
    profits = [max(0.0, p) for p in pnls]
    losses = [abs(min(0.0, p)) for p in pnls]
    agg_pf = float(sum(profits) / sum(losses)) if sum(losses) > 0 else float("inf")
    summary = {
        "total_periods": len(results),
        "total_pnl": float(sum(pnls)),
        "avg_pnl": float(np.mean(pnls) if pnls else 0.0),
        "avg_win_rate": float(np.mean(win_rates) if win_rates else 0.0),
        "aggregate_profit_factor": agg_pf,
    }

    # Chart
    charts_dir = Path(__file__).resolve().parent / "reports" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"WF-Fixed Performance: ALBROOKS {cfg.ticker} ({cfg.interval})", fontsize=16)
    colors = ["#2ca02c" if p > 0 else "#d62728" for p in pnls]
    ax1.bar(range(1, len(pnls) + 1), pnls, color=colors)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("P&L ($)")
    ax1.set_title("P&L per Validation Period")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax2.plot(range(1, len(win_rates) + 1), win_rates, "o-", color="#1f77b4")
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Period Number")
    ax2.set_ylabel("Win Rate")
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    chart_path = charts_dir / f"walk_forward_fixed_ALBROOKS_{cfg.ticker}_{cfg.interval}_{ts}.png"
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"WF-Fixed chart salvo em: {chart_path}")

    # Save JSONs
    snaps_dir = Path(__file__).resolve().parent / "reports" / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    (snaps_dir / f"wf_fixed_summary_{ts}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    payload = {"best_params": best_params, "best_score": best_score, "results": results, "summary": summary}
    (snaps_dir / f"wf_fixed_results_{ts}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WF-Fixed summary/results salvos em: {snaps_dir}")


if __name__ == "__main__":
    main()

