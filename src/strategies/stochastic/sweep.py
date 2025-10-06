from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path
import json

import optuna

from .optimize import load_data, make_objective
from .backtest import backtest_stochastic, StochParams
from .config import save_active_config


def run_for_interval(ticker: str, interval: str, days: int, train_frac: float, lot_size: float, fee_rate: float,
                     trials: int, seed: int) -> dict:
    df = load_data(ticker, interval, days)
    n = len(df)
    split = int(n * train_frac)
    df_train = df.iloc[:split].copy()
    df_valid = df.iloc[split:].copy()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(make_objective(df_train, lot_size, fee_rate), n_trials=trials)

    p = study.best_params

    # Avaliação treino/validação com os melhores parâmetros
    params = StochParams(
        k_period=p["k"],
        oversold=p["oversold"],
        overbought=p["overbought"],
        d_period=p.get("d_period", 3),
        use_kd_cross=p.get("use_kd_cross", True),
        ema_period=(p.get("ema_period") or None) if p.get("enable_trend", False) else None,
        confirm_bars=p.get("confirm_bars", 0),
        cooldown_bars=p.get("cooldown_bars", 0),
        min_hold_bars=p.get("min_hold_bars", 0),
    )
    tr_trades, tr_pnl, tr_stats = backtest_stochastic(df_train.copy(), params=params, lot_size=lot_size, fee_rate=fee_rate)
    va_trades, va_pnl, va_stats = backtest_stochastic(df_valid.copy(), params=params, lot_size=lot_size, fee_rate=fee_rate)

    rec = {
        "ticker": ticker,
        "interval": interval,
        "days": days,
        "train_frac": train_frac,
        "lot_size": lot_size,
        "fee_rate": fee_rate,
        "best_params": {
            "k_period": p["k"],
            "oversold": p["oversold"],
            "overbought": p["overbought"],
            "d_period": p.get("d_period", 3),
            "use_kd_cross": p.get("use_kd_cross", True),
            "ema_period": (p.get("ema_period") if p.get("enable_trend", False) else None),
            "confirm_bars": p.get("confirm_bars", 0),
            "cooldown_bars": p.get("cooldown_bars", 0),
            "min_hold_bars": p.get("min_hold_bars", 0),
            "lot_size": lot_size,
        },
        "train_metrics": tr_stats,
        "valid_metrics": va_stats,
        "best_value": float(study.best_value),
        "best_params_raw": p,
    }

    # Salva config ativa por intervalo
    save_active_config(rec)
    return rec


def main():
    ap = argparse.ArgumentParser(description="Sweep optimize Stochastic across intervals")
    ap.add_argument("--ticker", default="BTCUSDT")
    ap.add_argument("--intervals", default="5m,15m,1h", help="Lista separada por vírgulas")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--lot_size", type=float, default=0.001)
    ap.add_argument("--fee_rate", type=float, default=0.001)
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    results = []
    for itv in intervals:
        print(f"\n=== Otimizando {args.ticker} @ {itv} por {args.days} dias, {args.trials} trials ===")
        rec = run_for_interval(
            ticker=args.ticker,
            interval=itv,
            days=args.days,
            train_frac=args.train_frac,
            lot_size=args.lot_size,
            fee_rate=args.fee_rate,
            trials=args.trials,
            seed=args.seed,
        )
        results.append(rec)
        print(f"Intervalo {itv}: valid P&L ${rec['valid_metrics']['pnl']:.2f}, PF {rec['valid_metrics'].get('profit_factor', 0):.2f}")

    # Escolhe melhor por P&L de validação (pós taxas)
    best = max(results, key=lambda r: r["valid_metrics"]["pnl"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = f"stoch_sweep_{args.ticker}_{ts}"

    # JSON
    summary = {
        "ticker": args.ticker,
        "fee_rate": args.fee_rate,
        "lot_size": args.lot_size,
        "days": args.days,
        "train_frac": args.train_frac,
        "intervals": intervals,
        "results": results,
        "best": {"interval": best["interval"], "valid_metrics": best["valid_metrics"], "params": best["best_params"]},
    }
    (outdir / f"{base}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown
    md = ["# Stochastic Interval Sweep", "", f"- Ticker: {args.ticker}", f"- Fee rate: {args.fee_rate}", f"- Lot size: {args.lot_size}", f"- Days: {args.days}", f"- Train frac: {args.train_frac}", "", "## Results"]
    for r in results:
        vm = r["valid_metrics"]
        md.append(f"- {r['interval']}: PnL ${vm['pnl']:.2f}, PF {vm.get('profit_factor', 0):.2f}, trades {vm.get('num_trades', 0)}")
    md += ["", f"## Best Interval: {best['interval']}", f"- Valid PnL: ${best['valid_metrics']['pnl']:.2f}", f"- Profit Factor: {best['valid_metrics'].get('profit_factor', 0):.2f}"]
    (outdir / f"{base}.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nResumo salvo em {outdir / (base + '.md')} e {outdir / (base + '.json')}")
    print(f"Melhor intervalo: {best['interval']} | PnL val: ${best['valid_metrics']['pnl']:.2f}")


if __name__ == "__main__":
    main()

