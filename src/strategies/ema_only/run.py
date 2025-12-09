from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict

from .backtest import backtest_ema_only
from .optimize import prepare_dataset_with_reference, _build_params_from_config


def main() -> None:
    """
    Executa um backtest único da estratégia EMA-only lendo `config.json`.

    Uso recomendado (offline, consumindo apenas o cache SQLite):
        BINANCE_OFFLINE=1 poetry run python -m src.strategies.ema_only.run
    """
    cfg_path = Path(__file__).with_name("config.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    data_cfg: Dict[str, Any] = cfg.get("data", {})
    strat_cfg: Dict[str, Any] = cfg.get("strategy", {})
    backtest_cfg: Dict[str, Any] = cfg.get("backtest", {})

    symbol = data_cfg["symbol"]
    timeframe = data_cfg["timeframe"]
    days = int(data_cfg.get("days", 365))
    ref_timeframe = data_cfg.get("ref_timeframe")
    ref_days = data_cfg.get("ref_days")
    ref_ema_period = strat_cfg.get("ref_ema_period")

    monthly_target_pct = float(backtest_cfg.get("monthly_target_pct", 0.01))
    initial_capital = float(backtest_cfg.get("initial_capital", 1_000.0))

    df = prepare_dataset_with_reference(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        use_cache_only=True,
        ref_timeframe=ref_timeframe,
        ref_days=ref_days,
        ref_ema_period=ref_ema_period,
    )

    params = _build_params_from_config(strat_cfg)

    trades, total_pnl, stats = backtest_ema_only(
        df,
        params=params,
        initial_capital=initial_capital,
        monthly_target_pct=monthly_target_pct,
    )

    # Resumo didático com meta mensal explícita (para auditoria).
    avg_m = stats.get("avg_monthly_return_pct", 0.0)
    hit_ratio = stats.get("monthly_target_hit_ratio", 0.0)
    total_ret = stats.get("total_return_pct", 0.0)
    max_dd = stats.get("max_drawdown_pct", 0.0)
    win_rate = stats.get("win_rate", 0.0)

    period_start = df["Date"].iloc[0]
    period_end = df["Date"].iloc[-1]

    print(f"[ema_only.run] Símbolo: {symbol}  Timeframe: {timeframe}  Dias: {days}")
    print(
        f"[ema_only.run] Período: {period_start.isoformat()} -> "
        f"{period_end.isoformat()}"
    )
    print(f"[ema_only.run] Objetivo mensal: {monthly_target_pct:.2%}")
    print(
        f"[ema_only.run] Retorno médio mensal: {avg_m:.2%}  | "
        f"Meses >= objetivo: {hit_ratio:.1%}"
    )
    print(
        f"[ema_only.run] Retorno total: {total_ret:.2%}  | "
        f"PnL: {total_pnl:.2f}  | Max DD: {max_dd:.2%}"
    )
    print(
        f"[ema_only.run] Trades: {stats.get('num_trades', 0)}  | "
        f"Win rate: {win_rate:.1%}  | Sharpe: {stats.get('sharpe', 0.0):.3f}"
    )

    # Tabela mensal detalhada (para auditoria).
    monthly = stats.get("monthly_breakdown", {}) or {}
    if monthly:
        print("\n[ema_only.run] Backtest mensal — resumo:")
        print("Mês     | StartEq  | EndEq    |   PnL   | Ret%")
        print("--------+----------+----------+---------+--------")
        for month in sorted(monthly.keys()):
            row = monthly[month]
            se = row["start_equity"]
            ee = row["end_equity"]
            pnl_m = row["pnl"]
            ret_m = row["return_pct"]
            print(
                f"{month} | "
                f"{se:8.2f} | "
                f"{ee:8.2f} | "
                f"{pnl_m:7.2f} | "
                f"{ret_m*100:6.2f}"
            )

    # Marca significância estatística mínima (para evidenciar quando o experimento é frágil).
    min_trades = int(backtest_cfg.get("min_trades_for_significance", 30))
    min_candles = int(backtest_cfg.get("min_candles_for_significance", 2000))
    sig_flags: Dict[str, Any] = {
        "is_significant": (
            stats.get("num_trades", 0) >= min_trades and len(df) >= min_candles
        ),
        "num_trades": int(stats.get("num_trades", 0)),
        "num_candles": len(df),
        "min_trades_for_significance": min_trades,
        "min_candles_for_significance": min_candles,
    }
    if not sig_flags["is_significant"]:
        print(
            "[ema_only.run] AVISO: amostra com baixa significância "
            f"(trades={sig_flags['num_trades']}, candles={sig_flags['num_candles']})."
        )

    # Persistência em JSON (orientado à auditoria).
    outdir = Path(backtest_cfg.get("outdir", "src/strategies/ema_only/reports/backtest"))
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    period = {
        "start": df["Date"].iloc[0].isoformat(),
        "end": df["Date"].iloc[-1].isoformat(),
    }

    result = {
        "strategy": "ema_only",
        "symbol": symbol,
        "interval": timeframe,
        "period": period,
        "initial_capital": initial_capital,
        "objective": {
            "monthly_target_pct": monthly_target_pct,
        },
        "stats": stats,
        "significance": sig_flags,
        "trades": len(trades),
        "total_pnl": total_pnl,
        "config_path": str(cfg_path),
        "run_timestamp": datetime.now(UTC).isoformat(),
    }

    json_path = outdir / f"ema_only_backtest_{symbol}_{timeframe}_{ts}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ema_only.run] Resultado salvo em: {json_path}")


if __name__ == "__main__":
    main()
