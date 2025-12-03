from __future__ import annotations

"""
Backtest mensal do agente RL EMA-only usando o modelo PPO treinado.

Uso:

  BINANCE_OFFLINE=1 poetry run python -m src.strategies.ema_only.rl_backtest

Lê `config.json`, carrega o modelo salvo em
`src/strategies/ema_only/reports/rl/ppo_ema_only.zip` e roda um backtest
determinístico no período de validação definido em `rl.train`
(`end` -> `val_end`), calculando o PnL por mês.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .rl_train import load_cfg, make_env_from_cfg


OUTDIR = Path("src/strategies/ema_only/reports/rl")
MODEL_PATH = OUTDIR / "ppo_ema_only.zip"


def run_monthly_backtest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rl_cfg = cfg.get("rl", {})
    train_cfg = rl_cfg.get("train", {})
    data_cfg = cfg.get("data", {})

    symbol = str(data_cfg.get("symbol", "BTCUSDT"))
    timeframe = str(data_cfg.get("timeframe", "4h"))
    start = str(train_cfg.get("end", "2025-01-01 00:00:00"))
    end = str(train_cfg.get("val_end", "2025-12-01 00:00:00"))

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo PPO não encontrado em {MODEL_PATH}. Treine primeiro com train.py.")

    env, _, _ = make_env_from_cfg(cfg, start=start, end=end)
    model = PPO.load(MODEL_PATH, env=None)

    obs, _ = env.reset()
    records: List[Dict[str, Any]] = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = env.step(int(action))
        idx = max(0, env.idx - 1)
        row = env.df.iloc[idx]
        records.append(
            {
                "Date": row["Date"],
                "equity": float(getattr(env, "last_equity", env.equity)),
                "action": int(action),
            }
        )

    if not records:
        raise RuntimeError("Backtest RL não gerou registros. Verifique o período e os dados.")

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["month"] = df["Date"].dt.to_period("M")

    months = sorted(df["month"].unique())
    results: List[Dict[str, Any]] = []
    init_equity = float(getattr(env.cfg, "init_equity", 1000.0))
    prev_end_equity = init_equity

    for m in months:
        g = df[df["month"] == m]
        if g.empty:
            continue
        end_equity = float(g["equity"].iloc[-1])
        pnl = end_equity - prev_end_equity
        ret_pct = (pnl / prev_end_equity * 100.0) if prev_end_equity != 0 else 0.0
        results.append(
            {
                "month": str(m),
                "start_equity": round(prev_end_equity, 4),
                "end_equity": round(end_equity, 4),
                "pnl": round(pnl, 4),
                "return_pct": round(ret_pct, 4),
            }
        )
        prev_end_equity = end_equity

    payload: Dict[str, Any] = {
        "strategy": "ema_only_rl",
        "symbol": symbol,
        "timeframe": timeframe,
        "start": df["Date"].iloc[0].isoformat(),
        "end": df["Date"].iloc[-1].isoformat(),
        "init_equity": init_equity,
        "months": results,
    }
    return payload


def main() -> None:
    cfg = load_cfg()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    result = run_monthly_backtest(cfg)

    # Salva JSON com o resumo mensal
    out_path = OUTDIR / "monthly_pnl_ema_only_rl.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Imprime tabela simples no console
    print(f"Backtest mensal RL — {result['symbol']} {result['timeframe']}")
    print(f"Período: {result['start']} -> {result['end']}")
    print(f"Equity inicial: {result['init_equity']:.2f}")
    print("Mês     | StartEq | EndEq  | PnL   | Ret%")
    print("----------------------------------------------")
    for row in result["months"]:
        print(
            f"{row['month']} | "
            f"{row['start_equity']:7.2f} | "
            f"{row['end_equity']:7.2f} | "
            f"{row['pnl']:6.2f} | "
            f"{row['return_pct']:5.2f}"
        )
    print(f"\nResumo salvo em {out_path}")


if __name__ == "__main__":
    main()

