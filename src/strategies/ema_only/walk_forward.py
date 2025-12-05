from __future__ import annotations

"""Runner de walk-forward com amostragem por regime opcional."""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .rl_backtest import OUTDIR as RL_OUTDIR
from .rl_env import EmaEnv
from .rl_train import load_cfg, make_env_from_cfg, train_from_config


def iter_windows(start: pd.Timestamp, end: pd.Timestamp, train_months: int, val_months: int, step_months: int):
    cur_start = start
    delta_train = pd.DateOffset(months=train_months)
    delta_val = pd.DateOffset(months=val_months)
    delta_step = pd.DateOffset(months=step_months)
    while True:
        train_start = cur_start
        train_end = train_start + delta_train
        val_end = train_end + delta_val
        if val_end > end:
            break
        yield train_start, train_end, val_end
        cur_start = cur_start + delta_step


def run_eval(env: EmaEnv, model_path: Path) -> Dict[str, Any]:
    """Roda backtest mensal em um env já configurado, usando modelo PPO salvo."""

    model = PPO.load(model_path, env=None)
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
        "start": df["Date"].iloc[0].isoformat() if not df.empty else "",
        "end": df["Date"].iloc[-1].isoformat() if not df.empty else "",
        "init_equity": init_equity,
        "months": results,
    }
    return payload


def run_walk_forward(cfg: Dict[str, Any]) -> None:
    wf_cfg = cfg.get("walk_forward", {})
    train_months = int(wf_cfg.get("train_months", 18))
    val_months = int(wf_cfg.get("val_months", 6))
    step_months = int(wf_cfg.get("step_months", 6))
    rs_override = wf_cfg.get("regime_sampling", {}) or {}

    rl_train_cfg = cfg.get("rl", {}).get("train", {})
    start_all = pd.to_datetime(rl_train_cfg.get("start", "2017-01-01 00:00:00"))
    end_all = pd.to_datetime(rl_train_cfg.get("val_end", rl_train_cfg.get("end", "2025-01-01 00:00:00")))

    outdir = Path(RL_OUTDIR) / "walk_forward"
    outdir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []

    for i, (tr_start, tr_end, val_end) in enumerate(iter_windows(start_all, end_all, train_months, val_months, step_months)):
        run_id = f"wf_{i:02d}"
        run_dir = Path(RL_OUTDIR) / "walk_forward" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[WF] Janela {run_id}: train {tr_start} -> {tr_end}, val {tr_end} -> {val_end}")

        # Treino com sampling opcional
        train_from_config(
            cfg,
            start_override=str(tr_start),
            end_override=str(tr_end),
            use_regime_sampling=rs_override.get("enabled", False),
            run_name=f"walk_forward/{run_id}",
        )

        model_path = run_dir / "ppo_ema_only.zip"
        env, _, _ = make_env_from_cfg(
            cfg,
            start=str(tr_end),
            end=str(val_end),
            use_regime_sampling=False,
        )
        result = run_eval(env, model_path)
        result.update({"run_id": run_id, "train_start": str(tr_start), "train_end": str(tr_end), "val_end": str(val_end)})

        out_path = outdir / f"{run_id}_summary.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        summary.append(result)

    summary_path = outdir / "wf_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Walk-forward concluído. Resumo em {summary_path}")


def main() -> None:
    cfg = load_cfg()
    run_walk_forward(cfg)


if __name__ == "__main__":
    main()
