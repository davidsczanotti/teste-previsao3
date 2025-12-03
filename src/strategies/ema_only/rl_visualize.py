from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .rl_train import load_cfg, make_env_from_cfg


def _load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def plot_metrics(metrics_df: pd.DataFrame, outdir: Path):
    if metrics_df.empty:
        print("No metrics to plot.")
        return
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    metrics_df.plot(x="step", y="reward_mean", ax=ax[0], title="Reward (mean)")
    metrics_df.plot(x="step", y="pnl", ax=ax[1], title="PnL")
    metrics_df.plot(x="step", y="trades", ax=ax[2], title="Trades")
    ax[2].set_xlabel("Step")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "metrics.png", dpi=150)
    plt.close(fig)


def plot_actions(df: pd.DataFrame, outdir: Path):
    if df.empty:
        print("No rollout actions to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["close"], label="close", alpha=0.6)
    ax.plot(df["Date"], df["ema_fast"], label="ema_fast", alpha=0.8)
    ax.plot(df["Date"], df["ema_slow"], label="ema_slow", alpha=0.8)
    ax.plot(df["Date"], df["ref_ema"], label="ref_ema", alpha=0.8)
    # Ações: 1 = alvo long, 2 = alvo short (0=hold, 3=flat)
    longs = df[df["action"] == 1]
    shorts = df[df["action"] == 2]
    ax.scatter(longs["Date"], longs["close"], marker="^", color="green", label="long", s=20)
    ax.scatter(shorts["Date"], shorts["close"], marker="v", color="red", label="short", s=20)
    ax.legend()
    ax.set_title("Ações do agente vs preço/EMAs")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "actions.png", dpi=150)
    plt.close(fig)


def main():
    cfg = load_cfg()
    rl_cfg = cfg.get("rl", {})
    train_cfg = rl_cfg.get("train", {})
    # Métricas de treino (se existirem)
    metrics_path = Path("src/strategies/ema_only/reports/rl/metrics.csv")
    metrics_df = _load_metrics(metrics_path)
    plot_metrics(metrics_df, Path("src/strategies/ema_only/reports/rl"))

    # Rollout de validação curto para gerar actions.png
    start = train_cfg.get("end", "2025-01-01 00:00:00")
    val_end = train_cfg.get("val_end", "2025-12-01 00:00:00")
    val_env, _, _ = make_env_from_cfg(cfg, start, val_end)

    model_path = Path("src/strategies/ema_only/reports/rl/ppo_ema_only.zip")
    actions = []
    if model_path.exists():
        # Usa o modelo treinado para gerar as ações
        model = PPO.load(model_path, env=None)
        obs, _ = val_env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = val_env.step(int(action))
            idx = max(0, val_env.idx - 1)
            base_row = val_env.df.iloc[idx].to_dict()
            feat_row = val_env.features.iloc[idx].to_dict() if hasattr(val_env, "features") else {}
            base_row.update(
                {
                    "ema_fast": feat_row.get("ema_fast"),
                    "ema_slow": feat_row.get("ema_slow"),
                    "ref_ema": feat_row.get("ref_ema"),
                    "action": int(action),
                }
            )
            actions.append(base_row)
    else:
        # Fallback: política aleatória apenas para gerar estrutura
        obs, _ = val_env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = val_env.action_space.sample()
            obs, reward, terminated, truncated, _ = val_env.step(int(action))
            idx = max(0, val_env.idx - 1)
            base_row = val_env.df.iloc[idx].to_dict()
            feat_row = val_env.features.iloc[idx].to_dict() if hasattr(val_env, "features") else {}
            base_row.update(
                {
                    "ema_fast": feat_row.get("ema_fast"),
                    "ema_slow": feat_row.get("ema_slow"),
                    "ref_ema": feat_row.get("ref_ema"),
                    "action": int(action),
                }
            )
            actions.append(base_row)

    actions_df = pd.DataFrame(actions)
    if not actions_df.empty:
        actions_df["Date"] = pd.to_datetime(actions_df["Date"])
    plot_actions(actions_df, Path("src/strategies/ema_only/reports/rl"))


if __name__ == "__main__":
    main()
