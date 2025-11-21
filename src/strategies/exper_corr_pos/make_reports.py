from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CFG_PATH = Path("src/strategies/exper_corr_pos/config.json")


def plot_metrics(outdir: Path, metrics_path: Path) -> None:
    dfm = pd.read_csv(metrics_path)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(dfm["episode"], dfm["policy_loss"], label="policy_loss")
    ax.plot(dfm["episode"], dfm["value_loss"], label="value_loss")
    ax.legend(); ax.set_title("Losses")

    ax = axes[0, 1]
    if "entropy" in dfm.columns:
        ax.plot(dfm["episode"], dfm["entropy"], label="entropy")
    if "entropy_coef" in dfm.columns:
        ax.plot(dfm["episode"], dfm["entropy_coef"], label="entropy_coef")
    if "load_balance" in dfm.columns:
        ax.plot(dfm["episode"], dfm["load_balance"], label="load_balance")
    ax.legend(); ax.set_title("Entropy / Balance")

    ax = axes[1, 0]
    if "avg_reward" in dfm.columns:
        ax.plot(dfm["episode"], dfm["avg_reward"], label="avg_reward")
    if "sum_reward" in dfm.columns:
        ax.plot(dfm["episode"], dfm["sum_reward"], label="sum_reward")
    ax.legend(); ax.set_title("Rewards")

    ax = axes[1, 1]
    if "greedy_equity" in dfm.columns:
        ge = pd.to_numeric(dfm["greedy_equity"], errors="coerce")
        mask = ge.notna()
        if mask.any():
            ax.plot(dfm["episode"][mask], ge[mask], label="greedy_equity", marker="o", markersize=3)
            y_min, y_max = float(ge[mask].min()), float(ge[mask].max())
            margin = max(1.0, (y_max - y_min) * 0.1)
            ax.set_ylim(y_min - margin, y_max + margin)
            if "greedy_ruined" in dfm.columns:
                ruined_mask = pd.to_numeric(dfm["greedy_ruined"], errors="coerce").fillna(0).astype(bool)
                ruined_mask &= mask
                if ruined_mask.any():
                    ax.scatter(dfm["episode"][ruined_mask], ge[ruined_mask], color="#ff6666", marker="x", s=30, label="ruína")
    ax.legend(); ax.set_title("Greedy equity (eval)")

    fig.tight_layout()
    fig.savefig(outdir / "metrics.png", dpi=120)
    plt.close(fig)


def plot_usage(outdir: Path, metrics_path: Path, num_experts: int, window: int) -> None:
    dfm = pd.read_csv(metrics_path)
    cols = [f"usage_e{i}" for i in range(num_experts)]
    cols = [c for c in cols if c in dfm.columns]
    if not cols:
        return
    tail = dfm.tail(window)
    means = tail[cols].mean(numeric_only=True)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(cols)), means.values, color="#4da6ff")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45)
    ax.set_ylim(0, 1)
    last_episode = int(tail["episode"].iloc[-1]) if "episode" in tail.columns and not tail.empty else 0
    ax.set_title(
        f"Expert usage (média das últimas {len(tail)} execuções — até ep {last_episode})"
    )
    fig.tight_layout()
    fig.savefig(outdir / "expert_usage.png", dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera métricas/plots consolidados")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Caminho alternativo para metrics.csv (ex.: quickrun/metrics.csv)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Diretório onde os gráficos serão salvos (default: train.outdir do config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(CFG_PATH.read_text())
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    default_outdir = Path(train_cfg.get("outdir", "src/strategies/exper_corr_pos/reports/train"))
    metrics_path = Path(args.metrics) if args.metrics else default_outdir / "metrics.csv"
    outdir = Path(args.outdir) if args.outdir else (metrics_path.parent if args.metrics else default_outdir)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {metrics_path}")

    plot_metrics(outdir, metrics_path)
    usage_window = int(train_cfg.get("usage_window", train_cfg.get("plot_every", 100)))
    plot_usage(outdir, metrics_path, int(model_cfg.get("num_experts", 4)), usage_window)
    print(f"Relatórios atualizados em {outdir}")


if __name__ == "__main__":
    main()
