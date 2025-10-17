from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd


def load_config() -> Dict[str, Any]:
    return json.loads(Path("src/strategies/experimento/config/config_active.json").read_text(encoding="utf-8"))


def latest_wfo_dir(artifacts_root: Path) -> Path | None:
    if not artifacts_root.exists():
        return None
    dirs = [p for p in artifacts_root.iterdir() if p.is_dir() and p.name.startswith("wfo-")]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def plot_equity(equity_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(equity_csv)
    if df.empty:
        return
    df["time"] = pd.to_datetime(df["time"])  # ISO
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(df["time"], df["equity"], where="post", label="Equity (OOS)")
    ax.legend()
    ax.set_title("WFO Equity (OOS)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_windows(summary_json: Path, out_path: Path) -> None:
    obj = json.loads(summary_json.read_text(encoding="utf-8"))
    wins = obj.get("windows", [])
    if not wins:
        return
    idx = [int(w["window"]) for w in wins]
    pf = [float(w["profit_factor"]) for w in wins]
    trades = [int(w["trades"]) for w in wins]
    pnl = [float(w["pnl_total"]) for w in wins]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(idx, pf, color="#3b82f6", alpha=0.7, label="PF")
    ax.set_xlabel("Window")
    ax.set_ylabel("Profit Factor")
    ax2 = ax.twinx()
    ax2.plot(idx, trades, color="#ef4444", marker="o", label="Trades")
    ax2.set_ylabel("Trades")
    ax.set_title("WFO Windows — PF and Trades")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    root = Path(cfg["storage"]["artifacts_dir"]) ; root.mkdir(parents=True, exist_ok=True)
    wfo_dir = latest_wfo_dir(root)
    if not wfo_dir:
        print("No WFO artifacts found.")
        return
    summary_json = wfo_dir / "wfo_summary.json"
    equity_csv = wfo_dir / "equity_curve.csv"
    plot_equity(equity_csv, wfo_dir / "wfo_equity.png")
    plot_windows(summary_json, wfo_dir / "wfo_windows.png")
    print(f"WFO report exported to {wfo_dir}")


if __name__ == "__main__":
    main()

