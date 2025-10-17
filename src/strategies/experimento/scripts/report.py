from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_config():
    return json.loads(Path("src/strategies/experimento/config/config_active.json").read_text(encoding="utf-8"))


def get_last_run_id(cx: sqlite3.Connection) -> str | None:
    cur = cx.execute("SELECT run_id FROM runs WHERE finished_at IS NOT NULL ORDER BY finished_at DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row else None


def export_csvs(cx: sqlite3.Connection, run_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = pd.read_sql_query("SELECT * FROM bars WHERE run_id=? ORDER BY idx", cx, params=(run_id,))
    trades = pd.read_sql_query("SELECT * FROM trades WHERE run_id=? ORDER BY trade_id", cx, params=(run_id,))
    metrics = pd.read_sql_query("SELECT * FROM metrics WHERE run_id=?", cx, params=(run_id,))
    bars.to_csv(out_dir / "bars.csv", index=False)
    trades.to_csv(out_dir / "trades.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)


def plot_run(cx: sqlite3.Connection, run_id: str, out_dir: Path) -> None:
    bars = pd.read_sql_query("SELECT * FROM bars WHERE run_id=? ORDER BY idx", cx, params=(run_id,))
    trades = pd.read_sql_query("SELECT * FROM trades WHERE run_id=? ORDER BY trade_id", cx, params=(run_id,))
    if bars.empty:
        return
    bars["close_time"] = pd.to_datetime(bars["close_time"])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(bars["close_time"], bars["close"], label="Close", color="black", linewidth=1)

    # Mark entries/exits
    for _, t in trades.iterrows():
        et = pd.to_datetime(t["entry_time"])
        xt = pd.to_datetime(t["exit_time"])
        ep = float(t["entry_price"]) ; xp = float(t["exit_price"])
        ax.scatter([et], [ep], color="green" if t["side"] == "long" else "red", marker="^", s=40)
        ax.scatter([xt], [xp], color="blue", marker="v", s=40)

    # Equity step curve from trades
    if not trades.empty:
        eq = []
        capital0 = 0.0
        cur = capital0
        exits = trades[["exit_time", "pnl"]].copy()
        exits["exit_time"] = pd.to_datetime(exits["exit_time"]) ; exits = exits.sort_values("exit_time")
        for _, r in exits.iterrows():
            cur += float(r["pnl"])
            eq.append((r["exit_time"], cur))
        if eq:
            times, values = zip(*eq)
            ax2.step(times, values, where="post", label="Equity (PnL cum)")
            ax2.legend()
    ax.set_title(f"Run {run_id}")
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "report.png", dpi=150)
    plt.close(fig)


def print_mc_metrics(cx: sqlite3.Connection, run_id: str) -> None:
    df = pd.read_sql_query("SELECT key, value FROM metrics WHERE run_id=?", cx, params=(run_id,))
    mc = df[df["key"].str.startswith("mc_")]
    if not mc.empty:
        vals = {row["key"]: float(row["value"]) for _, row in mc.iterrows()}
        print("Monte Carlo:", vals)


def main() -> None:
    cfg = load_config()
    db = cfg["storage"]["results_db"]
    out_root = Path(cfg["storage"]["artifacts_dir"]) ; out_root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as cx:
        rid = get_last_run_id(cx)
        if not rid:
            print("No finished runs found.")
            return
        out_dir = out_root / rid
        export_csvs(cx, rid, out_dir)
        plot_run(cx, rid, out_dir)
        print_mc_metrics(cx, rid)
        print(f"Report exported to {out_dir}")


if __name__ == "__main__":
    main()
