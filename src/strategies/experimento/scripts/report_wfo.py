from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import mplfinance as mpf


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


def per_window_reports(cfg: Dict[str, Any], wfo_dir: Path) -> None:
    # Generate per-window equity and candlestick charts
    summary = json.loads((wfo_dir / "wfo_summary.json").read_text(encoding="utf-8"))
    windows = summary.get("windows", [])
    if not windows:
        return
    db = cfg["storage"]["results_db"]
    capital0 = float(cfg["risk"]["capital"]) if windows else 0.0
    normalize = bool(cfg.get("report", {}).get("normalize_equity", True))
    show_dd = bool(cfg.get("report", {}).get("show_drawdown", True))
    windows_dir = wfo_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db) as cx:
        for w in windows:
            wi = int(w["window"]) ; rid = w["run_id"]
            # Equity per window
            tr = pd.read_sql_query("SELECT exit_time, pnl FROM trades WHERE run_id=? ORDER BY exit_time", cx, params=(rid,))
            eq = []
            cur = capital0
            for _, r in tr.iterrows():
                cur += float(r["pnl"]) ; eq.append((pd.to_datetime(r["exit_time"]), cur))
            if eq:
                df_eq = pd.DataFrame(eq, columns=["time", "equity"]).sort_values("time")
                if normalize and capital0 > 0:
                    df_eq["equity_norm"] = df_eq["equity"] / capital0
                # Drawdown
                if show_dd:
                    peak = df_eq["equity"].cummax()
                    dd = (df_eq["equity"] - peak) / peak.replace({0: float('nan')})
                    df_eq["drawdown"] = dd.fillna(0.0)
                out_csv = windows_dir / f"window_{wi:02d}_equity.csv"
                df_eq.to_csv(out_csv, index=False)
                # Plot equity + drawdown
                fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
                ax.step(df_eq["time"], df_eq["equity_norm"] if normalize and capital0 > 0 else df_eq["equity"], where="post", label="Equity")
                ax.legend() ; ax.set_title(f"Window {wi} — Equity")
                if show_dd and "drawdown" in df_eq:
                    ax2.fill_between(df_eq["time"], df_eq["drawdown"], 0, color="#ef4444", alpha=0.3, step="post")
                    ax2.set_title("Drawdown")
                fig.tight_layout()
                fig.savefig(windows_dir / f"window_{wi:02d}_equity.png", dpi=150)
                plt.close(fig)

            # Candlestick with EMAs and trade markers
            bars = pd.read_sql_query("SELECT close_time, open, high, low, close, volume, ema_fast_30m, ema_slow_30m FROM bars WHERE run_id=? ORDER BY idx", cx, params=(rid,))
            if not bars.empty:
                bars["Date"] = pd.to_datetime(bars["close_time"]) ; bars = bars.set_index("Date")
                ohlc = bars[["open", "high", "low", "close", "volume"]].copy()
                apds = [
                    mpf.make_addplot(bars["ema_fast_30m"], color="#3b82f6"),
                    mpf.make_addplot(bars["ema_slow_30m"], color="#f59e0b"),
                ]
                # Trade markers
                tr2 = pd.read_sql_query("SELECT entry_time, exit_time, side, entry_price, exit_price FROM trades WHERE run_id=? ORDER BY trade_id", cx, params=(rid,))
                if not tr2.empty:
                    tr2["entry_time"] = pd.to_datetime(tr2["entry_time"]) ; tr2["exit_time"] = pd.to_datetime(tr2["exit_time"])
                    # Entry markers
                    em = pd.Series(index=ohlc.index, dtype=float)
                    xm = pd.Series(index=ohlc.index, dtype=float)
                    for _, t in tr2.iterrows():
                        et = t["entry_time"] ; xt = t["exit_time"]
                        if et in em.index:
                            em.loc[et] = float(t["entry_price"])
                        if xt in xm.index:
                            xm.loc[xt] = float(t["exit_price"])
                    apds.append(mpf.make_addplot(em, scatter=True, markersize=50, marker='^', color='green'))
                    apds.append(mpf.make_addplot(xm, scatter=True, markersize=50, marker='v', color='red'))
                mpf.plot(ohlc, type='candle', volume=True, addplot=apds, style='yahoo', savefig=str(windows_dir / f"window_{wi:02d}_candles.png"))



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
    # New: per-window equity + candles
    per_window_reports(cfg, wfo_dir)
    print(f"WFO report exported to {wfo_dir}")


if __name__ == "__main__":
    main()
