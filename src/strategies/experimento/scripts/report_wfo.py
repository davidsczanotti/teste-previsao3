from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
# Using line plots instead of candles for visibility


def load_config() -> Dict[str, Any]:
    return json.loads(Path("src/strategies/experimento/config/config_active.json").read_text(encoding="utf-8"))


def latest_wfo_dir(artifacts_root: Path) -> Path | None:
    if not artifacts_root.exists():
        return None
    dirs = [p for p in artifacts_root.iterdir() if p.is_dir() and p.name.startswith("wfo-")]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def latest_wfo_group_from_db(db_path: str) -> str | None:
    with sqlite3.connect(db_path) as cx:
        df = pd.read_sql_query("SELECT DISTINCT value as grp FROM params WHERE key='wfo_group' ORDER BY grp", cx)
        if df.empty:
            return None
        return str(df["grp"].iloc[-1])


def build_wfo_artifacts_from_db(cfg: Dict[str, Any], group_id: str, out_dir: Path) -> None:
    """Construct wfo_summary.json and equity_curve.csv from DB runs tagged with wfo_group=group_id.
    Also generate per-window params JSON files for convenience.
    """
    db = cfg["storage"]["results_db"]
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db) as cx:
        # Get runs with this group and window indices
        runs = pd.read_sql_query(
            """
            SELECT r.run_id, r.finished_at,
                   MAX(CASE WHEN p.key='window_index' THEN CAST(p.value AS INTEGER) END) AS wi
            FROM runs r JOIN params p ON r.run_id=p.run_id
            WHERE r.finished_at IS NOT NULL AND EXISTS (
                SELECT 1 FROM params p2 WHERE p2.run_id=r.run_id AND p2.key='wfo_group' AND p2.value=?
            )
            GROUP BY r.run_id, r.finished_at
            ORDER BY wi
            """,
            cx,
            params=(group_id,),
        )

        summary: List[Dict[str, Any]] = []
        equity_rows: List[Dict[str, Any]] = []
        capital0 = float(cfg["risk"]["capital"]) if not runs.empty else 0.0
        cur_equity = capital0

        for _, row in runs.iterrows():
            rid = str(row["run_id"]) ; wi = int(row["wi"]) if row["wi"] is not None else 0
            # Metrics
            m = pd.read_sql_query("SELECT key,value FROM metrics WHERE run_id=?", cx, params=(rid,))
            md = {str(k): float(v) for k, v in zip(m["key"], m["value"])}
            # Best params -> write file for convenience
            params_df = pd.read_sql_query("SELECT key,value FROM params WHERE run_id=? AND key LIKE 'best.%'", cx, params=(rid,))
            best = {str(k): json.loads(v) if str(v).startswith('{') or str(v).startswith('[') else (json.loads(v) if v in ('true','false','null') else v) for k, v in zip(params_df["key"], params_df["value"])}
            if best:
                (windows_dir / f"window_{wi:02d}_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

            # Trades to build equity
            tr = pd.read_sql_query("SELECT exit_time, pnl FROM trades WHERE run_id=? ORDER BY exit_time", cx, params=(rid,))
            for _, t in tr.iterrows():
                cur_equity += float(t["pnl"]) ; equity_rows.append({"time": pd.to_datetime(t["exit_time"]).isoformat(), "equity": cur_equity})

            summary.append({
                "window": wi,
                "run_id": rid,
                "profit_factor": float(md.get("profit_factor", 0.0)),
                "sharpe": float(md.get("sharpe", 0.0)),
                "trades": float(md.get("trades", 0.0)),
                "pnl_total": float(md.get("pnl_total", 0.0)),
            })

    # Aggregate PF across runs
    agg_pf = 0.0
    if summary:
        with sqlite3.connect(db) as cx:
            profit = 0.0 ; loss = 0.0
            for s in summary:
                rid = s["run_id"]
                rows = pd.read_sql_query("SELECT pnl FROM trades WHERE run_id=?", cx, params=(rid,))
                p = rows[rows["pnl"] > 0]["pnl"].sum() ; l = -rows[rows["pnl"] < 0]["pnl"].sum()
                profit += float(p) ; loss += float(l)
            agg_pf = (profit / loss) if loss > 0 else float('inf')

    summary_obj = {
        "windows": summary,
        "agg": {
            "trades": int(sum(int(s["trades"]) for s in summary)),
            "pnl_total": float(sum(float(s["pnl_total"]) for s in summary)),
            "profit_factor": float(agg_pf),
        },
        "group": group_id,
    }
    (out_dir / "wfo_summary.json").write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
    pd.DataFrame(equity_rows).to_csv(out_dir / "equity_curve.csv", index=False)


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

            # Line chart with EMAs and trade markers (better visibility)
            bars = pd.read_sql_query(
                "SELECT close_time, close, ema_fast_30m, ema_slow_30m FROM bars WHERE run_id=? ORDER BY idx",
                cx,
                params=(rid,),
            )
            if not bars.empty:
                bars["Date"] = pd.to_datetime(bars["close_time"]) ; bars = bars.set_index("Date")
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(bars.index, bars["close"], color="black", linewidth=1, label="Close")
                if "ema_fast_30m" in bars:
                    ax.plot(bars.index, bars["ema_fast_30m"], color="#3b82f6", linewidth=1.0, label="EMA Fast")
                if "ema_slow_30m" in bars:
                    ax.plot(bars.index, bars["ema_slow_30m"], color="#f59e0b", linewidth=1.0, label="EMA Slow")

                # Trade markers
                tr2 = pd.read_sql_query(
                    "SELECT entry_time, exit_time, side, entry_price, exit_price FROM trades WHERE run_id=? ORDER BY trade_id",
                    cx,
                    params=(rid,),
                )
                if not tr2.empty:
                    tr2["entry_time"] = pd.to_datetime(tr2["entry_time"]) ; tr2["exit_time"] = pd.to_datetime(tr2["exit_time"])
                    # Entry and exit points mapped to nearest timestamps
                    for _, t in tr2.iterrows():
                        et = t["entry_time"] ; xt = t["exit_time"]
                        ep = float(t["entry_price"]) ; xp = float(t["exit_price"]) 
                        # Only plot if timestamp exists in index
                        if et in bars.index:
                            ax.scatter([et], [ep], color="green", marker="^", s=40, zorder=3)
                        if xt in bars.index:
                            ax.scatter([xt], [xp], color="red", marker="v", s=40, zorder=3)

                ax.set_title(f"Window {wi} — Price (line)")
                ax.legend()
                fig.tight_layout()
                fig.savefig(windows_dir / f"window_{wi:02d}_candles.png", dpi=150)
                plt.close(fig)



def main() -> None:
    cfg = load_config()
    root = Path(cfg["storage"]["artifacts_dir"]) ; root.mkdir(parents=True, exist_ok=True)
    wfo_dir = latest_wfo_dir(root)
    # If no artifacts, try to reconstruct from DB using latest group id
    if not wfo_dir:
        group = latest_wfo_group_from_db(cfg["storage"]["results_db"])
        if not group:
            print("No WFO artifacts or groups found.")
            return
        wfo_dir = root / group
        build_wfo_artifacts_from_db(cfg, group, wfo_dir)
    summary_json = wfo_dir / "wfo_summary.json"
    equity_csv = wfo_dir / "equity_curve.csv"
    plot_equity(equity_csv, wfo_dir / "wfo_equity.png")
    plot_windows(summary_json, wfo_dir / "wfo_windows.png")
    # New: per-window equity + candles
    per_window_reports(cfg, wfo_dir)
    print(f"WFO report exported to {wfo_dir}")


if __name__ == "__main__":
    main()
