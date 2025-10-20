from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
import pandas as pd


def get_groups(cx: sqlite3.Connection):
    df = pd.read_sql_query("SELECT DISTINCT value as grp FROM params WHERE key='wfo_group' ORDER BY grp", cx)
    return [str(x) for x in df["grp"].tolist()]


def agg_pf_for_group(cx: sqlite3.Connection, group: str):
    # Collect run_ids for group
    runs = pd.read_sql_query(
        "SELECT DISTINCT r.run_id FROM runs r JOIN params p ON r.run_id=p.run_id WHERE p.key='wfo_group' AND p.value=?",
        cx,
        params=(group,),
    )
    if runs.empty:
        return {"windows": 0, "trades": 0, "agg_pf": float("nan")}
    # Trades and PF
    windows = pd.read_sql_query(
        "SELECT DISTINCT CAST(value AS INTEGER) AS wi FROM params WHERE key='window_index' AND run_id IN (SELECT run_id FROM params WHERE key='wfo_group' AND value=?) ORDER BY wi",
        cx,
        params=(group,),
    )
    profit = 0.0
    loss = 0.0
    trades = 0
    for rid in runs["run_id"]:
        tr = pd.read_sql_query("SELECT pnl FROM trades WHERE run_id=?", cx, params=(rid,))
        trades += int(len(tr))
        p = tr[tr["pnl"] > 0]["pnl"].sum()
        l = -tr[tr["pnl"] < 0]["pnl"].sum()
        profit += float(p)
        loss += float(l)
    agg_pf = (profit / loss) if loss > 0 else float("inf")
    return {"windows": int(len(windows)), "trades": trades, "agg_pf": agg_pf}


def main() -> None:
    cfg_path = Path("src/strategies/experimento/config/config_active.json")
    import json
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    db = cfg["storage"]["results_db"]
    out_dir = Path(cfg["storage"]["artifacts_dir"]) ; out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "wfo_summary_history.csv"

    with sqlite3.connect(db) as cx:
        groups = get_groups(cx)
        rows = []
        for g in groups:
            agg = agg_pf_for_group(cx, g)
            rows.append({"group": g, "windows": agg["windows"], "trades": agg["trades"], "agg_pf": agg["agg_pf"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("WFO comparison exported:", out_csv)


if __name__ == "__main__":
    main()

