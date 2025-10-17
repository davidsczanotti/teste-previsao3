from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class MonteCarloConfig:
    sims: int = 1000
    steps: int | None = None  # default: use number of trades
    seed: int = 42


def _get_last_run_id(cx: sqlite3.Connection) -> str | None:
    cur = cx.execute("SELECT run_id FROM runs WHERE finished_at IS NOT NULL ORDER BY finished_at DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row else None


def _load_trades(cx: sqlite3.Connection, run_id: str) -> List[Dict]:
    cur = cx.execute(
        "SELECT entry_price, qty, pnl FROM trades WHERE run_id=? ORDER BY trade_id",
        (run_id,),
    )
    rows = cur.fetchall()
    trades = [
        {
            "entry_price": float(r[0]),
            "qty": float(r[1]),
            "pnl": float(r[2]),
        }
        for r in rows
    ]
    return trades


def _compute_returns(trades: List[Dict]) -> np.ndarray:
    rets = []
    for t in trades:
        denom = t["entry_price"] * max(t["qty"], 1e-12)
        if denom <= 0:
            continue
        rets.append(t["pnl"] / denom)
    return np.array(rets, dtype=float)


def run_monte_carlo(db_path: str | Path, run_id: str | None, cfg: MonteCarloConfig) -> Dict[str, float]:
    """Bootstrap dos retornos por trade e projeta distribuição de PnL futuro."""
    rng = np.random.default_rng(cfg.seed)
    with sqlite3.connect(str(db_path)) as cx:
        rid = run_id or _get_last_run_id(cx)
        if not rid:
            raise RuntimeError("No finished runs found to run Monte Carlo.")
        trades = _load_trades(cx, rid)
        if not trades:
            raise RuntimeError("Selected run has no trades to simulate.")
        returns = _compute_returns(trades)
        steps = cfg.steps or len(returns)

        # Bootstrap per-trade returns, assume IID
        sims = []
        for _ in range(cfg.sims):
            sample = rng.choice(returns, size=steps, replace=True)
            cum_ret = float(np.prod(1.0 + sample) - 1.0)
            sims.append(cum_ret)
        arr = np.array(sims)

        p05, p50, p95 = np.percentile(arr, [5, 50, 95])
        result = {
            "mc_p05": float(p05),
            "mc_p50": float(p50),
            "mc_p95": float(p95),
            "mc_mean": float(arr.mean()),
        }
        return result


def save_artifact(artifacts_dir: str | Path, run_id: str, obj: Dict) -> Path:
    path = Path(artifacts_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)
    out = path / "monte_carlo.json"
    out.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return out

