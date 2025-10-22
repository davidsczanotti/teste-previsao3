from __future__ import annotations

"""
Monte Carlo simulation (no flags) for the Al Brooks strategy.
Reads config from src/strategies/al_brooks/config.json, runs a backtest to collect
trades, then performs block-bootstrap Monte Carlo on the trade sequence.
Outputs:
- JSON with distribution stats
- Histograms PNG (P&L, Profit Factor, Max Drawdown)
All artifacts go to src/strategies/al_brooks/reports/monte_carlo/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import backtest_al_brooks_inside_bar
from .config import load_active_config
from ...utils.data_loader import load_data
from ...utils.metrics import calculate_metrics


def _equity_curve_from_trades(trades: List[Dict[str, Any]]) -> pd.Series:
    pnls = [t.get("pnl", 0.0) for t in trades if "pnl" in t]
    if not pnls:
        return pd.Series(dtype=float)
    return pd.Series(pnls).cumsum()


def _max_drawdown_from_curve(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    cummax = curve.cummax()
    dd = curve - cummax
    return float(dd.min())


def _block_bootstrap_indices(n: int, block: int, m: int) -> np.ndarray:
    """Return indices for m bootstrap sequences using block bootstrap of length `block`."""
    # number of blocks per sequence
    b = max(1, int(np.ceil(n / block)))
    idx = np.random.randint(0, n - block + 1, size=(m, b))
    # stitch blocks
    sequences = []
    for row in idx:
        seq = np.concatenate([np.arange(start, start + block) for start in row])[:n]
        sequences.append(seq)
    return np.vstack(sequences)


def monte_carlo_on_trades(trades: List[Dict[str, Any]], simulations: int = 500, block: int = 10) -> Dict[str, Any]:
    if not trades:
        return {"simulations": 0, "message": "no trades"}
    # Prepare array of trade PnLs
    closed = [t for t in trades if "pnl" in t]
    if not closed:
        return {"simulations": 0, "message": "no closed trades"}
    pnls = np.array([float(t["pnl"]) for t in closed], dtype=float)
    n = len(pnls)
    # Bootstrap indices
    idx_mat = _block_bootstrap_indices(n, block=max(1, block), m=simulations)
    totals, pfs, mdds = [], [], []
    for row in idx_mat:
        seq = pnls[row]
        total = float(seq.sum())
        wins = seq[seq > 0].sum()
        losses = -seq[seq <= 0].sum()
        pf = float(wins / losses) if losses > 0 else float("inf")
        curve = pd.Series(seq).cumsum()
        mdd = _max_drawdown_from_curve(curve)
        totals.append(total)
        pfs.append(pf)
        mdds.append(mdd)
    return {
        "simulations": simulations,
        "totals": totals,
        "pfs": pfs,
        "mdds": mdds,
        "n_trades": n,
    }


def _save_hist(data: List[float], title: str, out_path: Path, bins: int = 40) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color="#1f77b4", alpha=0.8)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    cfg = load_active_config("BTCUSDT", "1m")  # values ignored if config.json exists
    if not cfg:
        print("Nenhuma configuração encontrada em config.json.")
        return

    # Load data (cache-only by default via loader)
    df = load_data(cfg.ticker, cfg.interval, days=cfg.days, use_cache_only=True)
    trades, total_pnl, df_ind = backtest_al_brooks_inside_bar(
        df.copy(),
        ema_fast_period=cfg.ema_fast_period,
        ema_medium_period=cfg.ema_medium_period,
        ema_slow_period=cfg.ema_slow_period,
        risk_reward_ratio=cfg.risk_reward_ratio,
        max_avg_deviation_pct=cfg.max_avg_deviation_pct,
        lot_size=cfg.lot_size,
        adx_period=cfg.adx_period,
        adx_threshold=cfg.adx_threshold,
        atr_period=cfg.atr_period,
        atr_stop_multiplier=cfg.atr_stop_multiplier,
        atr_trail_multiplier=cfg.atr_trail_multiplier,
        htf_lookback=cfg.htf_lookback,
        use_htf_bias=cfg.use_htf_bias,
        use_inside_bar=cfg.use_inside_bar,
        inside_bar_inclusive=cfg.inside_bar_inclusive,
        min_atr=cfg.min_atr,
        pullback_lookback=cfg.pullback_lookback,
        taker_fee_pct=cfg.taker_fee_pct,
        slippage_pct=cfg.slippage_pct,
    )

    base_dir = Path(__file__).resolve().parent / "reports" / "monte_carlo"
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

    # Monte Carlo
    mc = monte_carlo_on_trades(trades, simulations=500, block=10)
    # Save distributions
    _save_hist(mc.get("totals", []), f"Monte Carlo P&L ({cfg.ticker}@{cfg.interval})", base_dir / f"mc_pnl_{stamp}.png")
    _save_hist(mc.get("pfs", []), f"Monte Carlo Profit Factor ({cfg.ticker}@{cfg.interval})", base_dir / f"mc_pf_{stamp}.png")
    _save_hist(mc.get("mdds", []), f"Monte Carlo Max Drawdown ({cfg.ticker}@{cfg.interval})", base_dir / f"mc_mdd_{stamp}.png")

    metrics = calculate_metrics(trades)
    out_json = {
        "strategy": "al_brooks",
        "symbol": cfg.ticker,
        "interval": cfg.interval,
        "days": cfg.days,
        "config_used": cfg.to_dict(),
        "backtest_metrics": metrics,
        "monte_carlo": {
            "simulations": mc.get("simulations", 0),
            "pnl_stats": {
                "mean": float(np.mean(mc.get("totals", [0])) if mc.get("totals") else 0),
                "p05": float(np.percentile(mc.get("totals", [0]), 5)) if mc.get("totals") else 0,
                "p50": float(np.percentile(mc.get("totals", [0]), 50)) if mc.get("totals") else 0,
                "p95": float(np.percentile(mc.get("totals", [0]), 95)) if mc.get("totals") else 0,
            },
            "pf_stats": {
                "mean": float(np.mean(mc.get("pfs", [0])) if mc.get("pfs") else 0),
                "p05": float(np.percentile(mc.get("pfs", [0]), 5)) if mc.get("pfs") else 0,
                "p50": float(np.percentile(mc.get("pfs", [0]), 50)) if mc.get("pfs") else 0,
                "p95": float(np.percentile(mc.get("pfs", [0]), 95)) if mc.get("pfs") else 0,
            },
            "mdd_stats": {
                "mean": float(np.mean(mc.get("mdds", [0])) if mc.get("mdds") else 0),
                "p05": float(np.percentile(mc.get("mdds", [0]), 5)) if mc.get("mdds") else 0,
                "p50": float(np.percentile(mc.get("mdds", [0]), 50)) if mc.get("mdds") else 0,
                "p95": float(np.percentile(mc.get("mdds", [0]), 95)) if mc.get("mdds") else 0,
            },
        },
        "artifacts": {
            "pnl_hist": str(base_dir / f"mc_pnl_{stamp}.png"),
            "pf_hist": str(base_dir / f"mc_pf_{stamp}.png"),
            "mdd_hist": str(base_dir / f"mc_mdd_{stamp}.png"),
        },
        "timestamp": stamp,
    }
    out_path = base_dir / f"mc_summary_{cfg.ticker}_{cfg.interval}_{stamp}.json"
    out_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(f"Monte Carlo summary salvo em: {out_path}")


if __name__ == "__main__":
    main()

