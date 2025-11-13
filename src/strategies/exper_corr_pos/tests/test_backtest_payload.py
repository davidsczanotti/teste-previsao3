from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import src.strategies.exper_corr_pos.backtest as bt


def _make_dummy_ohlcv(n: int) -> pd.DataFrame:
    # Simple increasing price to produce deterministic non-negative pnl
    base = np.linspace(100.0, 101.0, n)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.25,
            "volume": np.full(n, 10.0),
        },
        index=pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC"),
    )


def test_run_backtest_minimal_payload(monkeypatch, tmp_path: Path):
    # Patch CFG path output dir to temporary to avoid writing to repo
    bt.OUTDIR = tmp_path / "reports"  # type: ignore

    # Monkeypatch data loaders to avoid real cache access
    def _fake_load_primary(_cfg: Dict[str, Any]):
        return _make_dummy_ohlcv(30)

    def _fake_load_confirm(_cfg: Dict[str, Any]):
        return _make_dummy_ohlcv(30)

    def _fake_prepare_dataset(df, *, config=None, confirm_df=None):
        # Minimal dataset: copy OHLCV + a single feature column expected by env
        out = df.copy()
        out["atr_14"] = 1.0
        return out

    monkeypatch.setattr(bt, "load_primary_series", _fake_load_primary)
    monkeypatch.setattr(bt, "load_confirm_series", _fake_load_confirm)
    monkeypatch.setattr(bt, "prepare_dataset", _fake_prepare_dataset)

    cfg = {
        "data": {"base_symbol": "FAKEBTC", "confirm_symbol": "FAKEETH", "timeframe": "1d"},
        "env": {},
        "train": {"eval_days": 10, "seed": 7},
        "model": {"num_experts": 2},
    }

    result = bt.run_backtest(cfg)
    # Required keys from minimal results schema
    for key in [
        "strategy",
        "symbol",
        "interval",
        "period",
        "trades",
        "total_pnl",
        "win_rate",
        "profit_factor",
        "avg_win",
        "avg_loss",
        "chart_path",
        "config_path",
        "seed",
        "run_env",
    ]:
        assert key in result

    # Types/structure sanity
    assert isinstance(result["period"], dict) and "start" in result["period"] and "end" in result["period"]
    assert isinstance(result["run_env"], dict) and "python" in result["run_env"]

    # Chart file should be created
    assert Path(result["chart_path"]).exists()

