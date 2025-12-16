import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.backtest import backtest_ema_only
from src.core.indicators import add_indicators
from src.core.signals import apply_signals


def test_trend_surfer_cci_uses_close_source():
    closes = [10, 11, 12, 11, 10, 11, 12]
    dates = pd.date_range("2025-01-01", periods=len(closes), freq="h")

    def _df(high_bump: float, low_bump: float) -> pd.DataFrame:
        base = pd.DataFrame(
            {
                "Date": dates,
                "open": closes,
                "high": [c + high_bump for c in closes],
                "low": [c - low_bump for c in closes],
                "close": closes,
                "volume": 1.0,
            }
        )
        return base

    config = {
        "strategy": {
            "signal_mode": "trend_surfer_v4",
            "ts_fast_period": 2,
            "ts_slow_period": 3,
            "ts_ema_macro_period": 2,
            "ts_cci_period": 2,
            "ts_cci_min": -1000,
        }
    }

    df1 = add_indicators(_df(high_bump=0.1, low_bump=0.1), config)
    df2 = add_indicators(_df(high_bump=10.0, low_bump=10.0), config)

    pd.testing.assert_series_equal(
        df1["ts_cci"],
        df2["ts_cci"],
        check_names=False,
        check_exact=False,
        atol=1e-12,
        rtol=0.0,
    )


def test_trend_surfer_backtest_fills_next_open_and_stop_next_bar():
    # Série feita para gerar cross-up com SMA(2) vs SMA(3) no índice 4.
    closes = [3, 2, 1, 2, 3, 4, 4]
    dates = pd.date_range("2025-01-01", periods=len(closes), freq="h")

    df = pd.DataFrame(
        {
            "Date": dates,
            "open": [3.0, 3.0, 2.0, 1.0, 2.0, 3.2, 4.0],
            "high": [3.0, 3.0, 2.0, 2.2, 3.2, 4.5, 4.1],
            "low": [3.0, 2.0, 1.0, 0.8, 1.8, 0.0, 3.0],
            "close": closes,
            "volume": 1.0,
        }
    )

    config = {
        "strategy": {
            "signal_mode": "trend_surfer_v4",
            "ts_fast_period": 2,
            "ts_slow_period": 3,
            "ts_ema_macro_period": 2,
            "ts_cci_period": 2,
            "ts_cci_min": -1000,
            "risk_per_trade_pct": 0.02,
            "initial_stop_pct": 0.05,
            "trailing_stop_pct": 0.10,
            "fee_pct": 0.0,
            "allow_short": False,
        },
        "backtest": {"initial_capital": 1000.0},
    }

    # Descobre onde o sinal acontece (close do candle do sinal).
    df_sig = add_indicators(df.copy(), config)
    df_sig = apply_signals(df_sig, config)
    signal_idxs = df_sig.index[df_sig["signal"] == 1].tolist()
    assert signal_idxs, "Esperávamos pelo menos 1 sinal de compra."
    signal_idx = signal_idxs[0]

    result = backtest_ema_only(df.copy(), config)
    trades = result["trades"]
    assert len(trades) == 1, "Esperávamos exatamente 1 trade."
    t0 = trades[0]

    fill_idx = signal_idx + 1
    assert t0["entry_time"] == df.loc[fill_idx, "Date"]
    assert t0["entry"] == df.loc[fill_idx, "open"]

    # O stop só fica ativo no candle seguinte ao de criação/atualização.
    assert t0["exit_time"] != df.loc[fill_idx, "Date"]
    assert t0["reason"] == "trailing_stop"
