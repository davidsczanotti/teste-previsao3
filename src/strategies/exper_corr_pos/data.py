from __future__ import annotations

from typing import Optional, Dict, Any

import pandas as pd

from ...utils.data_loader import load_data as _load_cached, load_data_range as _load_range
from .features import compute_features


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee that the dataframe is indexed by datetime (UTC)."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "Date" not in out.columns:
            raise ValueError("DataFrame must include a DatetimeIndex or a 'Date' column.")
        out["Date"] = pd.to_datetime(out["Date"], utc=True, errors="coerce")
        out = out.dropna(subset=["Date"]).set_index("Date")
    else:
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    return out.sort_index()


def load_symbol_history(
    symbol: str,
    timeframe: str,
    *,
    days: int = 3650,
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_cache_only: bool = True,
) -> pd.DataFrame:
    """Load OHLCV history from the local cache (or API fallback)."""
    if start and end:
        df = _load_range(symbol, timeframe, start, end, use_cache_only=use_cache_only)
    else:
        df = _load_cached(symbol, timeframe, days=days, use_cache_only=use_cache_only)
    if df.empty:
        raise ValueError(f"Nenhum dado disponível para {symbol} @ {timeframe}.")
    return _ensure_datetime_index(df)


def load_primary_series(config: Dict[str, Any]) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    symbol = data_cfg.get("base_symbol", "BTCUSDT")
    timeframe = data_cfg.get("timeframe", "1h")
    days = int(data_cfg.get("lookback_days", 3650))
    start = data_cfg.get("start")
    end = data_cfg.get("end")
    return load_symbol_history(symbol, timeframe, days=days, start=start, end=end)


def load_confirm_series(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    data_cfg = config.get("data", {})
    confirm_symbol = data_cfg.get("confirm_symbol")
    if not confirm_symbol:
        return None
    timeframe = data_cfg.get("timeframe", "1h")
    days = int(data_cfg.get("lookback_days", 3650))
    start = data_cfg.get("start")
    end = data_cfg.get("end")
    return load_symbol_history(confirm_symbol, timeframe, days=days, start=start, end=end)


def prepare_dataset(
    df: pd.DataFrame,
    *,
    config: Optional[Dict[str, Any]] = None,
    confirm_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    cfg = config or {}
    base = _ensure_datetime_index(df)
    confirm = confirm_df if confirm_df is not None else load_confirm_series(cfg)

    data_cfg = cfg.get("data", {})
    higher_tf = data_cfg.get("higher_timeframe", "4h")
    ml_horizon = int(data_cfg.get("ml_horizon", 3))
    ml_decay = float(data_cfg.get("ml_decay", 0.995))
    ml_ridge = float(data_cfg.get("ml_ridge", 0.001))
    spread_window = int(data_cfg.get("spread_window", 240))

    feats = compute_features(
        base,
        higher_tf=higher_tf,
        confirm_df=confirm,
        ml_horizon=ml_horizon,
        ml_decay=ml_decay,
        ml_ridge=ml_ridge,
        spread_window=spread_window,
    )

    base_aligned = base.loc[feats.index].copy()
    dataset = base_aligned.copy()
    for col in feats.columns:
        dataset[col] = feats[col]
    drop_cols = [c for c in ("Date",) if c in dataset.columns]
    if drop_cols:
        dataset = dataset.drop(columns=drop_cols)
    return dataset
