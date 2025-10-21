from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _safe_log(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def _timeframe_to_pandas_freq(tf: str) -> str:
    unit = tf[-1]
    value = int(tf[:-1])
    if unit == "m":
        # Use explicit 'min' to avoid deprecation of aliases
        return f"{value}min"
    if unit == "h":
        # Lowercase 'h' (uppercase 'H' is deprecated)
        return f"{value}h"
    if unit == "d":
        return f"{value}D"
    if unit == "w":
        return f"{value}W"
    raise ValueError(f"Unsupported timeframe: {tf}")


@dataclass
class SeqConfig:
    lookback: int = 64  # L
    horizon: int = 16   # H
    features: Tuple[str, ...] = ("close", "volume", "open", "high", "low")
    

class OHLCVDiffusionDataset(Dataset):
    """
    Builds (condition, target) pairs from a OHLCV DataFrame for diffusion training.

    - cond_x: past window (L x d) -> encoded later to a vector
    - y: future univariate sequence (H x 1) of log-returns of close
    """

    def __init__(self, df: pd.DataFrame, cfg: SeqConfig):
        df = df.sort_values("Date").reset_index(drop=True)
        self.cfg = cfg
        self.symbol = df.attrs.get("ticker", "UNKNOWN")

        # base arrays
        close = df["close"].astype(float).values
        volume = df["volume"].astype(float).values
        open_ = df["open"].astype(float).values
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values

        # log returns for close; stabilize by small eps
        log_close = _safe_log(close)
        log_ret = np.diff(log_close, prepend=log_close[0])

        # simple feature matrix: normalized z-scores over rolling window for stability
        feats: List[np.ndarray] = []
        for name in cfg.features:
            if name == "close":
                arr = close
            elif name == "volume":
                arr = volume
            elif name == "open":
                arr = open_
            elif name == "high":
                arr = high
            elif name == "low":
                arr = low
            else:
                raise ValueError(f"Unsupported feature: {name}")
            feats.append(arr)

        X = np.stack(feats, axis=1)  # [N, d]
        # z-score normalize per column
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

        L, H = cfg.lookback, cfg.horizon
        xs, ys = [], []
        # target: future log returns of close (univariate)
        for end in range(L, len(df) - H):
            x_win = X[end - L : end]  # [L, d]
            y_seq = log_ret[end : end + H][:, None]  # [H, 1]
            xs.append(x_win.astype(np.float32))
            ys.append(y_seq.astype(np.float32))

        self.xs = np.stack(xs, axis=0) if xs else np.empty((0, L, X.shape[1]), dtype=np.float32)
        self.ys = np.stack(ys, axis=0) if ys else np.empty((0, H, 1), dtype=np.float32)
        self.dates = df["Date"].values
        self.timeframe = None  # can be set by caller (string like '15m')
        self.last_close = float(close[-1]) if len(close) > 0 else 0.0

    def __len__(self) -> int:
        return self.xs.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.xs[idx])  # [L, d]
        y = torch.from_numpy(self.ys[idx])  # [H, 1]
        # Flatten condition to a vector for encoder simplicity
        cond_vec = x.reshape(-1)
        return cond_vec, y

    @property
    def cond_dim(self) -> int:
        return int(self.xs.shape[1] * self.xs.shape[2])

    @property
    def horizon(self) -> int:
        return int(self.ys.shape[1]) if self.ys.size else 0

    @property
    def lookback(self) -> int:
        return int(self.xs.shape[1]) if self.xs.size else 0

    def future_index(self, last_date: pd.Timestamp, timeframe: str) -> pd.DatetimeIndex:
        freq = _timeframe_to_pandas_freq(timeframe)
        return pd.date_range(last_date, periods=self.horizon + 1, freq=freq, inclusive="right")
