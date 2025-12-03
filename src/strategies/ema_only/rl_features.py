from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import compute_ema


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def build_features(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    fast: int,
    slow: int,
    ref_ema_period: int = 200,
    atr_period: int = 14,
    pattern_window: int = 3,
) -> pd.DataFrame:
    """Constroi features para RL usando EMAs, MACD e viés de timeframe superior, com sinais de experts."""
    out = df.copy().sort_values("Date").reset_index(drop=True)

    out["ema_fast"] = compute_ema(out["close"].astype(float), fast)
    out["ema_slow"] = compute_ema(out["close"].astype(float), slow)
    # MACD padrão (12, 26, 9)
    close_prices = out["close"].astype(float)
    macd_fast = compute_ema(close_prices, 12)
    macd_slow = compute_ema(close_prices, 26)
    out["macd_line"] = macd_fast - macd_slow
    out["macd_signal"] = compute_ema(out["macd_line"], 9)
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]

    ref_sorted = ref_df.sort_values("Date").reset_index(drop=True).copy()
    ref_sorted["ref_ema"] = compute_ema(ref_sorted["close"].astype(float), ref_ema_period)
    out = pd.merge_asof(out, ref_sorted[["Date", "ref_ema"]], on="Date", direction="backward")

    out["dist_fast"] = (out["close"] - out["ema_fast"]) / out["close"]
    out["dist_slow"] = (out["close"] - out["ema_slow"]) / out["close"]
    out["dist_ref"] = (out["close"] - out["ref_ema"]) / out["close"]

    out["slope_fast"] = out["ema_fast"].diff()
    out["slope_slow"] = out["ema_slow"].diff()
    out["atr"] = compute_atr(out, atr_period)
    out["atr_rel"] = out["atr"] / out["close"]
    # Retornos curtos (velocidade)
    out["ret_1"] = out["close"].pct_change().fillna(0.0)
    out["ret_5"] = out["close"].pct_change(5).fillna(0.0)
    out["ret_20"] = out["close"].pct_change(20).fillna(0.0)
    # Volume normalizado
    vol = out["volume"].astype(float)
    out["vol_z"] = (vol - vol.rolling(100, min_periods=20).mean()) / vol.rolling(100, min_periods=20).std().replace(0.0, np.nan)
    out["vol_z"] = out["vol_z"].fillna(0.0)
    # OBV simples
    direction = np.sign(out["ret_1"])
    out["obv"] = (direction * vol).fillna(0.0).cumsum()
    out["obv_z"] = (out["obv"] - out["obv"].rolling(200, min_periods=20).mean()) / out["obv"].rolling(200, min_periods=20).std().replace(0.0, np.nan)
    out["obv_z"] = out["obv_z"].fillna(0.0)

    # Experts (MoE-style) focados em EMAs e MACD
    out["exp_trend"] = (out["ema_fast"] > out["ema_slow"]).astype(float)
    out["exp_ref"] = (out["close"] > out["ref_ema"]).astype(float)
    out["exp_macd"] = (out["macd_hist"] > 0).astype(float)
    out["exp_slope"] = (out["slope_fast"] > 0).astype(float)

    # Especialista intraday é injetado em rl_train (colunas exp_intraday_trend / intraday_align_ratio).
    if "exp_intraday_trend" not in out.columns:
        out["exp_intraday_trend"] = 0.0
    if "intraday_align_ratio" not in out.columns:
        out["intraday_align_ratio"] = 0.0

    # Consenso: média de especialistas (trend, ref, macd, slope, intraday)
    expert_cols = ["exp_trend", "exp_ref", "exp_macd", "exp_slope", "exp_intraday_trend"]
    out["experts_mean"] = out[expert_cols].mean(axis=1)

    features = out[
        [
            "ema_fast",
            "ema_slow",
            "ref_ema",
            "dist_fast",
            "dist_slow",
            "dist_ref",
            "slope_fast",
            "slope_slow",
            "atr_rel",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "ret_1",
            "ret_5",
            "ret_20",
            "vol_z",
            "obv_z",
            "exp_trend",
            "exp_ref",
            "exp_macd",
            "exp_slope",
            "intraday_align_ratio",
            "experts_mean",
            "exp_intraday_trend",
        ]
    ].copy()

    # Preenche valores iniciais
    features = features.ffill().fillna(0.0)
    return features.astype(np.float32)
