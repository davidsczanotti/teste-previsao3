from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / (std + 1e-9)


def _online_ridge_predictions(
    X: np.ndarray,
    y: np.ndarray,
    decay: float = 0.995,
    ridge: float = 1e-3,
    *,
    horizon: int = 0,
) -> np.ndarray:
    """
    Recursive least squares with forgetting factor using delayed updates to avoid lookahead.

    For a forward horizon H, the prediction at index t is computed using weights that have
    only seen targets up to index (t-H). The update using sample i is applied when the
    target y[i] becomes available at time i+H.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float64)
    cov = np.eye(n_features, dtype=np.float64) / max(ridge, 1e-12)
    preds = np.zeros(n_samples, dtype=np.float64)

    H = max(0, int(horizon))
    for t in range(n_samples):
        xt = X[t]
        # Always compute a prediction with current weights (before any same-step update)
        preds[t] = float(np.dot(weights, xt)) if np.all(np.isfinite(xt)) else 0.0

        # Apply delayed update for the sample whose target has just become available
        i = t - H
        if i >= 0:
            xi = X[i]
            yi = y[i]
            if np.all(np.isfinite(xi)) and np.isfinite(yi):
                xi_col = xi.reshape(-1, 1)
                denom_term = (xi_col.T @ cov @ xi_col).item()
                denom = float(decay) + float(denom_term)
                if denom > 1e-12:
                    gain = (cov @ xi_col) / denom
                    # use current prediction for residual (consistent RLS)
                    pred_i = float(np.dot(weights, xi))
                    weights = weights + (yi - pred_i) * gain.ravel()
                    cov = (cov - gain @ xi_col.T @ cov) / float(max(decay, 1e-12))
    return preds


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return agg.dropna(how="any")


def _prepare_confirm_df(confirm_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    aligned = confirm_df.reindex(index)
    return aligned.interpolate(method="time").ffill().bfill()


def _linear_regression_slope(series: pd.Series, window: int) -> pd.Series:
    """Slope of a rolling linear regression (normalized by window length)."""
    if window <= 1:
        return pd.Series(0.0, index=series.index)
    idx = np.arange(window, dtype=np.float64)
    denom = np.sum((idx - idx.mean()) ** 2)

    def _slope(sub: np.ndarray) -> float:
        if np.any(~np.isfinite(sub)):
            return np.nan
        y = sub.astype(np.float64)
        num = np.sum((idx - idx.mean()) * (y - y.mean()))
        return num / (denom + 1e-9)

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


def compute_features(
    df: pd.DataFrame,
    *,
    higher_tf: Optional[str] = "4h",
    confirm_df: Optional[pd.DataFrame] = None,
    ema_fast: int = 21,
    ema_slow: int = 55,
    htf_ema_fast: int = 34,
    htf_ema_slow: int = 89,
    ml_horizon: int = 3,
    ml_decay: float = 0.995,
    ml_ridge: float = 1e-3,
    spread_window: int = 240,
) -> pd.DataFrame:
    """
    Feature generator specialised for the exper_corr_pos MoE agent.

    Specialists:
      - TrendML: confluência entre técnicos de tendência e um preditor direcional online (pseudo-LightGBM).
      - MultiFrame: filtros de tendência/trigger usando múltiplos timeframes.
      - Spread: cointegração com par positivo (spread z-score, triggers Bollinger).
      - Pattern: forma de candles e contagens rolling normalizadas por ATR.
    """
    base = df.copy()
    base = base.sort_index()

    open_ = base["open"]
    high = base["high"]
    low = base["low"]
    close = base["close"]
    volume = base["volume"]
    returns = close.pct_change()
    log_returns = np.log(close.replace(0.0, np.nan)).diff()

    out = pd.DataFrame(index=base.index)

    # --- TrendML specialist -------------------------------------------------
    ema_fast_series = _ema(close, max(1, ema_fast))
    ema_slow_series = _ema(close, max(2, ema_slow))
    out["ema_fast"] = ema_fast_series
    out["ema_slow"] = ema_slow_series
    out["ema_cross"] = ema_fast - ema_slow
    out["ema_ratio"] = ema_fast_series / (ema_slow_series + 1e-9)
    out["trend_strength"] = out["ema_cross"] / (close.rolling(max(ema_slow, 2)).std() + 1e-9)
    out["trend_accel"] = out["ema_cross"].diff()

    donchian_up = high.rolling(40, min_periods=40).max()
    donchian_down = low.rolling(40, min_periods=40).min()
    out["breakout_up"] = (close - donchian_up) / (close + 1e-9)
    out["breakout_down"] = (close - donchian_down) / (close + 1e-9)
    out["donchian_width"] = (donchian_up - donchian_down) / (close + 1e-9)

    atr_14 = _atr(base, 14)
    out["atr_14"] = atr_14
    out["atr_norm"] = atr_14 / (close.rolling(55).mean().abs() + 1e-9)
    atr_28 = _atr(base, 28)
    out["atr_ratio"] = atr_14 / (atr_28 + 1e-9)

    hv_14 = log_returns.rolling(14, min_periods=14).std() * math.sqrt(365.0)
    hv_30 = log_returns.rolling(30, min_periods=30).std() * math.sqrt(365.0)
    out["realized_vol_14"] = hv_14
    out["realized_vol_30"] = hv_30
    out["realized_vol_ratio"] = hv_14 / (hv_30 + 1e-9)
    out["vol_zscore_30"] = _zscore(hv_14, 30)

    ml_features = pd.DataFrame(index=base.index)
    ml_features["lag_ret_1"] = returns.shift(1)
    ml_features["lag_ret_2"] = returns.shift(2)
    ml_features["lag_ret_3"] = returns.shift(3)
    ml_features["lag_ret_6"] = returns.shift(6)
    ml_features["lag_ret_12"] = returns.shift(12)
    ml_features["lag_ret_24"] = returns.shift(24)
    ml_features["rsi_7"] = _rsi(close, 7) / 100.0
    ml_features["rsi_14"] = _rsi(close, 14) / 100.0
    ml_features["rsi_21"] = _rsi(close, 21) / 100.0
    ml_features["atr_rel"] = atr_14 / (close + 1e-9)
    ml_features["range_rel"] = (high - low) / (close + 1e-9)
    ml_features["mom_10"] = close.pct_change(10)
    ml_features["mom_20"] = close.pct_change(20)
    ml_features["price_slope_20"] = _linear_regression_slope(close, 20)
    ml_features["ema_fast_dev"] = (close - ema_fast) / (atr_14 + 1e-9)
    ml_features["ema_slow_dev"] = (close - ema_slow) / (atr_14 + 1e-9)
    ml_features["hour_sin"] = np.sin(2 * math.pi * base.index.hour / 24.0)
    ml_features["hour_cos"] = np.cos(2 * math.pi * base.index.hour / 24.0)
    # Use modern forward-fill API to avoid FutureWarning
    ml_features = ml_features.ffill().fillna(0.0)

    feature_cols = ml_features.columns.tolist()
    X = ml_features.to_numpy(dtype=np.float64)
    y = close.pct_change(ml_horizon).shift(-ml_horizon).to_numpy(dtype=np.float64)
    preds = _online_ridge_predictions(X, y, decay=ml_decay, ridge=ml_ridge, horizon=ml_horizon)
    ml_pred = pd.Series(preds, index=base.index)
    out["ml_pred_return"] = ml_pred
    scale = ml_pred.rolling(200, min_periods=50).std().replace(0.0, np.nan)
    score = ml_pred / (scale + 1e-9)
    score = score.clip(-5, 5)
    prob_up = 1.0 / (1.0 + np.exp(-score))
    out["ml_prob_up"] = prob_up
    out["ml_confidence"] = (prob_up - 0.5).abs()
    out["trend_ml_alignment"] = np.sign(out["ema_cross"].fillna(0.0)) * (prob_up - 0.5)

    # --- MultiFrame specialist ----------------------------------------------
    if higher_tf:
        try:
            higher = _resample_ohlcv(base, higher_tf)
            higher["ema_fast_htf"] = _ema(higher["close"], max(1, htf_ema_fast))
            higher["ema_slow_htf"] = _ema(higher["close"], max(2, htf_ema_slow))
            higher["htf_trend_state"] = np.sign(higher["ema_fast_htf"] - higher["ema_slow_htf"])
            higher["htf_trend_strength"] = higher["ema_fast_htf"] - higher["ema_slow_htf"]
            higher["htf_rsi"] = _rsi(higher["close"], 14)
            higher["htf_atr"] = _atr(higher, 14)
            higher["htf_vol"] = np.log(higher["close"] + 1e-9).diff().rolling(30, min_periods=30).std()
            higher = higher.reindex(out.index, method="ffill")
            out["htf_trend_state"] = higher["htf_trend_state"]
            out["htf_trend_strength"] = higher["htf_trend_strength"]
            out["htf_rsi"] = higher["htf_rsi"] / 100.0
            out["htf_atr_rel"] = higher["htf_atr"] / (higher["close"] + 1e-9)
            out["htf_vol"] = higher["htf_vol"]
        except Exception:
            out["htf_trend_state"] = 0.0
            out["htf_trend_strength"] = 0.0
            out["htf_rsi"] = 0.5
            out["htf_atr_rel"] = 0.0
            out["htf_vol"] = 0.0
    else:
        out["htf_trend_state"] = 0.0
        out["htf_trend_strength"] = 0.0
        out["htf_rsi"] = 0.5
        out["htf_atr_rel"] = 0.0
        out["htf_vol"] = 0.0

    out["ltf_rsi_5"] = _rsi(close, 5) / 100.0
    out["ltf_pullback"] = (close - ema_fast) / (atr_14 + 1e-9)
    out["multiframe_confluence"] = out["htf_trend_state"] * (out["ltf_rsi_5"] - 0.5)
    out["pullback_alignment"] = out["ltf_pullback"] * out["htf_trend_state"]

    # --- Spread / cointegração specialist -----------------------------------
    if confirm_df is not None:
        confirm_aligned = _prepare_confirm_df(confirm_df, out.index)
        ref_close = confirm_aligned["close"]
        log_base = np.log(close + 1e-9)
        log_ref = np.log(ref_close + 1e-9)

        window = max(60, int(spread_window))
        cov = log_base.rolling(window, min_periods=window).cov(log_ref)
        var = log_ref.rolling(window, min_periods=window).var()
        beta = cov / (var + 1e-9)
        # Replace infs and forward-fill using the dedicated method (FutureWarning-safe)
        beta = beta.replace([np.inf, -np.inf], np.nan).ffill()
        spread = log_base - beta * log_ref
        spread_mean = spread.rolling(window, min_periods=window).mean()
        spread_std = spread.rolling(window, min_periods=window).std()
        spread_z = (spread - spread_mean) / (spread_std + 1e-9)

        out["spread_beta"] = beta
        out["spread_z"] = spread_z
        out["spread_z_slope"] = spread_z.diff()
        out["spread_vol"] = spread_std
        out["spread_bb_upper"] = spread_mean + 2 * spread_std
        out["spread_bb_lower"] = spread_mean - 2 * spread_std
        out["spread_revert_signal"] = -spread_z
        out["vecm_error"] = spread.diff() - spread.diff().rolling(window, min_periods=window).mean()
        confirm_returns = ref_close.pct_change()
        out["roll_corr_confirm_60"] = returns.rolling(60, min_periods=30).corr(confirm_returns)
        out["roll_corr_confirm_120"] = returns.rolling(120, min_periods=60).corr(confirm_returns)
        out["spread_z_zscore"] = _zscore(spread_z, window)
    else:
        out["spread_beta"] = 0.0
        out["spread_z"] = 0.0
        out["spread_z_slope"] = 0.0
        out["spread_vol"] = 0.0
        out["spread_bb_upper"] = 0.0
        out["spread_bb_lower"] = 0.0
        out["spread_revert_signal"] = 0.0
        out["vecm_error"] = 0.0
        out["roll_corr_confirm_60"] = 0.0
        out["roll_corr_confirm_120"] = 0.0
        out["spread_z_zscore"] = 0.0

    # --- Pattern specialist --------------------------------------------------
    candle_range = (high - low).replace(0.0, np.nan)
    body = close - open_
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    out["body"] = body
    out["body_abs"] = body.abs()
    out["body_range_pct"] = body / (candle_range + 1e-9)
    out["upper_wick"] = upper_wick
    out["lower_wick"] = lower_wick

    atr = atr_14.replace(0.0, np.nan)
    out["body_atr"] = body / (atr + 1e-9)
    out["upper_wick_atr"] = upper_wick / (atr + 1e-9)
    out["lower_wick_atr"] = lower_wick / (atr + 1e-9)

    body_ratio = body.abs() / (candle_range + 1e-9)
    out["doji_flag"] = (body_ratio < 0.1).astype(float)
    out["hammer_flag"] = (
        (lower_wick >= 2 * body.abs()) & (upper_wick <= 0.5 * body.abs())
    ).astype(float)
    out["shooting_star_flag"] = (
        (upper_wick >= 2 * body.abs()) & (lower_wick <= 0.5 * body.abs())
    ).astype(float)

    prev_body = body.shift()
    prev_open = open_.shift()
    prev_close = close.shift()
    bullish_engulf = (
        (close > open_)
        & (prev_close < prev_open)
        & (close >= prev_open)
        & (open_ <= prev_close)
    )
    bearish_engulf = (
        (close < open_)
        & (prev_close > prev_open)
        & (close <= prev_open)
        & (open_ >= prev_close)
    )
    out["bullish_engulf_flag"] = bullish_engulf.astype(float)
    out["bearish_engulf_flag"] = bearish_engulf.astype(float)
    out["hammer_roll3"] = out["hammer_flag"].rolling(3, min_periods=1).mean()
    out["shooting_star_roll3"] = out["shooting_star_flag"].rolling(3, min_periods=1).mean()
    out["engulf_diff_roll5"] = (
        out["bullish_engulf_flag"] - out["bearish_engulf_flag"]
    ).rolling(5, min_periods=1).mean()
    out["wick_imbalance"] = (out["upper_wick_atr"] - out["lower_wick_atr"])
    out["body_direction"] = np.sign(body).fillna(0.0)
    out["gap_up"] = (open_ > prev_close).astype(float)
    out["gap_down"] = (open_ < prev_close).astype(float)
    out["body_range_pct_roll5"] = out["body_range_pct"].rolling(5, min_periods=1).mean()

    # Context helpers shared across experts
    out["ret_1"] = returns
    out["ret_4"] = close.pct_change(4)
    out["ret_vol_24"] = returns.rolling(24, min_periods=12).std()
    out["volume_zscore"] = _zscore(volume, 120)
    out["ret_skew_60"] = returns.rolling(60, min_periods=30).skew()
    out["ret_kurt_60"] = returns.rolling(60, min_periods=30).kurt()
    out["cumulative_return_90"] = (1 + returns).rolling(90, min_periods=30).apply(np.prod, raw=True) - 1.0
    out["drawdown_rolling_90"] = (close / close.rolling(90, min_periods=30).max()) - 1.0

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out
