import numpy as np
import pandas as pd
import pandas_ta as ta


def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Calcula e adiciona todos os indicadores necessários ao DataFrame."""
    df = df.copy()

    ema_fast = params["ema_fast_period"]
    ema_medium = params["ema_medium_period"]
    ema_slow = params["ema_slow_period"]
    adx_period = params.get("adx_period", 14)
    atr_period = params.get("atr_period", 14)
    htf_lookback = params.get("htf_lookback", 0)

    df["ema_fast"] = ta.ema(df["close"], length=ema_fast)
    df["ema_medium"] = ta.ema(df["close"], length=ema_medium)
    df["ema_slow"] = ta.ema(df["close"], length=ema_slow)

    df["is_inside_bar"] = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    df["avg_deviation_pct"] = abs((df["close"] - df["ema_slow"]) / df["ema_slow"]) * 100

    df["atr"] = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=atr_period)

    adx_df = ta.adx(high=df["high"], low=df["low"], close=df["close"], length=adx_period)
    adx_col = f"ADX_{adx_period}"
    df["adx"] = adx_df[adx_col] if adx_df is not None and adx_col in adx_df else np.nan

    if htf_lookback > 0:
        df["ema_slow_slope"] = df["ema_slow"] - df["ema_slow"].shift(htf_lookback)
    else:
        df["ema_slow_slope"] = 0.0

    trend = np.sign(df["ema_slow_slope"])
    df["trend_bias"] = pd.Series(trend, index=df.index).replace(0, np.nan)

    return df
