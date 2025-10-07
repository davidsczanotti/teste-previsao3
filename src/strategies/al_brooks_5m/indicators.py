import pandas as pd
import pandas_ta as ta


def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Calcula e adiciona todos os indicadores necessários ao DataFrame."""
    df["ema_fast"] = ta.ema(df["close"], length=params["ema_fast_period"])
    df["ema_medium"] = ta.ema(df["close"], length=params["ema_medium_period"])
    df["ema_slow"] = ta.ema(df["close"], length=params["ema_slow_period"])

    df["is_inside_bar"] = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))

    df["avg_deviation_pct"] = abs((df["close"] - df["ema_slow"]) / df["ema_slow"]) * 100

    return df
