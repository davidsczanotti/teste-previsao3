from __future__ import annotations

import pandas as pd


def generate_signals(df: pd.DataFrame, fast_col: str, slow_col: str, side: str = "long", exit_on_cross: bool = False) -> pd.DataFrame:
    """
    Generate EMA-cross signals on base TF.
    - side: "long" (buy on fast>slow cross) or "both" (buy on up, sell on down)
    Returns df with 'signal' column: 1 buy, -1 sell, 0 hold.
    """
    s = pd.Series(0, index=df.index)
    cross_up = (df[fast_col] > df[slow_col]) & (df[fast_col].shift(1) <= df[slow_col].shift(1))
    cross_dn = (df[fast_col] < df[slow_col]) & (df[fast_col].shift(1) >= df[slow_col].shift(1))
    s = s.mask(cross_up, 1)
    if side in ("both", "short"):
        s = s.mask(cross_dn, -1)
    df_out = df.copy()
    df_out["signal"] = s.fillna(0).astype(int)
    if exit_on_cross:
        # For convenience, mark exit signals as opposite side while in position will be handled in engine
        df_out["exit_cross"] = 0
        df_out.loc[cross_dn, "exit_cross"] = -1
        df_out.loc[cross_up, "exit_cross"] = 1
    return df_out
