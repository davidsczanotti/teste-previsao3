"""Ferramentas para rotular regimes e amostrar blocos temporais balanceados.

Fluxo esperado:
- `label_regime_daily(ref_df)`: usa série diária (ref_ema) para rotular bull/bear/flat.
- `attach_regime(base_df, daily_regime)`: projeta o regime diário para candles 4h via merge_asof.
- `make_blocks(base_df, block_months)`: cria blocos contínuos de `block_months` meses com rótulo (moda do regime).
- `sample_blocks(blocks, num_blocks, seed)`: escolhe blocos com pesos inversos à frequência de regime.
- `concat_blocks(df, blocks)`: concatena os blocos escolhidos, preservando ordem interna e marcando reinícios.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


Regime = str


@dataclass
class Block:
    start: pd.Timestamp
    end: pd.Timestamp
    regime: Regime
    idx_start: int
    idx_end: int


def label_regime_daily(ref_df: pd.DataFrame, bull: float = 0.01, bear: float = -0.01, lookback: int = 30) -> pd.DataFrame:
    """Adiciona coluna `regime` na série diária baseada no slope percentual da ref_ema.

    bull/bear: thresholds de retorno acumulado em `lookback` dias.
    """

    if ref_df.empty:
        return ref_df.assign(regime="flat")

    ref = ref_df.copy().sort_values("Date").reset_index(drop=True)
    base_col = "ref_ema" if "ref_ema" in ref.columns else "close"
    pct = ref[base_col].astype(float).pct_change(periods=lookback)
    regime = np.where(pct > bull, "bull", np.where(pct < bear, "bear", "flat"))
    ref["regime"] = regime
    return ref


def attach_regime(base_df: pd.DataFrame, ref_daily: pd.DataFrame) -> pd.DataFrame:
    """Projeta regime diário para a série 4h via merge_asof."""

    if base_df.empty or ref_daily.empty or "regime" not in ref_daily.columns:
        return base_df.assign(regime="flat")
    base = base_df.copy().sort_values("Date").reset_index(drop=True)
    ref_sorted = ref_daily[["Date", "regime"]].sort_values("Date").reset_index(drop=True)
    merged = pd.merge_asof(base, ref_sorted, on="Date", direction="backward")
    merged["regime"].fillna("flat", inplace=True)
    return merged


def make_blocks(base_df: pd.DataFrame, block_months: int = 6) -> List[Block]:
    """Cria blocos contínuos de tamanho fixo (em meses) e rótulo pela moda de regime."""

    if base_df.empty:
        return []
    base = base_df.copy().sort_values("Date").reset_index(drop=True)
    blocks: List[Block] = []

    cur_start = pd.to_datetime(base["Date"].iloc[0])
    last_date = pd.to_datetime(base["Date"].iloc[-1])
    delta = pd.DateOffset(months=block_months)

    while cur_start < last_date:
        cur_end = cur_start + delta
        mask = (pd.to_datetime(base["Date"]) >= cur_start) & (pd.to_datetime(base["Date"]) < cur_end)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            cur_start = cur_end
            continue
        idx_start = int(idx[0])
        idx_end = int(idx[-1])
        regime_series = base.loc[mask, "regime"] if "regime" in base.columns else pd.Series([], dtype=str)
        if regime_series.empty:
            regime_label = "flat"
        else:
            regime_label = regime_series.mode().iat[0]
        blocks.append(Block(start=cur_start, end=cur_end, regime=str(regime_label), idx_start=idx_start, idx_end=idx_end))
        cur_start = cur_end
    return blocks


def sample_blocks(blocks: Sequence[Block], num_blocks: int, seed: Optional[int] = None) -> List[Block]:
    """Amostra blocos com pesos inversos à frequência de regime."""

    if not blocks or num_blocks <= 0:
        return []
    rng = random.Random(seed)
    regimes = [b.regime for b in blocks]
    counts: Dict[Regime, int] = {}
    for r in regimes:
        counts[r] = counts.get(r, 0) + 1
    weights = []
    for b in blocks:
        freq = counts.get(b.regime, 1)
        weights.append(1.0 / freq)
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    chosen: List[Block] = []
    for _ in range(num_blocks):
        idx = weighted_choice(probs, rng)
        chosen.append(blocks[idx])
    return chosen


def weighted_choice(probs: Sequence[float], rng: random.Random) -> int:
    """Retorna um índice proporcional às probabilidades normalizadas."""

    r = rng.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            return i
    return len(probs) - 1


def concat_blocks(base_df: pd.DataFrame, features: pd.DataFrame, blocks: Sequence[Block]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Concatena blocos selecionados preservando ordem interna e marca block_reset."""

    if not blocks:
        return base_df, features
    base = base_df.reset_index(drop=True)
    feats = features.reset_index(drop=True)
    parts_base = []
    parts_feats = []
    for i, b in enumerate(blocks):
        slice_base = base.iloc[b.idx_start : b.idx_end + 1].copy()
        slice_feats = feats.iloc[b.idx_start : b.idx_end + 1].copy()
        slice_feats["block_reset"] = 0.0
        if not slice_feats.empty:
            slice_feats.iloc[0, slice_feats.columns.get_loc("block_reset")] = 1.0
        parts_base.append(slice_base)
        parts_feats.append(slice_feats)
    out_base = pd.concat(parts_base, ignore_index=True)
    out_feats = pd.concat(parts_feats, ignore_index=True)
    return out_base, out_feats
