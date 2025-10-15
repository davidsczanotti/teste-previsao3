#!/usr/bin/env python3
"""
Auditoria: diff entre a lógica do backtest e os snapshots do modo live.

O objetivo é identificar candles onde o backtest teria entrado (com base no
candle seguinte) mas o modo live não registrou a entrada, e vice‑versa.

Entradas padrão:
  - live_csv:  reports/live/ALBROOKS_BTCUSDT_1m.csv
  - trades_csv: reports/live/ALBROOKS_BTCUSDT_1m_trades.csv (opcional)

Saídas:
  - imprime resumo no stdout
  - opcionalmente salva um CSV enriquecido com colunas:
      would_enter_backtest, missed_entry, spurious_entry

Uso:
  python scripts/albrooks_audit_diff.py \
    --live-csv reports/live/ALBROOKS_BTCUSDT_1m.csv \
    --trades-csv reports/live/ALBROOKS_BTCUSDT_1m_trades.csv \
    --out reports/live/ALBROOKS_BTCUSDT_1m_audit.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([False] * 0)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )


def load_live_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    # Normaliza tipos
    df["candle_time"] = pd.to_datetime(df["candle_time"], errors="coerce")
    for col in [
        "is_inside_bar",
        "uptrend",
        "downtrend",
        "pullback_long_ok",
        "pullback_short_ok",
        "allow_long",
        "allow_short",
        "adx_ok",
        "atr_ok",
        "deviation_ok",
        "setup_ok",
        "entry_confirmed",
    ]:
        if col in df.columns:
            df[col] = _to_bool_series(df[col])
        else:
            df[col] = False

    # Converte numéricos relevantes (ignora erros)
    for col in [
        "high_last_closed",
        "low_last_closed",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("candle_time").reset_index(drop=True)


def load_trades_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    # Filtra apenas entradas
    df = df[df["type"] == "entry"].copy()
    if df.empty:
        return df
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["side"] = df["side"].astype(str).str.lower()
    return df[["event_time", "side"]]


def compute_expected_signal(row: pd.Series) -> str:
    if not row.get("is_inside_bar", False):
        return "hold"
    cond_base = bool(row.get("deviation_ok", False) and row.get("adx_ok", False) and row.get("atr_ok", False))
    if not cond_base:
        return "hold"
    if row.get("allow_long", False) and row.get("uptrend", False) and row.get("pullback_long_ok", False):
        return "buy"
    if row.get("allow_short", False) and row.get("downtrend", False) and row.get("pullback_short_ok", False):
        return "sell"
    return "hold"


def build_audit_df(df_live: pd.DataFrame, trades: pd.DataFrame | None) -> pd.DataFrame:
    df = df_live.copy()
    n = len(df)
    signals = []
    would_enter = []
    for i in range(n):
        row = df.iloc[i]
        sig = compute_expected_signal(row)
        signals.append(sig)
        enter = False
        if i + 1 < n:
            nxt = df.iloc[i + 1]
            if sig == "buy":
                # backtest entra se a máxima do candle seguinte >= máxima do candle inside
                enter = (
                    pd.notna(nxt.get("high_last_closed"))
                    and pd.notna(row.get("high_last_closed"))
                    and float(nxt["high_last_closed"]) >= float(row["high_last_closed"])
                )
            elif sig == "sell":
                enter = (
                    pd.notna(nxt.get("low_last_closed"))
                    and pd.notna(row.get("low_last_closed"))
                    and float(nxt["low_last_closed"]) <= float(row["low_last_closed"])
                )
        would_enter.append(enter)

    df["expected_signal"] = signals
    df["would_enter_backtest"] = would_enter

    # Mapeia entradas do live (trades CSV)
    if trades is not None and not trades.empty:
        key = trades.apply(lambda r: (r["event_time"], r["side"]), axis=1)
        trade_set = set(key.to_list())
    else:
        trade_set = set()

    entry_side = df["expected_signal"].map({"buy": "long", "sell": "short"})
    has_trade = df.apply(lambda r: (r["candle_time"], entry_side.loc[r.name]) in trade_set if entry_side.loc[r.name] in {"long", "short"} else False, axis=1)
    df["has_trade"] = has_trade

    df["missed_entry"] = (df["would_enter_backtest"]) & (~df["has_trade"]) & (df["expected_signal"].isin(["buy", "sell"]))
    df["spurious_entry"] = (~df["would_enter_backtest"]) & (df["has_trade"]) & (df["expected_signal"].isin(["buy", "sell"]))

    # Divergência de polling: candle que fecharia entrada mas 'entry_confirmed' ficou False no snapshot
    if "entry_confirmed" in df.columns:
        df["polling_miss"] = df["would_enter_backtest"] & (~_to_bool_series(df["entry_confirmed"]))

    return df


def main():
    ap = argparse.ArgumentParser(description="Audita divergências entre backtest (teórico) e live (executado)")
    ap.add_argument("--live-csv", default="reports/live/ALBROOKS_BTCUSDT_1m.csv")
    ap.add_argument("--trades-csv", default="reports/live/ALBROOKS_BTCUSDT_1m_trades.csv")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=20, help="Quantas divergências imprimir")
    args = ap.parse_args()

    live_path = Path(args.live_csv)
    trades_path = Path(args.trades_csv)

    df_live = load_live_csv(live_path)
    if df_live.empty:
        print("Arquivo live CSV vazio ou inválido.")
        return
    df_trades = load_trades_csv(trades_path)

    audit = build_audit_df(df_live, df_trades)

    total = len(audit)
    missed = int(audit["missed_entry"].sum()) if "missed_entry" in audit.columns else 0
    spurious = int(audit["spurious_entry"].sum()) if "spurious_entry" in audit.columns else 0
    polling_miss = int(audit.get("polling_miss", pd.Series(dtype=bool)).sum()) if "polling_miss" in audit.columns else 0

    print(f"Total linhas: {total}")
    print(f"Possíveis entradas perdidas (backtest teria entrado): {missed}")
    print(f"Entradas não justificadas (live entrou mas backtest não): {spurious}")
    if "polling_miss" in audit.columns:
        print(f"Sinais confirmados no candle seguinte mas não no snapshot (indicativo de perda por polling): {polling_miss}")

    if missed:
        print("\nExemplos de missed_entry:")
        cols = [
            "candle_time",
            "expected_signal",
            "high_last_closed",
            "low_last_closed",
            "would_enter_backtest",
            "has_trade",
        ]
        print(audit.loc[audit["missed_entry"], cols].head(args.limit).to_string(index=False))

    if spurious:
        print("\nExemplos de spurious_entry:")
        cols = [
            "candle_time",
            "expected_signal",
            "high_last_closed",
            "low_last_closed",
            "would_enter_backtest",
            "has_trade",
        ]
        print(audit.loc[audit["spurious_entry"], cols].head(args.limit).to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        audit.to_csv(out_path, index=False)
        print(f"\nArquivo de auditoria salvo em: {out_path}")


if __name__ == "__main__":
    main()

