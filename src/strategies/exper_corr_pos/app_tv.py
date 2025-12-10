from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from ...utils.data_loader import load_data as _load_cached
from .utils_cfg import enabled_expert_names


ROOT = Path(__file__).resolve().parents[3]
STRATEGIES_DIR = ROOT / "src" / "strategies"


@dataclass
class StrategyMeta:
    name: str
    config_path: Path


def _list_strategies() -> List[StrategyMeta]:
    metas: List[StrategyMeta] = []
    for path in STRATEGIES_DIR.iterdir():
        if path.is_dir():
            cfg = path / "config.json"
            if cfg.exists():
                metas.append(StrategyMeta(name=path.name, config_path=cfg))
    return sorted(metas, key=lambda m: m.name.lower())


def _load_config(cfg_path: Path) -> Dict[str, Any]:
    return json.loads(cfg_path.read_text())


def _hull_moving_average(series: pd.Series, period: int) -> pd.Series:
    period = max(2, int(period))
    half = max(1, period // 2)
    sqrt_p = max(1, int(math.sqrt(period)))
    wma_half = series.rolling(half).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
    wma_full = series.rolling(period).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
    raw = 2 * wma_half - wma_full
    hma = raw.rolling(sqrt_p).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
    return hma


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    closes = out["close"]
    bb_window = 20
    bb_std = 2.0
    ma = closes.rolling(bb_window).mean()
    std = closes.rolling(bb_window).std()
    out["bb_middle"] = ma
    out["bb_upper"] = ma + bb_std * std
    out["bb_lower"] = ma - bb_std * std
    out["hma"] = _hull_moving_average(closes, 9)
    return out


def _load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    # Suporta formatos de config diferentes entre estratégias:
    # - exper_corr_pos: base_symbol + lookback_days
    # - ema_only e outras: symbol + days
    symbol = data_cfg.get("base_symbol") or data_cfg.get("symbol", "BTCUSDT")
    timeframe = data_cfg.get("timeframe", "1d")
    days = int(data_cfg.get("lookback_days") or data_cfg.get("days", 365))
    df = _load_cached(symbol, timeframe, days=days, use_cache_only=True)
    if df.empty:
        raise RuntimeError(f"Nenhum dado disponível no cache para {symbol}@{timeframe}. Rode populate_cache.")
    return df


def _build_ema_only_payload(cfg: Dict[str, Any], progress: float) -> Dict[str, Any]:
    """
    Gera payload TV para a estratégia ema_only:
    - candles 4h (ou timeframe configurado)
    - EMAs (fast/mid/slow) + ref_ema
    - trades long-only reconstruídos a partir dos sinais.
    """
    # Import tardio para evitar dependência circular em contextos de teste
    from src.strategies.ema_only.backtest import (
        load_data_with_ref as ema_load_data_with_ref,
        calculate_mas as ema_calculate_mas,
        generate_signals as ema_generate_signals,
    )

    data_cfg = cfg.get("data", {})
    symbol = str(data_cfg.get("symbol", "BTCUSDT"))
    timeframe = str(data_cfg.get("timeframe", "4h"))

    df = ema_load_data_with_ref(cfg)
    df = ema_calculate_mas(df, cfg)
    df = ema_generate_signals(df, cfg)
    df = df.sort_values("Date")
    df = df.set_index(pd.to_datetime(df["Date"], utc=True)).drop(columns=["Date"])

    cutoff_ts: Optional[pd.Timestamp] = None
    if progress < 1.0 and not df.empty:
        total = len(df)
        upto = max(1, int(total * progress))
        cutoff_ts = df.index[upto - 1]
        df = df.iloc[:upto]

    candles_df = df[["open", "high", "low", "close"]]

    trades = _build_ema_only_trades(df, cfg)
    if cutoff_ts is not None:
        trades = [t for t in trades if t.get("time", 0) <= int(cutoff_ts.timestamp())]

    init_eq = float(cfg.get("backtest", {}).get("initial_capital", 1000.0))
    pnl_sum = sum(t.get("pnl", 0.0) for t in trades)
    cash = init_eq + pnl_sum

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": _candles_payload(candles_df),
        # Mantém chaves existentes para o front, mas vazias
        "bb_upper": [],
        "bb_middle": [],
        "bb_lower": [],
        "hma": [],
        # Overlays específicos da ema_only
        "ema_fast": _overlay_payload(df, "ema_fast") if "ema_fast" in df.columns else [],
        "ema_mid": _overlay_payload(df, "ema_mid") if "ema_mid" in df.columns else [],
        "ema_slow": _overlay_payload(df, "ema_slow") if "ema_slow" in df.columns else [],
        "ref_ema": _overlay_payload(df, "ref_ema") if "ref_ema" in df.columns else [],
        "trades": trades,
        "stats": {
            "init_equity": init_eq,
            "cash": cash,
            "pnl": pnl_sum,
            "living_cost": 300.0,
            "debt_remaining": max(0.0, 300.0 - pnl_sum),
            "bonus_value": 0.0,
            "bonus_pct": 0.0,
            "bonus_cap_pct": 0.0,
            "days_remaining": 0,
            "days_total": len(df),
            "mood": "neutral",
            "mood_count": {"happy": 0, "sad": 0, "neutral": len(trades)},
            "experts": [],
            "gate_top_k": None,
            "allow_short": False,
            "life_unit": 100.0,
            "lives_total": 10,
            "lives_remaining": 10,
            "lives_lost": 0,
        },
    }
    return payload


def _load_trades(config: Dict[str, Any], strategy_dir: Path) -> List[Dict[str, Any]]:
    trade_cfg = config.get("reports", {}).get("trade_ledger", {}) if isinstance(config.get("reports"), dict) else {}
    ledger_path = trade_cfg.get("path") or strategy_dir / "reports" / "train" / "trade_ledger.csv"
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return []
    df = pd.read_csv(ledger_path)
    trades: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            entry_ts = pd.to_datetime(row["entry_ts"])
            price_entry = float(row["entry_price"])
            price_exit = float(row["exit_price"])
            side_raw = str(row["side"]).lower()
            side = "long" if "long" in side_raw else "short"
            trades.append(
                {
                    "time": int(entry_ts.timestamp()),
                    "entry": price_entry,
                    "exit": price_exit,
                    "side": side,
                    "pnl": float(row.get("pnl_net", row.get("pnl", 0.0))),
                }
            )
        except Exception:
            continue
    return trades


def _candles_payload(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "time": int(ts.timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for ts, row in df.iterrows()
    ]


def _overlay_payload(df: pd.DataFrame, col: str) -> List[Dict[str, Any]]:
    out = []
    for ts, value in df[col].items():
        if pd.isna(value):
            continue
        out.append({"time": int(ts.timestamp()), "value": float(value)})
    return out


def _build_ema_only_trades(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Reconstrói trades long-only para ema_only a partir da coluna 'signal'.

    - Entrada: signal == 1 e posição zerada.
    - Saída: signal == -1 e posição > 0.
    """
    strat = cfg.get("strategy", {})
    lot = float(strat.get("lot_size", 1.0))
    trades: List[Dict[str, Any]] = []
    position = 0.0
    entry_price = 0.0
    entry_ts: Optional[int] = None

    for ts, row in df.iterrows():
        sig = row.get("signal", 0)
        close = float(row["close"])
        if sig == 1 and position == 0.0:
            position = lot
            entry_price = close
            entry_ts = int(ts.timestamp())
        elif sig == -1 and position > 0.0:
            exit_price = close
            pnl = (exit_price - entry_price) * position
            trades.append(
                {
                    "time": entry_ts or int(ts.timestamp()),
                    "entry": entry_price,
                    "exit": exit_price,
                    "side": "long",
                    "pnl": pnl,
                }
            )
            position = 0.0
            entry_price = 0.0
            entry_ts = None

    return trades


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/tv")
    def tv() -> str:
        strategies = _list_strategies()
        strategy = request.args.get("strategy") or "exper_corr_pos"
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        return render_template(
            "tv.html",
            strategies=[m.name for m in strategies],
            strategy=strategy,
            timeframes=timeframes,
        )

    @app.route("/api/tv_data")
    def api_tv_data():
        strategy = request.args.get("strategy") or "exper_corr_pos"
        timeframe = request.args.get("timeframe")
        try:
            progress = float(request.args.get("progress", "1"))
        except ValueError:
            progress = 1.0
        progress = min(1.0, max(0.0, progress))
        strategies = {m.name: m for m in _list_strategies()}
        meta = strategies.get(strategy)
        if meta is None:
            return jsonify({"error": f"Estratégia {strategy} não encontrada"}), 404
        cfg = _load_config(meta.config_path)
        if timeframe:
            cfg.setdefault("data", {})["timeframe"] = timeframe

        # Caminho específico para ema_only: EMAs + trades long-only
        if strategy == "ema_only":
            payload = _build_ema_only_payload(cfg, progress)
            return jsonify(payload)

        df = _load_dataset(cfg)
        df = df.set_index(pd.to_datetime(df["Date"], utc=True)).drop(columns=["Date"])
        df = _compute_indicators(df)
        trades = _load_trades(cfg, meta.config_path.parent)
        if progress < 1.0 and not df.empty:
            total = len(df)
            upto = max(1, int(total * progress))
            cutoff_ts = df.index[upto - 1]
            df = df.iloc[:upto]
            trades = [t for t in trades if t.get("time", 0) <= int(cutoff_ts.timestamp())]
        env_cfg = cfg.get("env", {})
        init_eq = float(env_cfg.get("init_equity", 1000.0))
        living_cost = float(env_cfg.get("living_cost_per_episode", 0.0))
        bonus_step = float(env_cfg.get("tier_bonus_step_pct", 0.0))
        bonus_cap = float(env_cfg.get("tier_bonus_max_pct", 0.0))
        bonus_cap_pct = float(env_cfg.get("tier_bonus_cap_pnl_pct", 0.0))
        pnl_sum = sum(t.get("pnl", 0.0) for t in trades)
        cash = init_eq + pnl_sum - living_cost
        profit = max(0.0, pnl_sum)
        profit_pct = profit / init_eq if init_eq > 0 else 0.0
        tier = 0.0
        if bonus_step > 0.0 and profit_pct > 0.0:
            tier = min(math.floor(profit_pct / bonus_step) * bonus_step, bonus_cap)
        bonus_value = profit * tier
        if bonus_cap_pct > 0.0:
            bonus_value = min(bonus_value, profit * bonus_cap_pct)
        # Conceito de dívida: alvo mínimo de lucro no mês
        debt_target = living_cost if living_cost > 0.0 else 300.0
        debt_remaining = max(0.0, debt_target - pnl_sum)
        debt_paid = pnl_sum >= debt_target
        window_bars = int(env_cfg.get("window_bars", 0)) or len(df)
        experts = enabled_expert_names(cfg)
        gate_top_k = cfg.get("model", {}).get("top_k")
        allow_short = bool(env_cfg.get("allow_short", True))
        # Conceito de vidas: cada 100 USD de perda consome 1 vida
        life_unit = 100.0
        lives_total = 10
        lives_lost = 0
        if init_eq > 0.0:
            lives_lost = int(max(0.0, (init_eq - cash) / life_unit))
        lives_lost = min(lives_total, max(0, lives_lost))
        lives_remaining = max(0, lives_total - lives_lost)
        mood = "neutral"
        mood_counts = {"happy": 0, "sad": 0, "neutral": 0}
        if trades:
            last_pnl = trades[-1].get("pnl", 0.0)
            mood = "happy" if last_pnl > 0 else "sad" if last_pnl < 0 else "neutral"
            for t in trades:
                pnl = t.get("pnl", 0.0)
                if pnl > 0:
                    mood_counts["happy"] += 1
                elif pnl < 0:
                    mood_counts["sad"] += 1
                else:
                    mood_counts["neutral"] += 1
        payload = {
            "symbol": cfg.get("data", {}).get("base_symbol", "BTCUSDT"),
            "timeframe": cfg.get("data", {}).get("timeframe", "1d"),
            "candles": _candles_payload(df),
            "bb_upper": _overlay_payload(df, "bb_upper"),
            "bb_middle": _overlay_payload(df, "bb_middle"),
            "bb_lower": _overlay_payload(df, "bb_lower"),
            "hma": _overlay_payload(df, "hma"),
            "trades": trades,
            "stats": {
                "init_equity": init_eq,
                "cash": cash,
                "pnl": pnl_sum,
                "living_cost": living_cost,
                 "debt_target": debt_target,
                "debt_remaining": debt_remaining,
                "debt_paid": debt_paid,
                "bonus_pct": tier,
                "bonus_value": bonus_value,
                "bonus_cap_pct": bonus_cap_pct,
                "days_remaining": max(0, (window_bars - len(df))),
                "days_total": window_bars,
                "mood": mood,
                "mood_count": mood_counts,
                "experts": experts,
                "gate_top_k": gate_top_k,
                "allow_short": allow_short,
                "life_unit": life_unit,
                "lives_total": lives_total,
                "lives_remaining": lives_remaining,
                "lives_lost": lives_lost,
            },
        }
        return jsonify(payload)

    return app


def main() -> None:
    app = create_app()
    port = int(os.environ.get("PORT", "8001"))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
