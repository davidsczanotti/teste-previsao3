from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import optuna
import pandas as pd

from ...utils.data_loader import load_data
from .backtest import backtest_ema_only, EmaOnlyParams, compute_ema


def _time_splits(df: pd.DataFrame, k: int) -> List[Tuple[int, int]]:
    """Divide o dataset em k blocos contíguos [start, end) sem sobreposição."""
    n = len(df)
    if k <= 0:
        return [(0, n)]
    base = n // k
    rem = n % k
    splits: List[Tuple[int, int]] = []
    start = 0
    for i in range(k):
        length = base + (1 if i < rem else 0)
        end = start + length
        splits.append((start, end))
        start = end
    return splits


def prepare_dataset_with_reference(
    symbol: str,
    timeframe: str,
    days: int,
    use_cache_only: bool,
    ref_timeframe: Optional[str],
    ref_days: Optional[int],
    ref_ema_period: Optional[int],
) -> pd.DataFrame:
    """
    Carrega o timeframe base e (opcionalmente) anexa uma EMA de referência (TF superior).
    """
    base = (
        load_data(symbol, timeframe, days=days, use_cache_only=use_cache_only)
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if ref_timeframe and ref_ema_period:
        ref_df = load_data(symbol, ref_timeframe, days=ref_days or days, use_cache_only=use_cache_only)
        ref_df = ref_df.sort_values("Date").reset_index(drop=True).copy()
        ref_df["ref_ema"] = compute_ema(ref_df["close"].astype(float), int(ref_ema_period))
        base = pd.merge_asof(base, ref_df[["Date", "ref_ema"]], on="Date", direction="backward")
    return base


def _build_params_from_config(strategy_cfg: Dict[str, Any]) -> EmaOnlyParams:
    """Monta EmaOnlyParams a partir do bloco strategy do config.json."""
    return EmaOnlyParams(
        ema_period=int(strategy_cfg.get("ema_period", strategy_cfg.get("ema_fast_period", 21))),
        slow_ema_period=int(strategy_cfg.get("slow_ema_period", strategy_cfg.get("ema_slow_period", 55))),
        signal_mode=strategy_cfg.get("signal_mode", "ema_cross"),
        pullback_pct=float(strategy_cfg.get("pullback_pct", 0.0)),
        use_trend_filter=bool(strategy_cfg.get("use_trend_filter", False)),
        trend_filter_period=strategy_cfg.get("trend_filter_period"),
        use_cross=bool(strategy_cfg.get("use_cross", False)),
        ref_filter_enabled=bool(strategy_cfg.get("ref_filter_enabled", False)),
        ref_ema_period=strategy_cfg.get("ref_ema_period"),
        ref_buffer_pct=float(strategy_cfg.get("ref_buffer_pct", 0.0)),
        lot_size=float(strategy_cfg.get("lot_size", 0.001)),
        fee_rate=float(strategy_cfg.get("fee_pct", 0.0004)),
        sma_fast_period=strategy_cfg.get("sma_fast_period"),
        sma_mid_period=strategy_cfg.get("sma_mid_period"),
        sma_slow_period=strategy_cfg.get("sma_slow_period"),
        ema_fast_period=strategy_cfg.get("ema_fast_period"),
        ema_mid_period=strategy_cfg.get("ema_mid_period"),
        ema_slow_period=strategy_cfg.get("ema_slow_period"),
        trailing_stop_type=strategy_cfg.get("trailing_stop_type", "none"),
        atr_period=int(strategy_cfg.get("atr_period", 14)),
        atr_stop_mult=float(strategy_cfg.get("atr_stop_mult", 2.0)),
        atr_trail_mult=float(strategy_cfg.get("atr_trail_mult", 1.0)),
        breakeven_rr=float(strategy_cfg.get("breakeven_rr", 1.0)),
        percent_trailing_pct=float(strategy_cfg.get("percent_trailing_pct", 0.01)),
        ma_trail_source=strategy_cfg.get("ma_trail_source", "ema_slow"),
        ma_trail_offset_atr_mult=float(strategy_cfg.get("ma_trail_offset_atr_mult", 1.0)),
        allow_short=bool(strategy_cfg.get("allow_short", False)),
    )


def _sample_from_search_space(
    trial: optuna.Trial,
    search_space: Dict[str, Any],
) -> Dict[str, Any]:
    """Interpreta o bloco optimize.search_space do config.json (int/float/categorical)."""
    sampled: Dict[str, Any] = {}
    for name, spec in search_space.items():
        spec_type = spec.get("type")
        if spec_type == "int":
            sampled[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif spec_type == "float":
            sampled[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif spec_type == "categorical":
            sampled[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Tipo de search_space não suportado para {name}: {spec_type}")
    return sampled


def _enforce_ma_order(params: Dict[str, Any]) -> bool:
    """Garante fast < mid < slow para SMAs/EMAs, quando os três existirem."""

    def _check(prefix: str) -> bool:
        f = params.get(f"{prefix}_fast_period")
        m = params.get(f"{prefix}_mid_period")
        s = params.get(f"{prefix}_slow_period")
        if f is None or m is None or s is None:
            return True
        return f < m < s

    return _check("ema") and _check("sma")


def _make_objective_from_config(
    df_train: pd.DataFrame,
    base_params: EmaOnlyParams,
    backtest_cfg: Dict[str, Any],
    opt_cfg: Dict[str, Any],
) -> Callable[[optuna.Trial], float]:
    monthly_target_pct = float(backtest_cfg.get("monthly_target_pct", 0.01))
    search_space = opt_cfg.get("search_space", {})
    wfa_splits = max(1, int(opt_cfg.get("wfa_splits", 1)))
    penalty_per_trade = float(opt_cfg.get("penalty_per_trade", 0.0))
    dd_penalty = float(opt_cfg.get("dd_penalty", 0.0))
    objective_mode = opt_cfg.get("objective", "combo")
    initial_capital = float(backtest_cfg.get("initial_capital", 1_000.0))

    def objective(trial: optuna.Trial) -> float:
        # Copia dos parâmetros base para este trial
        params_dict = dict(base_params.__dict__)
        sampled = _sample_from_search_space(trial, search_space)
        params_dict.update(sampled)

        # Regras de ordem para fast/mid/slow
        if not _enforce_ma_order(params_dict):
            return -1e12

        params = EmaOnlyParams(**params_dict)

        total_pnl = 0.0
        n_trades_total = 0
        max_dds: List[float] = []
        sharpes: List[float] = []
        calmars: List[float] = []
        avg_monthlies: List[float] = []
        target_hits: List[float] = []

        for s_idx, e_idx in _time_splits(df_train, wfa_splits):
            seg = df_train.iloc[s_idx:e_idx].copy()
            if len(seg) < params.ema_period + 5:
                continue

            _, pnl, stats = backtest_ema_only(
                seg,
                params=params,
                initial_capital=initial_capital,
                monthly_target_pct=monthly_target_pct,
            )
            total_pnl += float(pnl)
            n_trades_total += int(stats.get("num_trades", 0))
            max_dds.append(abs(float(stats.get("max_drawdown_pct", 0.0))))
            sharpes.append(float(stats.get("sharpe", 0.0)))
            calmars.append(float(stats.get("calmar", 0.0)))
            avg_monthlies.append(float(stats.get("avg_monthly_return_pct", 0.0)))
            target_hits.append(float(stats.get("monthly_target_hit_ratio", 0.0)))

        if n_trades_total == 0:
            return -1e12

        dd_pen = dd_penalty * max(max_dds) if max_dds else 0.0
        avg_sharpe = float(np.nanmean(sharpes)) if sharpes else 0.0
        avg_calmar = float(np.nanmean(calmars)) if calmars else 0.0
        avg_monthly = float(np.nanmean(avg_monthlies)) if avg_monthlies else 0.0
        avg_hit_ratio = float(np.nanmean(target_hits)) if target_hits else 0.0

        if objective_mode == "pnl":
            score = total_pnl - penalty_per_trade * n_trades_total - dd_pen
        elif objective_mode == "sharpe":
            score = avg_sharpe
        elif objective_mode == "calmar":
            score = avg_calmar
        elif objective_mode == "monthly":
            # Foca na diferença para a meta mensal.
            score = (avg_monthly - monthly_target_pct) * 100.0 - dd_pen
        else:  # "combo"
            score = (
                total_pnl
                - penalty_per_trade * n_trades_total
                - dd_pen
                + 50.0 * avg_sharpe
                + 100.0 * max(0.0, avg_monthly - monthly_target_pct)
                + 20.0 * avg_hit_ratio
            )
        return float(score)

    return objective


def optimize_from_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Fluxo principal de otimização Optuna guiado por config.json.

    - Lê data/strategy/backtest/optimize do config.
    - Executa Optuna com search_space tipado.
    - Salva JSON de resultado + config "ativo" com melhores parâmetros.
    """
    if config_path is None:
        config_path = Path(__file__).with_name("config.json")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})
    strat_cfg = cfg.get("strategy", {})
    backtest_cfg = cfg.get("backtest", {})
    opt_cfg = cfg.get("optimize", {})

    if not opt_cfg.get("enabled", True):
        print("optimize.enabled == false; nada a otimizar.")
        return {}

    symbol = data_cfg["symbol"]
    timeframe = data_cfg["timeframe"]
    days = int(data_cfg.get("days", 365))
    ref_timeframe = data_cfg.get("ref_timeframe")
    ref_days = data_cfg.get("ref_days")
    ref_ema_period = strat_cfg.get("ref_ema_period")

    df_all = prepare_dataset_with_reference(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        use_cache_only=True,
        ref_timeframe=ref_timeframe,
        ref_days=ref_days,
        ref_ema_period=ref_ema_period,
    )

    n = len(df_all)
    train_frac = float(opt_cfg.get("train_frac", 0.7))
    split = int(n * train_frac)
    df_train = df_all.iloc[:split].copy()
    df_valid = df_all.iloc[split:].copy()

    print(
        f"[ema_only.optimize] {symbol} {timeframe} days={days} | "
        f"N={n} train={len(df_train)} valid={len(df_valid)}"
    )

    base_params = _build_params_from_config(strat_cfg)
    objective = _make_objective_from_config(df_train, base_params, backtest_cfg, opt_cfg)

    sampler = optuna.samplers.TPESampler(seed=int(opt_cfg.get("seed", 123)))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(opt_cfg.get("trials", 50)))

    print("Best params (train splits):", study.best_params)
    print("Best objective:", study.best_value)

    # Reavalia melhor solução em train/full + valid.
    best_params_dict = dict(base_params.__dict__)
    best_params_dict.update(study.best_params)
    best_params = EmaOnlyParams(**best_params_dict)

    monthly_target_pct = float(backtest_cfg.get("monthly_target_pct", 0.01))
    initial_capital = float(backtest_cfg.get("initial_capital", 1_000.0))

    _, _, stats_tr = backtest_ema_only(
        df_train.copy(),
        params=best_params,
        initial_capital=initial_capital,
        monthly_target_pct=monthly_target_pct,
    )
    _, _, stats_val = backtest_ema_only(
        df_valid.copy(),
        params=best_params,
        initial_capital=initial_capital,
        monthly_target_pct=monthly_target_pct,
    )

    # Sinalização de significância.
    min_trades = int(backtest_cfg.get("min_trades_for_significance", 30))
    min_candles = int(backtest_cfg.get("min_candles_for_significance", 2000))
    sig_flags: Dict[str, Any] = {
        "train_is_significant": (
            stats_tr.get("num_trades", 0) >= min_trades and len(df_train) >= min_candles
        ),
        "valid_is_significant": (
            stats_val.get("num_trades", 0) >= min_trades and len(df_valid) >= min_candles
        ),
        "train_num_trades": int(stats_tr.get("num_trades", 0)),
        "valid_num_trades": int(stats_val.get("num_trades", 0)),
        "train_num_candles": len(df_train),
        "valid_num_candles": len(df_valid),
        "min_trades_for_significance": min_trades,
        "min_candles_for_significance": min_candles,
    }

    # Resumo de objetivo mensal para auditoria.
    print(
        "[ema_only.optimize] Objetivo mensal: "
        f"{monthly_target_pct:.2%} | "
        f"Train avg: {stats_tr.get('avg_monthly_return_pct', 0.0):.2%} "
        f"(hit {stats_tr.get('monthly_target_hit_ratio', 0.0):.1%}) | "
        f"Valid avg: {stats_val.get('avg_monthly_return_pct', 0.0):.2%} "
        f"(hit {stats_val.get('monthly_target_hit_ratio', 0.0):.1%})"
    )

    rec = {
        "strategy": "ema_only",
        "symbol": symbol,
        "interval": timeframe,
        "days": days,
        "train_frac": train_frac,
        "wfa_splits": int(opt_cfg.get("wfa_splits", 1)),
        "objective": opt_cfg.get("objective", "combo"),
        "monthly_target_pct": monthly_target_pct,
        "search_space": opt_cfg.get("search_space", {}),
        "best_trial_value": study.best_value,
        "best_trial_params": study.best_params,
        "best_strategy_params": {
            k: getattr(best_params, k)
            for k in best_params.__dict__.keys()
            if not k.startswith("_")
        },
        "train_metrics": stats_tr,
        "valid_metrics": stats_val,
        "significance": sig_flags,
    }

    outdir = Path(opt_cfg.get("outdir", "src/strategies/ema_only/reports/optimize"))
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base_name = f"ema_only_optuna_{symbol}_{timeframe}_{ts}"
    json_path = outdir / f"{base_name}.json"
    json_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    # Config "ativa" resumida (para backtests futuros).
    active_dir = outdir / "active"
    active_dir.mkdir(parents=True, exist_ok=True)
    active_path = active_dir / f"ema_only_{symbol}_{timeframe}.json"
    active_payload = {
        "strategy": "ema_only",
        "symbol": symbol,
        "interval": timeframe,
        "monthly_target_pct": monthly_target_pct,
        "best_strategy_params": rec["best_strategy_params"],
    }
    active_path.write_text(json.dumps(active_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved optimize result: {json_path}")
    print(f"Active config updated: {active_path}")

    return rec


def main() -> None:
    optimize_from_config()


if __name__ == "__main__":
    main()
