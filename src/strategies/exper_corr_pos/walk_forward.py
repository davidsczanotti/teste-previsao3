from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from numpy.random import default_rng, Generator

from .data import load_primary_series, load_confirm_series, prepare_dataset
from .env import BTCMixtureEnv, EnvConfig
from .models import MoEPolicy, PPOConfig
from .utils_cfg import build_policy, bars_for_days, hours_per_bar
from .trainer import PPOTrainer
from ...utils.metrics import calculate_metrics


@dataclass
class WFWindow:
    train_start: int
    train_end: int
    val_start: int
    val_end: int


def build_windows(length: int, train_len: int, val_len: int, step: int) -> List[WFWindow]:
    windows: List[WFWindow] = []
    start = 0
    while start + train_len + val_len < length:
        windows.append(
            WFWindow(
                train_start=start,
                train_end=start + train_len,
                val_start=start + train_len,
                val_end=start + train_len + val_len,
            )
        )
        start += step
    return windows


DEFAULT_CONFIG = Path("src/strategies/exper_corr_pos/config.json")


def eval_env_greedy(
    policy: MoEPolicy,
    env: BTCMixtureEnv,
    *,
    device: torch.device,
    bar_hours: float,
) -> Dict[str, Any]:
    obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    rewards = []
    done = False
    trades: List[Dict[str, Any]] = []
    ruined = False
    while not done:
        with torch.no_grad():
            dist, _, _ = policy(obs.unsqueeze(0))
            action = torch.argmax(dist.probs, dim=-1).item()
        next_obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if info.get("trade_closed"):
            trades.append(
                {
                    "pnl": float(info.get("trade_pnl", 0.0)),
                    "duration_bars": int(info.get("trade_bars", 0)),
                }
            )
        if info.get("ruined"):
            ruined = True
        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    metrics = calculate_metrics(trades)
    durs = [t.get("duration_bars", 0) for t in trades if t.get("duration_bars", 0) > 0]
    avg_dur_bars = float(np.mean(durs)) if durs else 0.0
    avg_dur_hours = avg_dur_bars * bar_hours
    return {
        "pnl": float(np.sum(rewards)),
        "equity_end": float(info.get("equity", 0.0)),
        "trades": int(len(trades)),
        "trade_list": trades,
        "win_rate": float(metrics.get("win_rate", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "total_pnl_trades": float(metrics.get("total_pnl", 0.0)),
        "avg_win": float(metrics.get("avg_win", 0.0)),
        "avg_loss": float(metrics.get("avg_loss", 0.0)),
        "avg_trade_duration_bars": avg_dur_bars,
        "avg_trade_duration_hours": avg_dur_hours,
        "ruined": bool(ruined),
    }


def perturb_prices(price_df: pd.DataFrame, rng: Generator, noise_std: float) -> pd.DataFrame:
    if noise_std <= 0:
        return price_df.copy()
    price_df = price_df.reset_index(drop=True)
    if len(price_df) < 2:
        return price_df.copy()
    close = price_df["close"].to_numpy(dtype=np.float64)
    log_returns = np.diff(np.log(close + 1e-9))
    noise = rng.normal(0.0, noise_std, size=log_returns.shape)
    log_path = np.concatenate(
        ([np.log(close[0])], np.log(close[0]) + np.cumsum(log_returns + noise))
    )
    new_close = np.exp(log_path)

    new_open = np.roll(new_close, 1)
    new_open[0] = price_df["open"].iloc[0]

    range_ratio = (price_df["high"] - price_df["low"]).to_numpy(dtype=np.float64) / (close + 1e-9)
    range_ratio = np.clip(np.nan_to_num(range_ratio, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 5.0)

    base_high = np.maximum(new_close, new_open)
    base_low = np.minimum(new_close, new_open)

    new_high = base_high * (1.0 + range_ratio)
    new_low = base_low * (1.0 - range_ratio)

    new_high = np.maximum(new_high, base_high)
    new_low = np.minimum(new_low, base_low)

    perturbed = price_df.copy()
    perturbed["close"] = new_close
    perturbed["open"] = new_open
    perturbed["high"] = new_high
    perturbed["low"] = new_low
    # volume preservado
    return perturbed


def perturb_features(feat_df: pd.DataFrame, rng: Generator, noise_std: float) -> pd.DataFrame:
    if noise_std <= 0:
        return feat_df.copy()
    feat_std = feat_df.std().replace(0.0, 1.0)
    noise = rng.normal(0.0, noise_std, size=feat_df.shape)
    perturbed = feat_df + noise * feat_std.values
    # Future-proof forward fill (avoid deprecated fillna(method="ffill"))
    return perturbed.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def run_monte_carlo_analysis(
    policy: MoEPolicy,
    base_prices: pd.DataFrame,
    base_feats: pd.DataFrame,
    env_cfg: EnvConfig,
    norm_mean: pd.Series,
    norm_std: pd.Series,
    timestamps: List,
    *,
    simulations: int,
    feature_noise_std: float,
    price_noise_std: float,
    device: torch.device,
    bar_hours: float,
    seed: int,
) -> Tuple[Dict[str, Any], List[float], List[float], int]:
    if simulations <= 0:
        return {}, [], [], 0
    rng = default_rng(seed)
    equity_samples: List[float] = []
    pnl_samples: List[float] = []
    ruin_count = 0
    for idx in range(simulations):
        prices = perturb_prices(base_prices, rng, price_noise_std)
        feats = perturb_features(base_feats, rng, feature_noise_std)
        env = BTCMixtureEnv(
            prices,
            feats,
            env_cfg,
            norm_mean=norm_mean,
            norm_std=norm_std,
            timestamps=timestamps,
        )
        res = eval_env_greedy(policy, env, device=device, bar_hours=bar_hours)
        equity_samples.append(res.get("equity_end", 0.0))
        pnl_samples.append(res.get("pnl", 0.0))
        if res.get("ruined"):
            ruin_count += 1
    eq_arr = np.array(equity_samples, dtype=np.float64)
    pnl_arr = np.array(pnl_samples, dtype=np.float64)
    summary = {
        "simulations": simulations,
        "equity_end": {
            "mean": float(eq_arr.mean()),
            "std": float(eq_arr.std()),
            "p05": float(np.percentile(eq_arr, 5)),
            "p50": float(np.percentile(eq_arr, 50)),
            "p95": float(np.percentile(eq_arr, 95)),
        },
        "pnl": {
            "mean": float(pnl_arr.mean()),
            "std": float(pnl_arr.std()),
            "p05": float(np.percentile(pnl_arr, 5)),
            "p50": float(np.percentile(pnl_arr, 50)),
            "p95": float(np.percentile(pnl_arr, 95)),
        },
        "ruin_rate": float(ruin_count / simulations),
        "ruin_count": int(ruin_count),
        "feature_noise_std": feature_noise_std,
        "price_noise_std": price_noise_std,
    }
    return summary, equity_samples, pnl_samples, ruin_count


def run_cost_sensitivity(
    policy: MoEPolicy,
    base_prices: pd.DataFrame,
    base_feats: pd.DataFrame,
    env_cfg: EnvConfig,
    norm_mean: pd.Series,
    norm_std: pd.Series,
    timestamps: List,
    *,
    factors: List[float],
    device: torch.device,
    bar_hours: float,
) -> List[Dict[str, Any]]:
    results = []
    for factor in factors:
        stress_cfg = EnvConfig(**env_cfg.__dict__)
        stress_cfg.fee_pct = env_cfg.fee_pct * factor
        stress_cfg.slippage_pct = env_cfg.slippage_pct * factor
        env = BTCMixtureEnv(
            base_prices,
            base_feats,
            stress_cfg,
            norm_mean=norm_mean,
            norm_std=norm_std,
            timestamps=timestamps,
        )
        res = eval_env_greedy(policy, env, device=device, bar_hours=bar_hours)
        results.append({"factor": factor, "metrics": res})
    return results


def run_lag_sensitivity(
    policy: MoEPolicy,
    base_prices: pd.DataFrame,
    base_feats: pd.DataFrame,
    env_cfg: EnvConfig,
    norm_mean: pd.Series,
    norm_std: pd.Series,
    timestamps: List,
    *,
    max_lag: int,
    device: torch.device,
    bar_hours: float,
) -> List[Dict[str, Any]]:
    results = []
    max_lag = max(1, int(max_lag))
    for lag in range(1, max_lag + 1):
        lag_feats = base_feats.shift(lag).dropna()
        if lag_feats.empty:
            continue
        lag_prices = base_prices.iloc[-len(lag_feats) :]
        lag_ts = timestamps[-len(lag_feats) :] if timestamps else []
        env = BTCMixtureEnv(
            lag_prices,
            lag_feats,
            env_cfg,
            norm_mean=norm_mean,
            norm_std=norm_std,
            timestamps=lag_ts,
        )
        res = eval_env_greedy(policy, env, device=device, bar_hours=bar_hours)
        results.append({"lag": lag, "metrics": res})
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation for MoE PPO agent")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to JSON configuration (default: src/strategies/exper_corr_pos/config.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())
    wf_cfg = cfg.get("walk_forward", {})
    env_cfg = EnvConfig(**cfg.get("env", {}))
    data_cfg = cfg.get("data", {})
    ppo_cfg = PPOConfig(**cfg.get("ppo", {}))

    primary_df = load_primary_series(cfg)
    confirm_df = load_confirm_series(cfg)
    dataset = prepare_dataset(primary_df, config=cfg, confirm_df=confirm_df)
    price_cols = ["open", "high", "low", "close", "volume"]
    timestamps = dataset.index.to_list()
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    timeframe = str(data_cfg.get("timeframe") or "").strip()
    if not timeframe:
        raise ValueError("Parâmetro obrigatório ausente: data.timeframe no config.json")
    bar_hours = hours_per_bar(timeframe)
    train_bars = bars_for_days(timeframe, int(wf_cfg.get("train_days", 720)))
    val_bars = bars_for_days(timeframe, int(wf_cfg.get("val_days", 180)))
    step_bars = bars_for_days(timeframe, int(wf_cfg.get("step_days", 90)))

    windows = build_windows(len(price_df), train_bars, val_bars, step_bars)
    outdir = Path(wf_cfg.get("outdir", "src/strategies/exper_corr_pos/reports/walk_forward"))
    outdir.mkdir(parents=True, exist_ok=True)

    progress_path = outdir / "wf_progress.log"
    progress_path.write_text("")

    def _log(msg: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        line = f"{timestamp} {msg}"
        print(line, flush=True)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    device = torch.device(wf_cfg.get("device", "cpu"))
    histories = []
    total_windows = len(windows)
    monte_cfg = wf_cfg.get("monte_carlo", {}) or {}
    stress_cost_cfg = wf_cfg.get("stress_costs", {}) or {}
    stress_lag_cfg = wf_cfg.get("stress_lag", {}) or {}
    regime_cfg = wf_cfg.get("regimes", {}) or {}
    global_mc_equity: List[float] = []
    global_mc_pnl: List[float] = []
    global_mc_ruin = 0
    global_mc_total = 0
    cost_aggregate: Dict[float, List[float]] = {}
    lag_aggregate: Dict[int, List[float]] = {}

    _log(
        f"[WF] Iniciando walk-forward com {total_windows} janelas "
        f"(episodes={wf_cfg.get('episodes', 200)}, rollout_steps={wf_cfg.get('rollout_steps', 2048)})"
    )

    for idx, window in enumerate(windows, start=1):
        _log(
            f"[WF] Janela {idx}/{total_windows} — treino [{window.train_start}:{window.train_end}) "
            f"val [{window.val_start}:{window.val_end})"
        )
        train_prices = price_df.iloc[window.train_start : window.train_end]
        train_feats = feat_df.iloc[window.train_start : window.train_end]
        val_prices = price_df.iloc[window.val_start : window.val_end]
        val_feats = feat_df.iloc[window.val_start : window.val_end]

        # Ajusta normalização com base APENAS no treino deste window
        train_norm_mean = train_feats.mean()
        train_norm_std = train_feats.std().replace(0.0, 1.0)

        train_ts = timestamps[window.train_start : window.train_end]
        env = BTCMixtureEnv(
            train_prices,
            train_feats,
            env_cfg,
            norm_mean=train_norm_mean,
            norm_std=train_norm_std,
            timestamps=train_ts,
        )
        input_dim = train_feats.shape[1]
        policy = build_policy(input_dim, cfg)
        trainer = PPOTrainer(policy, ppo_cfg, device=device)

        episodes = int(wf_cfg.get("episodes", 200))
        rollout_steps = int(wf_cfg.get("rollout_steps", 2048))
        for episode in range(episodes):
            trainer.train_step(env, rollout_steps)
            if (episode + 1) % 20 == 0 or episode + 1 == episodes:
                _log(f"[WF] Janela {idx}/{total_windows} — episódio {episode + 1}/{episodes}")

        # Evaluate on validation (sem vazamento): usa normalização do treino
        val_ts = timestamps[window.val_start : window.val_end]
        val_env_train_norm = BTCMixtureEnv(
            val_prices,
            val_feats,
            env_cfg,
            norm_mean=train_norm_mean,
            norm_std=train_norm_std,
            timestamps=val_ts,
        )
        res_train_norm = eval_env_greedy(policy, val_env_train_norm, device=device, bar_hours=bar_hours)

        # Evaluate on validation com normalização por janelão de validação (pode vazar info)
        val_env_val_norm = BTCMixtureEnv(val_prices, val_feats, env_cfg, timestamps=val_ts)
        res_val_norm = eval_env_greedy(policy, val_env_val_norm, device=device, bar_hours=bar_hours)

        # Monte Carlo perturbations
        monte_summary = {}
        if int(monte_cfg.get("simulations", 0)) > 0:
            mc_summary, mc_eq_samples, mc_pnl_samples, mc_ruins = run_monte_carlo_analysis(
                policy,
                val_prices,
                val_feats,
                env_cfg,
                train_norm_mean,
                train_norm_std,
                val_ts,
                simulations=int(monte_cfg.get("simulations", 0)),
                feature_noise_std=float(monte_cfg.get("feature_noise_std", 0.01)),
                price_noise_std=float(monte_cfg.get("price_noise_std", 0.002)),
                device=device,
                bar_hours=bar_hours,
                seed=int(monte_cfg.get("seed", 42)) + idx,
            )
            if mc_summary:
                monte_summary = mc_summary
                global_mc_equity.extend(mc_eq_samples)
                global_mc_pnl.extend(mc_pnl_samples)
                global_mc_ruin += mc_ruins
                global_mc_total += mc_summary.get("simulations", 0)

        # Cost stress
        cost_results: List[Dict[str, Any]] = []
        cost_factors = stress_cost_cfg.get("factors", [0.5, 1.0, 1.5])
        if cost_factors:
            try:
                parsed_factors = [float(f) for f in cost_factors]
            except TypeError:
                parsed_factors = [0.5, 1.0, 1.5]
            cost_results = run_cost_sensitivity(
                policy,
                val_prices,
                val_feats,
                env_cfg,
                train_norm_mean,
                train_norm_std,
                val_ts,
                factors=parsed_factors,
                device=device,
                bar_hours=bar_hours,
            )
            for entry in cost_results:
                factor = float(entry["factor"])
                cost_aggregate.setdefault(factor, []).append(entry["metrics"]["equity_end"])

        # Lag sensitivity (1..N)
        lag_results: List[Dict[str, Any]] = []
        max_lag = int(stress_lag_cfg.get("max_lag", 5))
        if max_lag > 0:
            lag_results = run_lag_sensitivity(
                policy,
                val_prices,
                val_feats,
                env_cfg,
                train_norm_mean,
                train_norm_std,
                val_ts,
                max_lag=max_lag,
                device=device,
                bar_hours=bar_hours,
            )
            for entry in lag_results:
                lag_aggregate.setdefault(int(entry["lag"]), []).append(entry["metrics"]["equity_end"])
        lag1_metrics = next((entry["metrics"] for entry in lag_results if entry.get("lag") == 1), {"pnl": 0.0, "equity_end": 0.0, "trades": 0})

        # Heurísticas simples de alerta (não-fatais):
        norm_delta = res_val_norm["equity_end"] - res_train_norm["equity_end"]
        norm_leak_suspected = norm_delta > 0.05 * max(env_cfg.init_equity, 1e-6)
        lag_drop = res_train_norm["equity_end"] - lag1_metrics.get("equity_end", 0.0)
        timing_suspected = lag_drop > 0.1 * max(env_cfg.init_equity, 1e-6)

        val_returns = val_prices["close"].pct_change().dropna()
        window_vol = float(val_returns.std()) if not val_returns.empty else 0.0

        histories.append(
            {
                "window": idx,
                "train_norm": res_train_norm,
                "val_norm": res_val_norm,
                "lag1": lag1_metrics,
                "lag_sensitivity": lag_results,
                "cost_sensitivity": cost_results,
                "monte_carlo": monte_summary,
                "volatility": window_vol,
                "norm_delta_equity": norm_delta,
                "norm_leak_suspected": bool(norm_leak_suspected),
                "lag_equity_drop": lag_drop,
                "timing_suspected": bool(timing_suspected),
                "regime": None,
            }
        )

        ckpt = outdir / f"policy_window{idx}.pt"
        torch.save(policy.state_dict(), ckpt)
        _log(
            "[WF] Janela "
            f"{idx}/{total_windows} concluída — eq_train={res_train_norm['equity_end']:.2f} "
            f"eq_val={res_val_norm['equity_end']:.2f} trades_val={res_train_norm['trades']}"
        )

    # Estatísticas agregadas
    val_metrics = [h["val_norm"] for h in histories]
    total_periods = len(val_metrics)
    periods_with_profit = sum(1 for m in val_metrics if m.get("pnl", 0.0) > 0)
    periods_with_loss = sum(1 for m in val_metrics if m.get("pnl", 0.0) <= 0)
    success_rate = (periods_with_profit / total_periods) if total_periods > 0 else 0.0

    total_pnl = float(sum(m.get("pnl", 0.0) for m in val_metrics))
    avg_pnl = float(total_pnl / total_periods) if total_periods > 0 else 0.0
    avg_win_rate = float(np.mean([m.get("win_rate", 0.0) for m in val_metrics])) if total_periods > 0 else 0.0
    avg_profit_factor = float(
        np.mean([m.get("profit_factor", 0.0) for m in val_metrics])
    ) if total_periods > 0 else 0.0
    total_trades = int(sum(m.get("trades", 0) for m in val_metrics))
    avg_trade_duration_bars = float(
        np.mean([m.get("avg_trade_duration_bars", 0.0) for m in val_metrics])
    ) if total_periods > 0 else 0.0
    avg_trade_duration_hours = float(
        np.mean([m.get("avg_trade_duration_hours", 0.0) for m in val_metrics])
    ) if total_periods > 0 else 0.0

    # Regime classification
    vols = np.array([h.get("volatility", 0.0) for h in histories], dtype=np.float64)
    regime_low_q = float(np.quantile(vols, regime_cfg.get("low_quantile", 0.33))) if len(vols) > 0 else 0.0
    regime_high_q = float(np.quantile(vols, regime_cfg.get("high_quantile", 0.66))) if len(vols) > 0 else 0.0
    for h, vol in zip(histories, vols, strict=False):
        if len(vols) == 0:
            h["regime"] = "unknown"
        elif vol <= regime_low_q:
            h["regime"] = "low"
        elif vol >= regime_high_q:
            h["regime"] = "high"
        else:
            h["regime"] = "mid"

    regime_summary: Dict[str, Dict[str, float]] = {}
    for label in {"low", "mid", "high"}:
        subset = [h for h in histories if h.get("regime") == label]
        if not subset:
            continue
        vals = [h["val_norm"]["equity_end"] for h in subset]
        regime_summary[label] = {
            "count": len(subset),
            "mean_equity": float(np.mean(vals)),
            "std_equity": float(np.std(vals)),
            "success_rate": float(
                np.mean([1.0 if h["val_norm"].get("pnl", 0.0) > 0 else 0.0 for h in subset])
            ),
        }

    # Monte Carlo overall aggregation
    if global_mc_total > 0:
        mc_eq_arr = np.array(global_mc_equity, dtype=np.float64)
        mc_pnl_arr = np.array(global_mc_pnl, dtype=np.float64)
        monte_carlo_overall = {
            "simulations": int(global_mc_total),
            "equity_end": {
                "mean": float(mc_eq_arr.mean()),
                "std": float(mc_eq_arr.std()),
                "p05": float(np.percentile(mc_eq_arr, 5)),
                "p50": float(np.percentile(mc_eq_arr, 50)),
                "p95": float(np.percentile(mc_eq_arr, 95)),
            },
            "pnl": {
                "mean": float(mc_pnl_arr.mean()),
                "std": float(mc_pnl_arr.std()),
                "p05": float(np.percentile(mc_pnl_arr, 5)),
                "p50": float(np.percentile(mc_pnl_arr, 50)),
                "p95": float(np.percentile(mc_pnl_arr, 95)),
            },
            "ruin_rate": float(global_mc_ruin / global_mc_total),
        }
    else:
        monte_carlo_overall = {}

    cost_summary = {
        str(k): {
            "count": len(v),
            "mean_equity": float(np.mean(v)),
            "std_equity": float(np.std(v)),
        }
        for k, v in cost_aggregate.items()
    }
    lag_summary = {
        str(k): {
            "count": len(v),
            "mean_equity": float(np.mean(v)),
            "std_equity": float(np.std(v)),
        }
        for k, v in lag_aggregate.items()
    }

    summary = {
        "total_periods": total_periods,
        "successful_periods": periods_with_profit,
        "success_rate": success_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "avg_win_rate": avg_win_rate,
        "avg_profit_factor": avg_profit_factor,
        "total_trades": total_trades,
        "avg_trade_duration_bars": avg_trade_duration_bars,
        "avg_trade_duration_hours": avg_trade_duration_hours,
        "periods_with_profit": periods_with_profit,
        "periods_with_loss": periods_with_loss,
        "monte_carlo_overall": monte_carlo_overall,
        "cost_sensitivity": cost_summary,
        "lag_sensitivity": lag_summary,
        "regime_summary": regime_summary,
    }

    payload = {
        "summary": summary,
        "windows": histories,
        "config": {
            "monte_carlo": monte_cfg,
            "stress_costs": stress_cost_cfg,
            "stress_lag": stress_lag_cfg,
            "regimes": regime_cfg,
        },
    }
    summary_path = outdir / "wf_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# Walk-Forward Summary",
        "",
        f"- Total windows: **{total_periods}**",
        f"- Success rate: **{success_rate:.2%}**",
        f"- Total PnL: **{total_pnl:.2f}**",
        "",
        "## Monte Carlo",
    ]
    if monte_carlo_overall:
        mc = monte_carlo_overall
        md_lines.extend(
            [
                f"- Simulations: {mc['simulations']}",
                f"- Equity (p05 / p50 / p95): {mc['equity_end']['p05']:.2f} / {mc['equity_end']['p50']:.2f} / {mc['equity_end']['p95']:.2f}",
                f"- Ruin rate: {mc['ruin_rate']:.2%}",
            ]
        )
    else:
        md_lines.append("- Monte Carlo desabilitado.")

    md_lines.append("")
    md_lines.append("## Cost Sensitivity")
    if cost_summary:
        md_lines.append("| Factor | Mean Equity | Std Equity | Count |")
        md_lines.append("|--------|-------------|------------|-------|")
        for factor, stats in sorted(cost_summary.items(), key=lambda x: float(x[0])):
            md_lines.append(
                f"| {factor} | {stats['mean_equity']:.2f} | {stats['std_equity']:.2f} | {stats['count']} |"
            )
    else:
        md_lines.append("- Nenhum stress de custo executado.")

    md_lines.append("")
    md_lines.append("## Lag Sensitivity")
    if lag_summary:
        md_lines.append("| Lag | Mean Equity | Std Equity | Count |")
        md_lines.append("|-----|-------------|------------|-------|")
        for lag, stats in sorted(lag_summary.items(), key=lambda x: int(float(x[0]))):
            md_lines.append(
                f"| {lag} | {stats['mean_equity']:.2f} | {stats['std_equity']:.2f} | {stats['count']} |"
            )
    else:
        md_lines.append("- Nenhum stress de lag executado.")

    md_lines.append("")
    md_lines.append("## Regime Summary")
    if regime_summary:
        md_lines.append("| Regime | Windows | Mean Equity | Std Equity | Success Rate |")
        md_lines.append("|---------|---------|-------------|------------|--------------|")
        for regime, stats in regime_summary.items():
            md_lines.append(
                f"| {regime} | {stats['count']} | {stats['mean_equity']:.2f} | {stats['std_equity']:.2f} | {stats['success_rate']:.2%} |"
            )
    else:
        md_lines.append("- Regimes não avaliados.")

    md_path = outdir / "wf_summary.md"
    md_path.write_text("\n".join(md_lines))

    _log(
        f"[WF] Concluído. Períodos={total_periods} trades={total_trades} "
        f"P&L_total={total_pnl:.2f} win_rate_médio={avg_win_rate:.2%} PF_médio={avg_profit_factor:.2f}"
    )
    _log(f"[WF] Resumo salvo em {summary_path}")
    _log(f"[WF] Relatório Markdown salvo em {md_path}")


if __name__ == "__main__":
    main()
