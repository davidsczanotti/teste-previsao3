from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

from ..data import load_primary_series, load_confirm_series, prepare_dataset
from ..env import BTCMixtureEnv, EnvConfig
from ..models import MoEPolicy
from ..utils_cfg import build_policy, hours_per_bar
from ..walk_forward import (
    eval_env_greedy,
    run_monte_carlo_analysis,
    run_cost_sensitivity,
    run_lag_sensitivity,
    DEFAULT_CONFIG,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress tests para exper_corr_pos (custo/lag/Monte Carlo)")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Arquivo de configuração (default: src/strategies/exper_corr_pos/config.json)",
    )
    parser.add_argument(
        "--model-path",
        default="src/strategies/exper_corr_pos/reports/train/moe_policy_final.pt",
        help="Checkpoint do modelo MoE (default: reports/train/moe_policy_final.pt)",
    )
    parser.add_argument(
        "--cost-factors",
        default="0.5,1.0,1.5",
        help="Fatores multiplicativos para stress de custo (fee + slippage). Ex: '0.5,1.0,1.5'",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=5,
        help="Máximo de barras de lag para teste de sensibilidade (default: 5)",
    )
    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=500,
        help="Quantidade de simulações Monte Carlo (default: 500, 0 desliga)",
    )
    parser.add_argument(
        "--feature-noise-std",
        type=float,
        default=0.01,
        help="Desvio padrão relativo para ruído nas features (default: 0.01)",
    )
    parser.add_argument(
        "--price-noise-std",
        type=float,
        default=0.002,
        help="Desvio padrão para ruído nos preços (default: 0.002)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed base para as simulações (default: 42)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device para avaliação (default: cpu)",
    )
    parser.add_argument(
        "--outdir",
        default="src/strategies/exper_corr_pos/reports/train",
        help="Diretório de saída dos relatórios (default: reports/train)",
    )
    return parser.parse_args()


def load_policy(config: Dict[str, Any], input_dim: int, model_path: Path, device: torch.device) -> MoEPolicy:
    policy = build_policy(input_dim, config)
    try:
        state = torch.load(model_path, map_location=device)
        policy.load_state_dict(state, strict=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint não encontrado em {model_path}")
    policy.to(device)
    policy.eval()
    return policy


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())

    primary_df = load_primary_series(cfg)
    confirm_df = load_confirm_series(cfg)
    dataset = prepare_dataset(primary_df, config=cfg, confirm_df=confirm_df)
    price_cols = ["open", "high", "low", "close", "volume"]
    timestamps = dataset.index.to_list()
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    env_cfg = EnvConfig(**cfg.get("env", {}))
    device = torch.device(args.device)
    bar_hours = hours_per_bar(cfg.get("data", {}).get("timeframe", "1d"))

    norm_mean = feat_df.mean()
    norm_std = feat_df.std().replace(0.0, 1.0)

    policy = load_policy(cfg, feat_df.shape[1], Path(args.model_path), device=device)

    base_env = BTCMixtureEnv(
        price_df,
        feat_df,
        env_cfg,
        norm_mean=norm_mean,
        norm_std=norm_std,
        timestamps=timestamps,
    )
    baseline = eval_env_greedy(policy, base_env, device=device, bar_hours=bar_hours)

    factors = []
    if args.cost_factors:
        try:
            factors = [float(x.strip()) for x in args.cost_factors.split(",") if x.strip()]
        except ValueError:
            factors = [0.5, 1.0, 1.5]
    cost_results = run_cost_sensitivity(
        policy,
        price_df,
        feat_df,
        env_cfg,
        norm_mean,
        norm_std,
        timestamps,
        factors=factors,
        device=device,
        bar_hours=bar_hours,
    ) if factors else []

    lag_results = run_lag_sensitivity(
        policy,
        price_df,
        feat_df,
        env_cfg,
        norm_mean,
        norm_std,
        timestamps,
        max_lag=args.max_lag,
        device=device,
        bar_hours=bar_hours,
    ) if args.max_lag > 0 else []

    monte_summary = {}
    if args.monte_carlo > 0:
        monte_summary, _, _, _ = run_monte_carlo_analysis(
            policy,
            price_df,
            feat_df,
            env_cfg,
            norm_mean,
            norm_std,
            timestamps,
            simulations=args.monte_carlo,
            feature_noise_std=args.feature_noise_std,
            price_noise_std=args.price_noise_std,
            device=device,
            bar_hours=bar_hours,
            seed=args.seed,
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(Path(args.config)),
        "model_path": str(Path(args.model_path)),
        "baseline": baseline,
        "cost_sensitivity": cost_results,
        "lag_sensitivity": lag_results,
        "monte_carlo": monte_summary,
        "parameters": {
            "cost_factors": factors,
            "max_lag": args.max_lag,
            "monte_carlo_simulations": args.monte_carlo,
            "feature_noise_std": args.feature_noise_std,
            "price_noise_std": args.price_noise_std,
            "seed": args.seed,
        },
    }

    out_json = outdir / "stress_tests.json"
    out_json.write_text(json.dumps(payload, indent=2))

    lines: List[str] = [
        "# Stress Tests",
        "",
        f"- Baseline equity: {baseline.get('equity_end', float('nan')):.2f}",
        f"- Baseline trades: {baseline.get('trades', 0)}",
        "",
        "## Cost Sensitivity",
    ]
    if cost_results:
        lines.append("| Factor | Equity | Trades | Win Rate | PF |")
        lines.append("|--------|--------|--------|----------|----|")
        for entry in cost_results:
            metrics = entry["metrics"]
            lines.append(
                f"| {entry['factor']} | {metrics.get('equity_end', 0.0):.2f} | "
                f"{metrics.get('trades', 0)} | {metrics.get('win_rate', 0.0):.2%} | "
                f"{metrics.get('profit_factor', 0.0):.2f} |"
            )
    else:
        lines.append("- (Nenhum stress de custo executado.)")

    lines.append("")
    lines.append("## Lag Sensitivity")
    if lag_results:
        lines.append("| Lag | Equity | Trades | Win Rate | PF |")
        lines.append("|-----|--------|--------|----------|----|")
        for entry in lag_results:
            metrics = entry["metrics"]
            lines.append(
                f"| {entry['lag']} | {metrics.get('equity_end', 0.0):.2f} | "
                f"{metrics.get('trades', 0)} | {metrics.get('win_rate', 0.0):.2%} | "
                f"{metrics.get('profit_factor', 0.0):.2f} |"
            )
    else:
        lines.append("- (Nenhum stress de lag executado.)")

    lines.append("")
    lines.append("## Monte Carlo")
    if monte_summary:
        lines.extend(
            [
                f"- Simulações: {monte_summary['simulations']}",
                f"- Equity (p05 / p50 / p95): {monte_summary['equity_end']['p05']:.2f} / "
                f"{monte_summary['equity_end']['p50']:.2f} / {monte_summary['equity_end']['p95']:.2f}",
                f"- Ruin rate: {monte_summary['ruin_rate']:.2%}",
            ]
        )
    else:
        lines.append("- (Monte Carlo desabilitado.)")

    out_md = outdir / "stress_tests.md"
    out_md.write_text("\n".join(lines))

    print(f"[stress] Baseline equity: {baseline.get('equity_end', float('nan')):.2f}")
    if cost_results:
        best_cost = max(cost_results, key=lambda x: x["metrics"].get("equity_end", 0.0))
        print(
            f"[stress] Melhor fator de custo: {best_cost['factor']} "
            f"(equity={best_cost['metrics'].get('equity_end', 0.0):.2f})"
        )
    if lag_results:
        worst_lag = min(lag_results, key=lambda x: x["metrics"].get("equity_end", 0.0))
        print(
            f"[stress] Pior lag: {worst_lag['lag']} "
            f"(equity={worst_lag['metrics'].get('equity_end', 0.0):.2f})"
        )
    if monte_summary:
        print(
            "[stress] Monte Carlo equity p05/p50/p95: "
            f"{monte_summary['equity_end']['p05']:.2f} / "
            f"{monte_summary['equity_end']['p50']:.2f} / "
            f"{monte_summary['equity_end']['p95']:.2f}"
        )
    print(f"[stress] Relatórios salvos em {outdir}")


if __name__ == "__main__":
    main()
