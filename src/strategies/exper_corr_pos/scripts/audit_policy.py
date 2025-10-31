"""Auditoria rápida de ações e trades do agente MoE.

Uso:

  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.scripts.audit_policy \\
      --days 365 --checkpoint src/strategies/exper_corr_pos/reports/train/moe_policy_final.pt

Sem argumentos o script usa os valores do config.json (visualize.prefer) e
carrega 180 dias do cache local.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import torch

from ..data import load_primary_series, load_confirm_series, prepare_dataset
from ..env import BTCMixtureEnv, EnvConfig
from ..models import MoEPolicy
from ..visualize import _find_checkpoint


CFG_PATH = Path("src/strategies/exper_corr_pos/config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita ações e PnL do policy atual")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint específico (.pt). Quando omitido usa visualize.prefer",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Quantidade de dias para carregar (default: visualize.days ou 180)",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Mantém random_start do config (default: força False para avaliação determinística)",
    )
    return parser.parse_args()


def load_policy(cfg: dict, checkpoint: Path | None) -> tuple[MoEPolicy, Path]:
    model_cfg = cfg.get("model", {})
    prefer = cfg.get("visualize", {}).get("prefer", "latest")
    chosen = Path(checkpoint) if checkpoint else _find_checkpoint(prefer)

    # Monta política com o mesmo input_dim das features
    dummy_base = load_primary_series(cfg)
    dummy_confirm = load_confirm_series(cfg)
    dummy_dataset = prepare_dataset(dummy_base.tail(600), config=cfg, confirm_df=dummy_confirm)
    price_cols = ["open", "high", "low", "close", "volume"]
    input_dim = dummy_dataset.drop(columns=price_cols).shape[1]

    policy = MoEPolicy(
        input_dim=input_dim,
        num_actions=3,
        expert_hidden=model_cfg.get("expert_hidden", [64, 32]),
        gating_hidden=model_cfg.get("gating_hidden", [64, 32]),
        num_experts=model_cfg.get("num_experts", 4),
        temperature=model_cfg.get("temperature", 1.6),
        top_k=model_cfg.get("top_k", 3),
    )
    state = torch.load(chosen, map_location="cpu")
    model_state = policy.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    policy.load_state_dict(filtered, strict=False)
    policy.eval()
    return policy, chosen


def main() -> None:
    args = parse_args()
    cfg = json.loads(CFG_PATH.read_text())

    days = args.days or int(cfg.get("visualize", {}).get("days", 180))
    base_df = load_primary_series(cfg)
    confirm_df = load_confirm_series(cfg)
    min_bars = max(int(cfg.get("data", {}).get("spread_window", 240)) + 20, 600)
    base_df = base_df.tail(max(days * 24, min_bars))
    dataset = prepare_dataset(base_df, config=cfg, confirm_df=confirm_df)
    price_cols = ["open", "high", "low", "close", "volume"]
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    policy, checkpoint = load_policy(cfg, args.checkpoint)

    env_cfg = EnvConfig(**cfg.get("env", {}))
    if not args.random_start:
        env_cfg.random_start = False
        env_cfg.window_bars = 0
    env = BTCMixtureEnv(price_df, feat_df, env_cfg)

    obs = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
    action_counts = {0: 0, 1: 0, 2: 0}
    trade_pnls: list[float] = []
    trade_bars: list[int] = []

    done = False
    while not done:
        with torch.no_grad():
            dist, _, _ = policy(obs)
            action = torch.argmax(dist.probs, dim=-1).item()
        next_obs, reward, done, info = env.step(action)
        action_counts[action] += 1
        if info.get("trade_closed"):
            trade_pnls.append(float(info.get("trade_pnl", 0.0)))
            trade_bars.append(int(info.get("trade_bars", 0)))
        obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

    steps = sum(action_counts.values())
    print("=== Auditoria do policy ===")
    print(f"checkpoint: {checkpoint}")
    print(f"dias avaliados: {days}")
    print("contagem de ações:")
    for k, v in action_counts.items():
        ratio = v / steps if steps else 0.0
        label = {0: "short", 1: "flat", 2: "long"}.get(k, str(k))
        print(f"  {label:5s}: {v:6d} ({ratio:.2%})")

    if trade_pnls:
        winners = sum(p > 0 for p in trade_pnls)
        print(f"trades fechados: {len(trade_pnls)} | win rate: {winners/len(trade_pnls):.2%}")
        print(f"PnL médio: {mean(trade_pnls):.5f} | mediana: {median(trade_pnls):.5f}")
        print(f"Duração média (barras): {mean(trade_bars):.2f} | mediana: {median(trade_bars)}")
    else:
        print("Nenhum trade fechado na janela avaliada.")

    print(f"Equity final: {info.get('equity', float('nan')):.2f}")


if __name__ == "__main__":  # pragma: no cover
    main()
