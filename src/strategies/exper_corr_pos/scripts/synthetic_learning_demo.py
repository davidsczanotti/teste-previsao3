"""Mini-experimentos sintéticos para inspecionar o aprendizado do agente.

Uso típico (dentro do repo):

  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.scripts.synthetic_learning_demo

Por padrão roda o cenário 'trend_up' com poucos episódios e salva
as métricas em:
  src/strategies/exper_corr_pos/reports/train/synthetic_trend_up.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from ..env import BTCMixtureEnv, EnvConfig
from ..models import MoEPolicy, PPOConfig
from ..trainer import PPOTrainer


REPORT_DIR = Path("src/strategies/exper_corr_pos/reports/train")


@dataclass
class Scenario:
    name: str
    make_env: Callable[[], BTCMixtureEnv]
    oracle_actions: List[int]


def _make_trend_env(*, length: int, direction: str, fee_pct: float = 0.0) -> BTCMixtureEnv:
    """Cria um ambiente sintético simples com tendência conhecida."""
    idx = np.arange(length, dtype=float)
    if direction == "up":
        close = 100.0 + idx
        trend_state = np.ones(length, dtype=float)
        trend_strength = np.ones(length, dtype=float)
    elif direction == "down":
        close = 100.0 - idx
        trend_state = -np.ones(length, dtype=float)
        trend_strength = np.ones(length, dtype=float)
    elif direction == "flat":
        close = np.full(length, 100.0, dtype=float)
        trend_state = np.zeros(length, dtype=float)
        trend_strength = np.zeros(length, dtype=float)
    else:
        raise ValueError(f"direction inválido: {direction}")

    price_df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(length, 10.0, dtype=float),
        }
    )
    feat_df = pd.DataFrame(
        {
            "atr_14": np.ones(length, dtype=float),
            "htf_trend_state": trend_state,
            "htf_trend_strength": trend_strength,
        }
    )
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=1.0,
        fee_pct=fee_pct,
        slippage_pct=0.0,
        random_start=False,
        window_bars=length,
        reward_scale_divisor=1.0,
        risk_atr_scale=0.0,
        idle_penalty_factor=0.0,
        trend_bonus_coef=0.0,
        trend_bonus_coef_pct=0.0,
        giveback_threshold_pct=0.0,
        giveback_penalty_pct=0.0,
        short_trade_penalty=0.0,
        short_trade_min_bars=1,
        min_hold_bars_enabled=False,
        max_drawdown_pct=1.0,
        hold_bonus_alpha=0.0,
        hold_bonus_positive_only=True,
        hold_shaping_alpha=0.02,
    )
    return BTCMixtureEnv(price_df, feat_df, cfg)


def _build_policy(input_dim: int) -> MoEPolicy:
    """Política MoE pequena para os cenários sintéticos."""
    return MoEPolicy(
        input_dim=input_dim,
        num_actions=3,
        expert_hidden=[32, 32],
        gating_hidden=[32, 32],
        num_experts=2,
        temperature=0.7,
        top_k=2,
        gating_use_attention=False,
        attention_dim=32,
        attention_heads=2,
        attention_dropout=0.0,
        attention_weight=1.0,
    )


def _build_trainer(policy: MoEPolicy) -> PPOTrainer:
    cfg = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.1,
        learning_rate=1e-4,
        entropy_coef=0.0,
        value_coef=0.5,
        max_grad_norm=0.5,
        train_iters=4,
        batch_size=256,
    )
    return PPOTrainer(policy, cfg, device=torch.device("cpu"), lb_coef=0.0)


def _run_oracle(make_env: Callable[[], BTCMixtureEnv], actions: List[int]) -> Tuple[float, float]:
    env = make_env()
    _ = env.reset()
    total_reward = 0.0
    done = False
    t = 0
    last_equity = float(env._equity)  # type: ignore[attr-defined]
    while not done:
        action = actions[t] if t < len(actions) else 1  # default: ficar flat
        _, reward, done, info = env.step(int(action))
        total_reward += float(reward)
        last_equity = float(info.get("equity", last_equity))
        t += 1
    return total_reward, last_equity


def _evaluate_policy(make_env: Callable[[], BTCMixtureEnv], policy: MoEPolicy) -> Tuple[float, float]:
    env = make_env()
    obs = torch.tensor(env.reset(), dtype=torch.float32)
    done = False
    total_reward = 0.0
    last_equity = float(env._equity)  # type: ignore[attr-defined]
    while not done:
        with torch.no_grad():
            dist, _, _ = policy(obs.unsqueeze(0))
            action = torch.argmax(dist.probs, dim=-1)
        next_obs, reward, done, info = env.step(int(action.item()))
        total_reward += float(reward)
        obs = torch.tensor(next_obs, dtype=torch.float32)
        last_equity = float(info.get("equity", last_equity))
    return total_reward, last_equity


def _trend_up_scenario() -> Scenario:
    length = 24
    make_env = lambda: _make_trend_env(length=length, direction="up", fee_pct=0.0)
    # Oracle: abre long na primeira barra, mantém até o final e fecha.
    oracle_actions = [2] * (length - 2) + [1, 1]
    return Scenario("trend_up", make_env, oracle_actions)


def _trend_down_scenario() -> Scenario:
    length = 24
    make_env = lambda: _make_trend_env(length=length, direction="down", fee_pct=0.0)
    # Oracle: abre short na primeira barra, mantém até o final e fecha.
    oracle_actions = [0] * (length - 2) + [1, 1]
    return Scenario("trend_down", make_env, oracle_actions)


def _stay_flat_scenario() -> Scenario:
    length = 24
    def make_env() -> BTCMixtureEnv:
        # Ambiente totalmente flat com custos relativamente altos para entrar.
        # A política ótima é não operar; qualquer trade gera PnL líquido negativo
        # (taxa + penalidade de turnover).
        env = _make_trend_env(length=length, direction="flat", fee_pct=0.001)
        env.cfg.turnover_penalty_pct = 0.05
        env.cfg.turnover_penalty = 0.0
        env.cfg.short_trade_penalty = 0.0
        env.cfg.short_trade_min_bars = 1
        return env
    # Oracle: nunca entra; sempre flat (ação 1 em todas as barras).
    oracle_actions = [1] * length
    return Scenario("stay_flat", make_env, oracle_actions)


SCENARIOS: Dict[str, Callable[[], Scenario]] = {
    "trend_up": _trend_up_scenario,
    "trend_down": _trend_down_scenario,
    "stay_flat": _stay_flat_scenario,
}


def run_synthetic_experiment(scenario: Scenario, episodes: int, rollout_steps: int, eval_every: int) -> Path:
    np.random.seed(0)
    torch.manual_seed(0)

    train_env = scenario.make_env()
    input_dim = train_env.features.shape[1]
    policy = _build_policy(input_dim)
    trainer = _build_trainer(policy)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"synthetic_{scenario.name}.csv"
    with out_path.open("w", newline="") as f:
        fieldnames = [
            "episode",
            "policy_loss",
            "value_loss",
            "entropy",
            "avg_reward",
            "sum_reward",
            "eval_reward",
            "eval_equity",
            "oracle_reward",
            "oracle_equity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        oracle_reward, oracle_equity = _run_oracle(scenario.make_env, scenario.oracle_actions)

        for ep in range(1, episodes + 1):
            metrics = trainer.train_step(train_env, rollout_steps=rollout_steps)
            eval_reward = float("nan")
            eval_equity = float("nan")
            if ep % eval_every == 0 or ep == 1:
                eval_reward, eval_equity = _evaluate_policy(scenario.make_env, policy)
                print(
                    f"[{scenario.name}] ep={ep:04d} "
                    f"avg_reward={metrics['avg_reward']:.4f} "
                    f"eval_reward={eval_reward:.4f} "
                    f"oracle={oracle_reward:.4f}"
                )

            row = {
                "episode": ep,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
                "avg_reward": metrics["avg_reward"],
                "sum_reward": metrics["sum_reward"],
                "eval_reward": eval_reward,
                "eval_equity": eval_equity,
                "oracle_reward": oracle_reward,
                "oracle_equity": oracle_equity,
            }
            writer.writerow(row)

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini-experimentos sintéticos para inspecionar o aprendizado do MoE.")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        default="trend_up",
        help="Cenário sintético a rodar.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Quantidade de passos de treino (train_step) no experimento.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=64,
        help="Passos coletados por train_step.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Frequência (em episódios) de avaliação greedy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_builder = SCENARIOS[args.scenario]
    scenario = scenario_builder()
    out_path = run_synthetic_experiment(
        scenario=scenario,
        episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        eval_every=args.eval_every,
    )
    print(f"Resultados salvos em {out_path}")


if __name__ == "__main__":
    main()
