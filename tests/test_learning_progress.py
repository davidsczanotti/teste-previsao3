from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import warnings

warnings.filterwarnings("ignore", message="var\\(\\): degrees of freedom is <= 0\\.")

from src.strategies.exper_corr_pos.models import MoEPolicy, PPOConfig
from src.strategies.exper_corr_pos.trainer import PPOTrainer


class DummyActionSpace:
    def __init__(self, n: int) -> None:
        self.n = n


class DummyObservationSpace:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class SimpleTrendEnv:
    """Ambiente determinístico: recompensa máxima ao manter posição long."""

    def __init__(self, max_steps: int = 40) -> None:
        self.max_steps = max_steps
        self.step_idx = 0
        self.action_space = DummyActionSpace(3)
        self.observation_space = DummyObservationSpace((2,))

    def reset(self) -> np.ndarray:
        self.step_idx = 0
        return np.array([1.0, 0.0], dtype=np.float32)

    def step(self, action: int):
        # ação 2 (long) é recompensada; short é punida; flat levemente negativa
        if action == 2:
            reward = 1.0
        elif action == 1:
            reward = -0.2
        else:  # action == 0 (short)
            reward = -1.0
        self.step_idx += 1
        done = self.step_idx >= self.max_steps
        obs = np.array([1.0, 0.0], dtype=np.float32)
        info = {"equity": 1000.0 + reward * self.step_idx, "position": action - 1}
        return obs, reward, done, info


def _evaluate_policy(policy: MoEPolicy, env: SimpleTrendEnv) -> float:
    obs = torch.tensor(env.reset(), dtype=torch.float32)
    total_reward = 0.0
    done = False
    while not done:
        with torch.no_grad():
            dist, _, _ = policy(obs.unsqueeze(0))
            action = torch.argmax(dist.probs, dim=-1).item()
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = torch.tensor(next_obs, dtype=torch.float32)
    return float(total_reward)


def test_policy_improves_after_training():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    env_eval_before = SimpleTrendEnv()
    env_train = SimpleTrendEnv()

    input_dim = env_train.observation_space.shape[0]
    policy = MoEPolicy(
        input_dim=input_dim,
        num_actions=3,
        expert_hidden=[32],
        gating_hidden=[16],
        num_experts=1,
        top_k=1,
    )

    baseline_reward = _evaluate_policy(policy, env_eval_before)

    trainer = PPOTrainer(
        policy,
        PPOConfig(
            learning_rate=1e-3,
            batch_size=128,
            train_iters=6,
            clip_ratio=0.2,
            entropy_coef=0.01,
        ),
        device=torch.device("cpu"),
    )

    for _ in range(60):
        trainer.train_step(env_train, rollout_steps=64)

    env_eval_after = SimpleTrendEnv()
    improved_reward = _evaluate_policy(policy, env_eval_after)

    assert improved_reward > baseline_reward
