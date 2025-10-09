from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from tqdm import trange

from ..Reinforcement_Learning_in_Finance.env import TradingEnvironment
from .agent import DQNAgent


@dataclass
class TrainingConfig:
    """Parameters to control the training loop execution."""

    episodes: int = 20
    max_steps: Optional[int] = None
    render_episodes: int = 1
    render_every: Optional[int] = None


@dataclass
class EpisodeResult:
    """Aggregated results from a single training episode."""

    index: int
    steps: int
    total_reward: float
    final_balance: float
    final_equity: float
    trade_count: int
    win_rate: float
    epsilon: float


class Trainer:
    """Orchestrates the training loop for a DQN agent."""

    def __init__(
        self,
        env: TradingEnvironment,
        agent: DQNAgent,
        config: TrainingConfig,
        logger: logging.Logger,
    ) -> None:
        self.env = env
        self.agent = agent
        self.config = config
        self.logger = logger

    def train(self) -> List[EpisodeResult]:
        """Run the main training loop for the configured number of episodes."""
        results = []
        for i in (pbar := trange(1, self.config.episodes + 1, desc="Treinando")):
            result = self._run_episode(i)
            results.append(result)
            pbar.set_postfix(
                reward=f"{result.total_reward:.2f}",
                equity=f"{result.final_equity:.2f}",
                trades=result.trade_count,
                epsilon=f"{result.epsilon:.3f}",
            )

            if i % self.agent.config.target_update_freq == 0:
                self.logger.debug("Atualizando a target network no episódio %d", i)
                self.agent.update_target_net()

        return results

    def _run_episode(self, episode_index: int) -> EpisodeResult:
        """Execute a single episode from start to finish."""
        observation = self.env.reset()
        self.agent.reset_episode()

        should_render = episode_index <= self.config.render_episodes or (
            self.config.render_every and episode_index % self.config.render_every == 0
        )

        if should_render:
            self.logger.info("\n===== Episódio %03d =====", episode_index)
            self.logger.info("Vida  Passo  Ação     Preço      Recompensa   Saldo        Equity      ")

        total_reward = 0.0
        for step in range(1, (self.config.max_steps or 999_999) + 1):
            action = self.agent.select_action(observation)
            next_observation, reward, done, info = self.env.step(action)

            self.agent.remember(observation, action, reward, next_observation, done)
            self.agent.learn()

            total_reward += reward
            observation = next_observation

            if should_render:
                event = info.get("event", "")
                self.logger.info(
                    "%-5d %-6d %-8s %-10.2f %-12.2f %-12.2f %-12.2f %s",
                    episode_index,
                    step,
                    info["action"],
                    info["price"],
                    reward,
                    info["balance"],
                    info["equity"],
                    event,
                )

            if done:
                break

        self.agent.decay_epsilon()
        win_rate = 0.0
        if len(self.env.trade_log) > 0:
            wins = sum(1 for t in self.env.trade_log if t.pnl > 0)
            win_rate = (wins / len(self.env.trade_log)) * 100

        return EpisodeResult(
            index=episode_index,
            steps=observation.step,
            total_reward=total_reward,
            final_balance=observation.balance,
            final_equity=observation.equity,
            trade_count=len(self.env.trade_log),
            win_rate=win_rate,
            epsilon=self.agent.epsilon,
        )
