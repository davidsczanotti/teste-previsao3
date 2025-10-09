from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .agent import QLearningAgent
from .env import Action, TradingEnvironment


@dataclass
class TrainingConfig:
    """Runtime options used during the training loop."""

    episodes: int = 50
    max_steps: Optional[int] = None
    render_episodes: int = 1
    render_every: Optional[int] = None


@dataclass
class EpisodeResult:
    """Summary of a finished episode."""

    index: int
    steps: int
    total_reward: float
    final_balance: float
    final_equity: float
    trades: int
    winning_trades: int
    epsilon: float


class Trainer:
    """Coordinate the interaction loop between environment and agent."""

    def __init__(
        self,
        environment: TradingEnvironment,
        agent: QLearningAgent,
        config: TrainingConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Store collaborators and prepare the stdout logger."""

        self.env = environment
        self.agent = agent
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def train(self) -> List[EpisodeResult]:
        """Run the configured number of training episodes."""

        results: List[EpisodeResult] = []

        for episode in range(self.config.episodes):
            observation = self.env.reset()
            self.agent.reset_episode()

            done = False
            steps = 0
            total_reward = 0.0

            render = self._should_render_episode(episode)
            if render:
                self._render_header(episode)

            while not done:
                action_index = self.agent.select_action(observation)
                next_observation, reward, done_flag, info = self.env.step(Action(action_index))
                self.agent.learn(observation, action_index, reward, next_observation, done_flag)
                self.agent.decay_epsilon()

                total_reward += reward
                steps += 1

                if render:
                    self._render_step(steps, Action(action_index), reward, info)

                observation = next_observation
                done = done_flag

                if self.config.max_steps and steps >= self.config.max_steps:
                    done = True
                    if render:
                        self.logger.info("-> Episódio interrompido pelo limite de passos configurado.")

            result = EpisodeResult(
                index=episode + 1,
                steps=steps,
                total_reward=total_reward,
                final_balance=self.env.balance,
                final_equity=self.env.last_equity,
                trades=len(self.env.trade_log),
                winning_trades=sum(1 for trade in self.env.trade_log if trade.pnl > 0),
                epsilon=self.agent.epsilon,
            )
            results.append(result)

            if render:
                self._render_episode_summary(result)

        return results

    def _should_render_episode(self, episode_index: int) -> bool:
        """Decide whether logs for a given episode should be printed."""

        if episode_index < self.config.render_episodes:
            return True
        if self.config.render_every and (episode_index + 1) % self.config.render_every == 0:
            return True
        return False

    def _render_header(self, episode_index: int) -> None:
        """Emit a table header to ease manual inspection in the terminal."""

        self.logger.info("")
        self.logger.info("===== Episódio %03d =====", episode_index + 1)
        self.logger.info("%-6s %-8s %-10s %-12s %-12s %-12s", "Passo", "Ação", "Preço", "Recompensa", "Saldo", "Equity")

    def _render_step(self, step: int, action: Action, reward: float, info: Dict[str, float]) -> None:
        """Print a single row describing the latest interaction."""

        price = info.get("price", 0.0)
        balance = info.get("balance", 0.0)
        equity = info.get("equity", 0.0)
        event = info.get("event", "")
        self.logger.info(
            "%-6d %-8s %-10.2f %-12.2f %-12.2f %-12.2f %s",
            step,
            action.name,
            price,
            reward,
            balance,
            equity,
            event,
        )

    def _render_episode_summary(self, result: EpisodeResult) -> None:
        """Display compact end-of-episode statistics."""

        win_rate = (result.winning_trades / result.trades) if result.trades else 0.0
        self.logger.info(
            "Resumo -> passos=%d | recompensa=%.2f | saldo final=%.2f | equity=%.2f | trades=%d | win_rate=%.0f%% | epsilon=%.3f",
            result.steps,
            result.total_reward,
            result.final_balance,
            result.final_equity,
            result.trades,
            win_rate * 100.0,
            result.epsilon,
        )
