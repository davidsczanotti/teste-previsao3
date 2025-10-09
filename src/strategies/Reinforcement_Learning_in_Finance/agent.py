from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .env import Action, Observation


@dataclass
class AgentConfig:
    """Hyperparameters that control the Q-learning update."""

    learning_rate: float = 0.2
    discount: float = 0.95
    epsilon: float = 0.4
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05


class StateDiscretizer:
    """Convert continuous observations into discrete table indices."""

    def __init__(self, window_size: int, threshold: float = 0.001) -> None:
        """Configure the discretizer.

        Args:
            window_size: Number of returns in the observation vector.
            threshold: Absolute return above which we classify a move as forte.
        """

        self.window_size = window_size
        self.threshold = abs(threshold)
        self.state_count = (3 ** window_size) * 2

    def encode(self, observation: Observation) -> int:
        """Convert an ``Observation`` into an index for the Q-table."""

        buckets = []
        for value in observation.features:
            if value <= -self.threshold:
                buckets.append(0)
            elif value >= self.threshold:
                buckets.append(2)
            else:
                buckets.append(1)

        state_index = 0
        for bucket in buckets:
            state_index = state_index * 3 + bucket

        state_index = state_index * 2 + int(observation.position)
        return state_index

    def size(self) -> int:
        """Return the number of discrete states represented."""

        return self.state_count


class QLearningAgent:
    """Table-based Q-learning agent for the trading environment."""

    def __init__(self, config: AgentConfig, discretizer: StateDiscretizer, seed: Optional[int] = None) -> None:
        """Initialise the agent with a Q-table of zeros."""

        self.config = config
        self.discretizer = discretizer
        self.rng = np.random.default_rng(seed)

        self.n_states = discretizer.size()
        self.n_actions = len(Action)
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.epsilon = config.epsilon

    def select_action(self, observation: Observation) -> int:
        """Choose an action index following an epsilon-greedy strategy."""

        state_index = self.discretizer.encode(observation)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        q_values = self.q_table[state_index]
        return int(np.argmax(q_values))

    def learn(
        self,
        observation: Observation,
        action_index: int,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> None:
        """Update the Q-table based on the transition experienced."""

        state_idx = self.discretizer.encode(observation)
        next_state_idx = self.discretizer.encode(next_observation)

        best_next = 0.0 if done else float(np.max(self.q_table[next_state_idx]))
        target = reward + self.config.discount * best_next
        td_error = target - self.q_table[state_idx, action_index]
        self.q_table[state_idx, action_index] += self.config.learning_rate * td_error

    def decay_epsilon(self) -> None:
        """Decrease the exploration rate until the configured minimum."""

        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def reset_episode(self) -> None:
        """Hook called at the start of every episode (currently no-op)."""

        # Mantemos o método para facilitar extensões futuras (ex.: warm restarts).
        return None

    def q_values_for(self, observation: Observation) -> np.ndarray:
        """Expose Q-values for debugging dashboards."""

        state_idx = self.discretizer.encode(observation)
        return self.q_table[state_idx]
