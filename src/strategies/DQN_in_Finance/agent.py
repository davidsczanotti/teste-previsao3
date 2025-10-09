from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..Reinforcement_Learning_in_Finance.env import Action, Observation
from .model import SimpleDQN


@dataclass
class AgentConfig:
    """Hyperparameters for the DQN agent."""

    learning_rate: float = 0.001
    discount: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    batch_size: int = 32
    buffer_size: int = 10_000
    hidden_size: int = 32
    target_update_freq: int = 10  # in episodes


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size: int, seed: Optional[int] = None):
        self.memory = collections.deque(maxlen=buffer_size)
        self.rng = np.random.default_rng(seed)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Save an experience."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        indices = self.rng.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.memory[i] for i in indices))

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent:
    """DQN Agent for the trading environment."""

    def __init__(self, config: AgentConfig, input_size: int, seed: Optional[int] = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.n_actions = len(Action)

        # Feature vector size + 1 for the position status
        self.input_size = input_size + 1

        self.policy_net = SimpleDQN(self.input_size, config.hidden_size, self.n_actions, seed)
        self.target_net = SimpleDQN(self.input_size, config.hidden_size, self.n_actions, seed)
        self.update_target_net()

        self.buffer = ReplayBuffer(config.buffer_size, seed)
        self.epsilon = config.epsilon

    def _observation_to_state(self, observation: Observation) -> np.ndarray:
        """Combine observation features and position into a single state vector."""
        return np.concatenate([observation.features, [observation.position]]).astype(np.float32)

    def select_action(self, observation: Observation) -> int:
        """Choose an action using an epsilon-greedy strategy."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        state = self._observation_to_state(observation)
        q_values, _, _ = self.policy_net.forward(state)
        return int(np.argmax(q_values))

    def remember(
        self,
        observation: Observation,
        action_index: int,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> None:
        """Store experience in replay buffer."""
        state = self._observation_to_state(observation)
        next_state = self._observation_to_state(next_observation)
        self.buffer.push(state, action_index, reward, next_state, done)

    def learn(self) -> None:
        """Update policy network by sampling from the replay buffer."""
        if len(self.buffer) < self.config.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)

        # Get Q-values for next states from the target network
        next_q_values, _, _ = self.target_net.forward(next_states)
        best_next_q = np.max(next_q_values, axis=1)

        # Zero out Q-values for terminal states
        best_next_q[dones] = 0.0

        # Calculate target Q-values
        targets = rewards + self.config.discount * best_next_q

        # Create a full target matrix, where only the taken actions are updated
        # All other actions will have an error of 0
        target_q_matrix, _, _ = self.policy_net.forward(states)
        rows = np.arange(self.config.batch_size)
        target_q_matrix[rows, actions] = targets

        # Update the policy network
        self.policy_net.update(
            states,
            actions,
            target_q_matrix,
            self.config.learning_rate,
        )

    def decay_epsilon(self) -> None:
        """Decrease the exploration rate."""
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def update_target_net(self) -> None:
        """Copy weights from the policy network to the target network."""
        self.target_net.set_params(self.policy_net.get_params())

    def reset_episode(self) -> None:
        """Hook called at the start of every episode."""
        # No state to reset within the agent itself for DQN
        return None
