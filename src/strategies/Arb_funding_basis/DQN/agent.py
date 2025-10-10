
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, seed: int = 0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.qnetwork_local = self._build_model()
        self.qnetwork_target = self._build_model()
        self.update_target_network()
    
    def _build_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
    
    def update_target_network(self):
        self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())
    
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.remember(state, action, reward, next_state, done)
        if done:
            self.update_target_network()
        if len(self.memory) > self.batch_size:
            experiences = self.sample_batch()
            self.learn(experiences)
    
    def sample_batch(self):
        experiences = random.sample(self.memory, self.batch_size)
        states = np.vstack([e[0] for e in experiences if e is not None])
        actions = np.array([e[1] for e in experiences if e is not None], dtype=np.intp)
        rewards = np.vstack([e[2] for e in experiences if e is not None])
        next_states = np.vstack([e[3] for e in experiences if e is not None])
        dones = np.array([e[4] for e in experiences if e is not None], dtype=np.uint8).reshape(-1, 1)
        return (states, actions, rewards, next_states, dones)
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target.predict(next_states, verbose=0)
        Q_targets = rewards + (self.gamma * np.max(Q_targets_next, axis=1, keepdims=True)) * (1 - dones)
        Q_expected = self.qnetwork_local.predict(states, verbose=0)
        indices = np.arange(len(actions), dtype=np.intp)
        Q_expected[indices, actions] = Q_targets.squeeze()
        self.qnetwork_local.fit(states, Q_expected, epochs=1, verbose=0)
    
    def act(self, state: np.ndarray, epsilon: float | None = None) -> int:
        eps = self.epsilon if epsilon is None else epsilon
        if epsilon is not None:
            self.epsilon = epsilon
        state = np.array(state).reshape(1, self.state_size)
        if np.random.rand() <= eps:
            return np.random.randint(self.action_size)
        q_values = self.qnetwork_local.predict(state, verbose=0)[0]
        return int(np.argmax(q_values))
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return
        experiences = self.sample_batch()
        self.learn(experiences)
