import random
from collections import deque, namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    """Agente que interage e aprende com o ambiente."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        learning_rate=5e-4,
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        tau=1e-3,
    ):
        """Inicializa um objeto Agente.

        Args:
            state_size (int): Dimensão de cada estado
            action_size (int): Dimensão de cada ação
            seed (int): Semente aleatória
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma  # Fator de desconto
        self.tau = tau  # Para soft update dos pesos da rede alvo

        # Rede Q (Q-Network)
        self.qnetwork_local = self._build_model()
        self.qnetwork_target = self._build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)

        # Replay memory
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def _build_model(self):
        """Constrói uma rede neural para aproximar os valores Q."""
        model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(self.state_size,)),
                Dense(64, activation="relu"),
                Dense(self.action_size, activation="linear"),
            ]
        )
        return model

    def step(self, state, action, reward, next_state, done):
        # Salva a experiência na memória de replay
        self.memory.append(self.experience(state, action, reward, next_state, done))

        # Aprende se houver amostras suficientes na memória
        if len(self.memory) > self.batch_size:
            experiences = random.sample(self.memory, k=self.batch_size)
            self._learn(experiences)

    def act(self, state, eps=0.0):
        """Retorna ações para um dado estado conforme a política atual.

        Args:
            state (array_like): estado atual
            eps (float): epsilon, para a política epsilon-greedy
        """
        state = np.array(state).reshape(1, -1)
        # Exploração Epsilon-Greedy
        if random.random() > eps:
            action_values = self.qnetwork_local(state)
            return np.argmax(action_values[0]).item()
        else:
            return random.choice(np.arange(self.action_size))

    def _learn(self, experiences):
        """Atualiza os pesos da rede usando um lote de experiências."""
        states = tf.convert_to_tensor([e.state for e in experiences if e is not None], dtype=tf.float32)
        actions = tf.convert_to_tensor([e.action for e in experiences if e is not None], dtype=tf.int32)
        rewards = tf.convert_to_tensor([e.reward for e in experiences if e is not None], dtype=tf.float32)
        next_states = tf.convert_to_tensor([e.next_state for e in experiences if e is not None], dtype=tf.float32)
        dones = tf.convert_to_tensor([e.done for e in experiences if e is not None], dtype=tf.float32)

        # Calcula os Q-targets para os próximos estados a partir da rede alvo
        q_targets_next = tf.reduce_max(self.qnetwork_target(next_states), axis=1)
        # Calcula os Q-targets para os estados atuais
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        with tf.GradientTape() as tape:
            # Pega os Q-values esperados da rede local
            q_expected = self.qnetwork_local(states)
            q_expected = tf.gather(q_expected, actions, batch_dims=1)

            loss = tf.keras.losses.MSE(q_targets, tf.squeeze(q_expected))

        grads = tape.gradient(loss, self.qnetwork_local.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.qnetwork_local.trainable_variables))

        # ------------------- atualiza a rede alvo ------------------- #
        self._soft_update(self.qnetwork_local, self.qnetwork_target)

    def _soft_update(self, local_model, target_model):
        """Soft update dos parâmetros do modelo.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        local_weights = local_model.get_weights()
        target_weights = target_model.get_weights()

        new_weights = []
        for local_w, target_w in zip(local_weights, target_weights):
            new_w = self.tau * local_w + (1.0 - self.tau) * target_w
            new_weights.append(new_w)
        target_model.set_weights(new_weights)
