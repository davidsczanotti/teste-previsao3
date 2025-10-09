from __future__ import annotations

from typing import List

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation function."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the ReLU function."""
    return (x > 0).astype(x.dtype)


class SimpleDQN:
    """A simple MLP for approximating Q-values, implemented with NumPy."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((input_size, hidden_size), dtype=np.float32) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden_size, output_size), dtype=np.float32) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs a forward pass, returning intermediate values for backprop."""
        # Ensure x is 2D for batch processing
        if x.ndim == 1:
            x = x.reshape(1, -1)

        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        q_values = a1 @ self.W2 + self.b2
        return q_values, a1, z1

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> None:
        """Update model weights using backpropagation."""
        q_values, a1, z1 = self.forward(states)

        # Create a mask for the actions taken
        action_mask = np.eye(self.b2.shape[0])[actions]

        # Calculate error only for the Q-values of the actions taken
        errors = (q_values - targets) * action_mask

        # Backpropagate error
        grad_W2 = a1.T @ errors
        grad_b2 = np.sum(errors, axis=0)

        grad_a1 = errors @ self.W2.T
        grad_z1 = grad_a1 * relu_derivative(z1)
        grad_W1 = states.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)

        # Update weights
        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1
        self.W2 -= learning_rate * grad_W2
        self.b2 -= learning_rate * grad_b2

    def get_params(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: List[np.ndarray]) -> None:
        self.W1, self.b1, self.W2, self.b2 = params
