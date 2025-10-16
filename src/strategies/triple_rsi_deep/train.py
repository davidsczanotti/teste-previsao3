from __future__ import annotations

import os
from typing import Dict, Any, Optional, List, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from .config import DeepTripleRsiConfig
from .env import TripleRsiEnv

# --- Advanced Transformer-based Actor-Critic Model ---

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class TransformerActorCritic(nn.Module):
    """
    Advanced Transformer-based Actor-Critic model with multi-head attention.
    Incorporates temporal dependencies and advanced feature processing.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3,
                 num_heads: int = 8, dropout: float = 0.1):
        super(TransformerActorCritic, self).__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()

        # Handle single observation (add batch and sequence dimensions)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, input_dim]
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]

        # Project input to hidden dimension
        x = self.input_projection(x)  # [batch, seq_len, hidden_dim]

        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)

        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)  # [batch, seq_len, hidden_dim]

        # Use the last sequence element for prediction
        features = transformer_out[:, -1, :]  # [batch, hidden_dim]

        # Actor: get action logits
        action_logits = self.policy_head(features)
        action_dist = Categorical(logits=action_logits)

        # Critic: get state value
        state_value = self.value_head(features).squeeze(-1)

        return action_dist, state_value

class EnsembleActorCritic(nn.Module):
    """
    Ensemble of models for improved stability and performance.
    """
    def __init__(self, models: List[nn.Module]):
        super(EnsembleActorCritic, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        # Average predictions across ensemble
        action_dists = []
        state_values = []

        for model in self.models:
            action_dist, state_value = model(x)
            action_dists.append(action_dist)
            state_values.append(state_value)

        # Average logits for action distribution
        avg_logits = torch.stack([dist.logits for dist in action_dists]).mean(0)
        avg_action_dist = Categorical(logits=avg_logits)

        # Average state values
        avg_state_value = torch.stack(state_values).mean(0)

        return avg_action_dist, avg_state_value

# --- Legacy MLP Model for Backward Compatibility ---

class MLPActorCritic(nn.Module):
    """
    Enhanced MLP Actor-Critic with skip connections and advanced architecture.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, use_skip: bool = True):
        super(MLPActorCritic, self).__init__()

        self.layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList() if use_skip else None

        # Build hidden layers
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_skip and i > 0:
                if self.skip_connections is not None:
                    self.skip_connections.append(nn.Linear(input_dim, hidden_dim))
            prev_dim = hidden_dim

        self.policy_head = nn.Linear(prev_dim, output_dim)
        self.value_head = nn.Linear(prev_dim, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()

        original_x = x.clone()

        # Forward through hidden layers
        for i, layer in enumerate(self.layers):
            x = self.relu(layer(x))
            if self.skip_connections and i > 0:
                skip_x = self.skip_connections[i-1](original_x)
                x = x + skip_x  # Residual connection
            x = self.dropout(x)

        # Actor: get action logits
        action_logits = self.policy_head(x)
        action_dist = Categorical(logits=action_logits)

        # Critic: get state value
        state_value = self.value_head(x).squeeze(-1)

        return action_dist, state_value


def _calculate_gae(rewards: list, values: list, dones: list, gamma: float, lam: float) -> list:
    """Calculate Generalized Advantage Estimation (GAE)."""
    advantages = []
    last_advantage = 0
    for i in reversed(range(len(rewards))):
        if dones[i]:
            delta = rewards[i] - values[i]
            last_advantage = delta
        else:
            delta = rewards[i] + gamma * values[i+1] - values[i]
            last_advantage = delta + gamma * lam * last_advantage
        advantages.insert(0, last_advantage)
    return advantages

# --- Main PPO Training Function ---

def train(config: Optional[DeepTripleRsiConfig] = None, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Trains the Actor-Critic agent using Proximal Policy Optimization (PPO).
    """
    cfg = config or DeepTripleRsiConfig()
    
    # Setup environment
    env = TripleRsiEnv(config=cfg)

    # Setup model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # The environment needs to be reset to determine observation_size
    env.reset(seed=cfg.seed)

    # Choose model architecture based on config
    if cfg.use_transformer:
        model = TransformerActorCritic(
            input_dim=env.observation_size,
            hidden_dim=cfg.transformer_dim,
            output_dim=env.action_size,
            num_layers=cfg.transformer_layers,
            num_heads=cfg.transformer_heads,
            dropout=cfg.dropout
        ).to(device)
    else:
        model = MLPActorCritic(
            input_dim=env.observation_size,
            hidden_dims=cfg.mlp_hidden_sizes,
            output_dim=env.action_size,
            use_skip=cfg.use_skip_connections
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Could not load model, starting from scratch. Error: {e}")

    # --- PPO Training Loop ---
    history = []
    for ep in range(cfg.episodes):
        # --- 1. Collect Rollout Data ---
        obs_list, reward_list, action_list, log_prob_list, values_list, done_list = [], [], [], [], [], []
        
        obs = env.reset(seed=cfg.seed + ep)
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            
            with torch.no_grad():
                action_dist, state_value = model(obs_tensor)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

            res = env.step(action.item())
            
            # Store rollout data
            obs_list.append(obs)
            action_list.append(action.cpu().numpy())
            reward_list.append(res.reward)
            log_prob_list.append(log_prob.cpu().numpy())
            values_list.append(state_value.cpu().numpy())
            done_list.append(res.done)
            
            obs = res.obs
            done = res.done
            episode_reward += res.reward
            steps += 1
            if cfg.max_steps is not None and steps >= cfg.max_steps:
                break
        
        # --- 2. Calculate Advantages and Returns ---
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            _, last_value = model(obs_tensor)
            values_list.append(last_value.cpu().numpy())

        advantages = _calculate_gae(reward_list, values_list, done_list, cfg.gamma, 0.95) # Using a standard lambda for GAE
        returns = (torch.from_numpy(np.array(advantages)) + torch.from_numpy(np.array(values_list[:-1]))).float().to(device)
        advantages = torch.from_numpy(np.array(advantages)).float().to(device)

        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 3. PPO Update Step ---
        dataset = TensorDataset(
            torch.from_numpy(np.array(obs_list)),
            torch.from_numpy(np.array(action_list)),
            torch.from_numpy(np.array(log_prob_list)),
            advantages,
            returns
        )
        loader = DataLoader(dataset, batch_size=cfg.ppo_batch_size, shuffle=True)

        for _ in range(cfg.ppo_epochs):
            for batch_obs, batch_actions, batch_log_probs, batch_advantages, batch_returns in loader:
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)
                batch_log_probs = batch_log_probs.to(device)
                batch_advantages = batch_advantages.to(device)
                batch_returns = batch_returns.to(device)

                # Get new action distributions and values
                new_dist, new_values = model(batch_obs)
                new_log_probs = new_dist.log_prob(batch_actions.squeeze())
                entropy = new_dist.entropy().mean()

                # Calculate PPO ratio
                ratio = (new_log_probs - batch_log_probs.squeeze()).exp()

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_clip_epsilon, 1.0 + cfg.ppo_clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(new_values, batch_returns.squeeze(-1))

                # Total loss
                total_loss = policy_loss + 0.5 * value_loss - cfg.entropy_beta * entropy

                # Update
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

        # --- 4. Logging ---
        sharpe_ratio = env.calculate_sharpe_ratio()
        history.append({"episode": ep + 1, "steps": steps, "reward": episode_reward, "sharpe_ratio": sharpe_ratio, "entropy": entropy.item()})
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep+1}/{cfg.episodes} | Steps={steps} | Reward={episode_reward:.2f} | Sharpe={sharpe_ratio:.2f} | Ent={entropy.item():.3f}")

    # --- Final Evaluation ---
    print("\n--- Running Final Evaluation ---")
    eval_env = TripleRsiEnv(config=cfg)
    obs = eval_env.reset(seed=4242) # Use a fixed seed for eval
    done = False
    while not done:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            action_dist, _ = model(obs_tensor)
            action = action_dist.probs.argmax().item() # Greedy action
        res = eval_env.step(action)
        obs = res.obs
        done = res.done
        if cfg.max_steps is not None and eval_env._i >= cfg.max_steps:
            break
            
    eval_sharpe = eval_env.calculate_sharpe_ratio()
    eval_reward = eval_env.portfolio_values[-1] - eval_env.initial_capital
    print(f"Greedy evaluation: Reward={eval_reward:.2f}, Sharpe Ratio={eval_sharpe:.2f}\n")

    # --- Save Model ---
    out_dir = os.path.join("reports", "agents", "triple_rsi_deep")
    os.makedirs(out_dir, exist_ok=True)
    final_model_path = os.path.join(out_dir, f"{cfg.symbol}_{cfg.interval}_ppo.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved agent to {final_model_path}")

    return {"history": history, "eval": {"reward": eval_reward, "sharpe": eval_sharpe}, "model_path": final_model_path}


if __name__ == "__main__":
    train()
