from __future__ import annotations

import os
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader

from .config import DeepTripleRsiConfig
from .env import TripleRsiEnv

# --- Enhanced PyTorch Actor-Critic Model with Skip Connection ---

class ActorCritic(nn.Module):
    """
    An Actor-Critic model with a shared MLP backbone and a skip connection.
    The skip connection can help with gradient flow for deeper or more complex models.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ActorCritic, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        # Skip connection will be from input to the second layer
        self.skip_connection = nn.Linear(input_dim, hidden_dim)
        
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()

        # Main path
        out1 = self.relu(self.layer1(x))
        # Add skip connection before the second layer's activation
        out2 = self.relu(self.layer2(out1) + self.skip_connection(x))
        
        # Actor: get action logits, then create a distribution
        action_logits = self.policy_head(out2)
        action_dist = Categorical(logits=action_logits)
        
        # Critic: get state value
        state_value = self.value_head(out2)
        
        return action_dist, state_value.squeeze(-1)


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
    model = ActorCritic(
        input_dim=env.observation_size,
        hidden_dim=cfg.hidden_size,
        output_dim=env.action_size
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
        returns = (torch.tensor(advantages, dtype=torch.float32) + torch.tensor(values_list[:-1], dtype=torch.float32)).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)

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
                value_loss = nn.functional.mse_loss(new_values, batch_returns)

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
    eval_reward = sum(eval_env.portfolio_values) - eval_env.initial_capital
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