from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .config import DeepTripleRsiConfig
from .env import TripleRsiEnv

# --- PyTorch Actor-Critic Model ---

class ActorCritic(nn.Module):
    """
    An Actor-Critic model with a shared MLP backbone.
    - Actor head: outputs action probabilities (policy)
    - Critic head: outputs state value
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ActorCritic, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """
        Forward pass.
        Returns a distribution over actions and the estimated state value.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()

        shared_features = self.shared_layer(x)
        
        # Actor: get action logits, then create a distribution
        action_logits = self.policy_head(shared_features)
        action_dist = Categorical(logits=action_logits)
        
        # Critic: get state value
        state_value = self.value_head(shared_features)
        
        return action_dist, state_value.squeeze(-1)


# --- Main Training Function ---

def train(config: Optional[DeepTripleRsiConfig] = None, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Trains the Actor-Critic agent using PyTorch.
    """
    cfg = config or DeepTripleRsiConfig()
    
    # Filter config to only pass env-specific args
    import inspect
    config_dict = asdict(cfg)
    env_arg_names = [p.name for p in inspect.signature(TripleRsiEnv).parameters.values()]
    env_config = {k: v for k, v in config_dict.items() if k in env_arg_names}

    # Setup environment
    env = TripleRsiEnv(**env_config)

    # Setup model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ActorCritic(
        input_dim=env.observation_size,
        hidden_dim=cfg.hidden_size,
        output_dim=env.action_size
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Load model if path is provided
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Could not load model, starting from scratch. Error: {e}")

    # --- Training Loop ---
    history = []
    for ep in range(cfg.episodes):
        # --- Collect episode data ---
        obs_list, reward_list = [], []
        log_probs_list, values_list, entropy_list = [], [], []
        
        obs = env.reset(seed=cfg.seed + ep)
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            
            # Get action and value from model
            action_dist, state_value = model(obs_tensor)
            
            # Epsilon-greedy exploration
            frac_left = 1.0 - (ep / max(cfg.episodes - 1, 1))
            eps = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * frac_left
            
            if torch.rand(1) < eps:
                action = torch.randint(0, env.action_size, (1,)).item()
            else:
                action = action_dist.sample().item()

            # Store results from the model
            log_probs_list.append(action_dist.log_prob(torch.tensor(action, device=device)))
            values_list.append(state_value)
            entropy_list.append(action_dist.entropy())

            # Step the environment
            res = env.step(action)
            
            obs_list.append(obs)
            reward_list.append(res.reward)
            
            obs = res.obs
            done = res.done
            episode_reward += res.reward
            steps += 1
            if cfg.max_steps is not None and steps >= cfg.max_steps:
                break

        # --- A2C Update Step ---
        
        # Calculate returns (Gt)
        returns = []
        R = 0
        for r in reversed(reward_list):
            R = r + cfg.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        log_probs = torch.stack(log_probs_list)
        values = torch.stack(values_list).squeeze()
        entropy = torch.stack(entropy_list).mean()

        # Normalize returns (optional, but can stabilize)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate advantages
        advantages = returns - values
        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # Calculate losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = nn.functional.mse_loss(values, returns.detach())
        
        # Entropy bonus for exploration
        frac = 1.0 - (ep / max(cfg.episodes - 1, 1))
        entropy_beta = cfg.entropy_beta_end + (cfg.entropy_beta - cfg.entropy_beta_end) * frac
        
        total_loss = policy_loss + 0.5 * value_loss - entropy_beta * entropy

        # Update model
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # --- Logging ---
        history.append({"episode": ep + 1, "steps": steps, "reward": episode_reward, "entropy": entropy.item()})
        if (ep + 1) % 5 == 0:
            print(f"Episode {ep+1}/{cfg.episodes} | steps={steps} reward={episode_reward:.2f} ent={entropy.item():.3f}")

    # --- Final Evaluation ---
    print("\n--- Running Final Evaluation ---")
    eval_env = TripleRsiEnv(**asdict(cfg))
    obs = eval_env.reset(seed=4242) # Use a fixed seed for eval
    done = False
    eval_reward = 0
    eval_steps = 0
    eval_trades = 0
    while not done:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            action_dist, _ = model(obs_tensor)
            action = action_dist.probs.argmax().item() # Greedy action
        
        res = eval_env.step(action)
        obs = res.obs
        done = res.done
        eval_reward += res.reward
        eval_steps += 1
        if "trade" in res.info or "trade_forced" in res.info:
            eval_trades += 1
        if cfg.max_steps is not None and eval_steps >= cfg.max_steps:
            break
            
    eval_res = {"reward": eval_reward, "steps": eval_steps, "trades": eval_trades}
    print(f"Greedy evaluation: reward={eval_res['reward']:.2f} steps={eval_res['steps']} trades={eval_res['trades']}\n")

    # --- Save Model ---
    out_dir = os.path.join("reports", "agents", "triple_rsi_deep")
    os.makedirs(out_dir, exist_ok=True)
    final_model_path = os.path.join(out_dir, f"{cfg.symbol}_{cfg.interval}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved agent to {final_model_path}")

    return {"history": history, "eval": eval_res, "model_path": final_model_path}


if __name__ == "__main__":
    train()