from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np

from .config import Candle7RlConfig
from .env import Candle7Env


def load_policy(model_path: str):
    """Loads a saved agent and returns (policy_fn, policy_type).

    - For NumPy REINFORCE (.npz saved by train.py): returns softmax probs; argmax is greedy.
    - For PyTorch PPO (.pt): returns logits from the actor head; argmax is greedy.
    """
    if model_path.endswith(".pt"):
        import torch
        from .training.train_ppo_torch import ActorCriticMLP, ActorCriticTransformer, ActorCriticLSTM

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=device)
        cfg_dict = ckpt.get("config", {})
        policy_type = cfg_dict.get("policy_type", "mlp")

        def make_agent(obs_size, act_dim, hidden):
            if policy_type == "transformer":
                return ActorCriticTransformer(
                    seq_input_shape=obs_size["sequential"],
                    non_seq_input_dim=obs_size["non_sequential"],
                    act_dim=act_dim,
                    hidden_size=hidden,
                ).to(device)
            if policy_type == "lstm":
                return ActorCriticLSTM(
                    seq_input_shape=obs_size["sequential"],
                    non_seq_input_dim=obs_size["non_sequential"],
                    act_dim=act_dim,
                    hidden_size=hidden,
                ).to(device)
            return ActorCriticMLP(obs_size, act_dim, hidden).to(device)

        # Build env skeleton just to infer obs/action dims for model init, honoring saved cfg
        tmp_env = Candle7Env(
            symbol=cfg_dict.get("ticker", "BTCUSDT"),
            interval=cfg_dict.get("interval", "15m"),
            days=cfg_dict.get("days", 365),
            obs_format=("structured" if policy_type in ("transformer", "lstm") else "flat"),
            include_mtf=bool(cfg_dict.get("include_mtf", False)),
            mtf_timeframes=tuple(cfg_dict.get("mtf_timeframes", ("1h", "4h"))),
            include_regime_features=bool(cfg_dict.get("include_regime_features", False)),
        )
        obs_size = tmp_env.observation_size
        act_dim = tmp_env.action_size
        hidden = cfg_dict.get("hidden_size", 128)
        agent = make_agent(obs_size, act_dim, hidden)
        agent.load_state_dict(ckpt["model_state_dict"])  # type: ignore
        agent.eval()

        norm_state = ckpt.get("normalizer_state", None)
        if norm_state:
            norm_mean = np.array(norm_state["mean"], dtype=np.float32)
            norm_M2 = np.array(norm_state["M2"], dtype=np.float32)
            norm_count = float(norm_state["count"])
        else:
            norm_mean = None
            norm_M2 = None
            norm_count = 0.0

        def normalize_vec(x: np.ndarray) -> np.ndarray:
            if norm_mean is None:
                return x
            var = norm_M2 / max(norm_count, 1e-4)
            std = np.sqrt(np.maximum(var, 1e-8))
            return (x - norm_mean) / std

        def policy_fn(obs):
            import torch
            with torch.no_grad():
                if policy_type in ("transformer", "lstm"):
                    # obs = {"sequential": [T,F], "non_sequential": [D]}
                    seq_t = torch.tensor(obs["sequential"], dtype=torch.float32).unsqueeze(0).to(device)
                    non_seq = np.asarray(obs["non_sequential"], dtype=np.float32)
                    # normalize only non-seq part
                    if norm_mean is not None:
                        non_seq = normalize_vec(non_seq)
                    non_seq_t = torch.tensor(non_seq, dtype=torch.float32).unsqueeze(0).to(device)
                    # forward
                    if hasattr(agent, "forward_transformer"):
                        base_out = agent.forward_transformer(seq_t, non_seq_t)
                    else:
                        base_out = agent.forward_lstm(seq_t, non_seq_t)
                    logits = agent.actor(base_out)
                    return logits.squeeze(0).cpu().numpy()
                else:
                    x = np.asarray(obs, dtype=np.float32)
                    if norm_mean is not None:
                        x = normalize_vec(x)
                    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = agent.actor(x_t)
                    return logits.squeeze(0).cpu().numpy()

        return policy_fn, policy_type, cfg_dict

    # NumPy model
    from .train import _mlp_forward as np_forward
    data = np.load(model_path, allow_pickle=True)
    params = [
        data["W1"].astype(np.float32),
        data["b1"].astype(np.float32),
        data["W2"].astype(np.float32),
        data["b2"].astype(np.float32),
        data["Wv"].astype(np.float32),
        data["bv"].astype(np.float32),
    ]

    def policy_fn(obs):
        probs, _, _ = np_forward(params, obs)
        return probs

    return policy_fn, "mlp", {}


def run_eval(env: Candle7Env, policy_fn) -> Dict[str, Any]:
    res = env.run_episode(policy_fn, max_steps=env.episode_len)
    return res


def main():
    p = argparse.ArgumentParser(description="Ablation report for Candle7 RL features")
    p.add_argument("--model", required=True, help="Path to model (.npz for NumPy, .pt for PyTorch)")
    p.add_argument("--ticker", default="BTCUSDT")
    p.add_argument("--interval", default="15m")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--episode_len", type=int, default=2048)
    p.add_argument("--include_mtf", action="store_true")
    p.add_argument("--mtf_timeframes", type=str, default="1h,4h")
    p.add_argument("--include_regimes", action="store_true")
    args = p.parse_args()

    policy_fn, policy_type = load_policy(args.model)

    # Build env baseline
    env_kwargs = dict(
        symbol=args.ticker,
        interval=args.interval,
        days=args.days,
        episode_len=args.episode_len,
        random_start=False,
        include_mtf=bool(args.include_mtf),
        mtf_timeframes=tuple([s.strip() for s in args.mtf_timeframes.split(',') if s.strip()]),
        include_regime_features=bool(args.include_regimes),
        obs_format=("structured" if policy_type in ("transformer", "lstm") else "flat"),
    )

    baseline_env = Candle7Env(**env_kwargs)
    baseline = run_eval(baseline_env, policy_fn)

    groups = ["seq", "non_seq_core", "non_seq_mtf", "non_seq_regime", "pos"]
    results = {"baseline": baseline}
    for g in groups:
        env = Candle7Env(**env_kwargs, ablation_groups=[g])
        res = run_eval(env, policy_fn)
        results[g] = res

    # Print simple report
    base_r = baseline["reward"]
    print("Ablation report (reward):")
    print(f"baseline: {base_r:.4f}")
    for g in groups:
        r = results[g]["reward"]
        delta = r - base_r
        pct = (delta / (abs(base_r) + 1e-9)) * 100.0
        print(f"- {g:16s} -> {r:.4f} (Δ {delta:+.4f}, {pct:+.2f}%)")

    # Save JSON
    out_dir = os.path.join("reports", "agents", "candle_pattern7_rl")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ablation_{os.path.basename(args.model)}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved ablation JSON to {out_path}")


if __name__ == "__main__":
    main()
