from __future__ import annotations

import argparse
import numpy as np

from .config import DeepTripleRsiConfig
from .env import TripleRsiEnv
from .train import _mlp_forward


def load_policy(path: str):
    data = np.load(path, allow_pickle=True)
    W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]
    Wv = data["Wv"] if "Wv" in data else None
    bv = data["bv"] if "bv" in data else None
    if Wv is None:
        Wv = (b1.reshape(-1, 1) * 0).astype(b1.dtype)  # shape (hidden,1)
    if bv is None:
        bv = (np.zeros((1,), dtype=b1.dtype))
    params = (W1, b1, W2, b2, Wv, bv)
    def policy(o: np.ndarray) -> np.ndarray:
        probs, _, _ = _mlp_forward(params, o)
        return probs
    return policy


def main():
    parser = argparse.ArgumentParser(description="Evaluate Triple RSI RL agent")
    parser.add_argument("--model", type=str, required=True, help="Path to saved .npz model")
    parser.add_argument("--ticker", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="15m")
    parser.add_argument("--days", type=int, default=120)
    args = parser.parse_args()

    policy = load_policy(args.model)
    env = TripleRsiEnv(
        symbol=args.ticker,
        interval=args.interval,
        days=args.days,
        episode_len=None,
        invalid_action_penalty=0.0,  # don't punish invalids in eval; treat as hold
        min_hold_bars=3,
        reopen_cooldown_bars=3,
        lot_size=0.001,
        fee_rate=0.001,
    )
    res = env.run_episode(policy)
    print(res)


if __name__ == "__main__":
    main()
