from __future__ import annotations

import os
import numpy as np
from typing import Dict, Any

from .config import DeepTripleRsiConfig
from .env import TripleRsiEnv
from .train import _mlp_init, _mlp_forward, _mlp_backward, _apply_grads


def _build_expert_dataset(env: TripleRsiEnv, lower: float, upper: float, min_hold: int = 1) -> Dict[str, np.ndarray]:
    env._ensure_data()  # prepare internal arrays
    feats = env._features.copy()  # [N, obs_dim - 2] (without pos flags)
    N = feats.shape[0]
    st = feats[:, 3]  # stoch_k normalized (0..1)
    prev = np.concatenate([[st[0]], st[:-1]])
    cross_down_upper = ((prev > upper) & (st <= upper))
    cross_up_lower = ((prev < lower) & (st >= lower))
    cross_up_upper = ((prev < upper) & (st >= upper))
    cross_down_lower = ((prev > lower) & (st <= lower))

    obs_list = []
    act_list = []
    pos = 0
    bars_since_entry = 0
    for i in range(N - 1):
        # observation at i (append pos flags)
        base = feats[i]
        pos_long = 1.0 if pos == 1 else 0.0
        pos_short = 1.0 if pos == -1 else 0.0
        obs = np.concatenate([base, np.array([pos_long, pos_short], dtype=np.float32)]).astype(np.float32)

        action = 0
        if pos == 0:
            if cross_up_lower[i]:
                action = 1  # open long
                pos = 1
                bars_since_entry = 0
            elif cross_down_upper[i]:
                if not env.long_only:
                    action = 3  # open short
                    pos = -1
                    bars_since_entry = 0
                else:
                    action = 0
        elif pos == 1:
            if bars_since_entry >= min_hold and (cross_up_upper[i] or cross_down_lower[i]):
                action = 2  # close long
                pos = 0
                bars_since_entry = 0
            else:
                bars_since_entry += 1
        elif pos == -1:
            if not env.long_only:
                if bars_since_entry >= min_hold and (cross_down_lower[i] or cross_up_upper[i]):
                    action = 4  # close short
                    pos = 0
                    bars_since_entry = 0
                else:
                    bars_since_entry += 1

        # Collect with downsampled holds
        if action == 0:
            if (i % 10) != 0:
                continue
        obs_list.append(obs)
        act_list.append(action)

    X = np.stack(obs_list).astype(np.float32)
    y = np.array(act_list, dtype=np.int64)
    return {"X": X, "y": y}


def pretrain(cfg: DeepTripleRsiConfig, epochs: int = 10, lr: float = 1e-3, out_path: str | None = None) -> Dict[str, Any]:
    env = TripleRsiEnv(
        symbol=cfg.ticker,
        interval=cfg.interval,
        days=cfg.days,
        short_period=cfg.short_period,
        med_period=cfg.med_period,
        long_period=cfg.long_period,
        stoch_period=cfg.stoch_period,
        stoch_upper=cfg.stoch_upper,
        stoch_lower=cfg.stoch_lower,
        lot_size=cfg.lot_size,
        fee_rate=cfg.fee_rate,
        long_only=cfg.long_only,
        episode_len=None,
        random_start=False,
    )
    data = _build_expert_dataset(env, lower=cfg.stoch_lower, upper=cfg.stoch_upper, min_hold=cfg.min_hold_bars)
    X, y = data["X"], data["y"]
    in_size = X.shape[1]
    out_size = env.action_size  # 3 if long_only, else 5
    hidden = cfg.hidden_size
    W1, b1, W2, b2, Wv, bv = _mlp_init(in_size, hidden, out_size)
    params = [W1, b1, W2, b2, Wv, bv]

    rng = np.random.default_rng(cfg.seed)
    batch_size = 256
    steps_per_epoch = max(1, X.shape[0] // batch_size)
    # Compute class weights (inverse frequency)
    classes, counts = np.unique(y, return_counts=True)
    freq = {int(c): float(n) for c, n in zip(classes, counts)}
    total = float(len(y))
    weights = {c: (total / (len(classes) * n)) for c, n in freq.items()}

    for ep in range(epochs):
        idx = rng.permutation(len(X))
        Xs = X[idx]
        ys = y[idx]
        for i in range(steps_per_epoch):
            bX = Xs[i * batch_size : (i + 1) * batch_size]
            by = ys[i * batch_size : (i + 1) * batch_size]
            if bX.size == 0:
                continue
            # accumulate grads
            W1, b1, W2, b2, Wv, bv = params
            gW1 = np.zeros_like(W1); gb1 = np.zeros_like(b1)
            gW2 = np.zeros_like(W2); gb2 = np.zeros_like(b2)
            gWv = np.zeros_like(Wv); gbv = np.zeros_like(bv)
            loss = 0.0
            for xi, ti in zip(bX, by):
                probs, v, cache = _mlp_forward(params, xi)
                # cross-entropy: -log p_ti
                w = float(weights.get(int(ti), 1.0))
                loss -= w * float(np.log(probs[ti] + 1e-8))
                dW1, db1, dW2, db2, dWv, dbv = _mlp_backward(params, cache, int(ti), advantage=1.0 * w, value_error=0.0)
                gW1 += dW1; gb1 += db1
                gW2 += dW2; gb2 += db2
                gWv += dWv; gbv += dbv
            params = _apply_grads(params, (gW1, gb1, gW2, gb2, gWv, gbv), lr, clip=1.0)
        print(f"Pretrain epoch {ep+1}/{epochs} | loss ~ {loss/ max(1,batch_size):.4f}")

    out_dir = os.path.join("reports", "agents", "triple_rsi_deep")
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_path or os.path.join(out_dir, f"{cfg.ticker}_{cfg.interval}.npz")
    np.savez(out_path, W1=params[0], b1=params[1], W2=params[2], b2=params[3], Wv=params[4], bv=params[5])
    print(f"Saved pretrained model to {out_path}")
    return {"samples": len(X), "model_path": out_path}


if __name__ == "__main__":
    cfg = DeepTripleRsiConfig()
    pretrain(cfg)
