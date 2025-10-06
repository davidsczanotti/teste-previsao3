from __future__ import annotations

import os
import json
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np

from .config import MaCrossoverRlConfig
from .env import MaCrossoverEnv


class RunningNorm:
    def __init__(self, size: int) -> None:
        self.mean = np.zeros(size, dtype=np.float32)
        self.M2 = np.zeros(size, dtype=np.float32)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        # Welford's algorithm
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        var = self.M2 / self.count
        std = np.sqrt(np.maximum(var, 1e-8))
        return (x - self.mean) / std


def _mlp_init(in_size: int, hidden: int, out_size: int, scale: float = 0.02):
    rng = np.random.default_rng(42)
    W1 = (rng.standard_normal((in_size, hidden)) * scale).astype(np.float32)
    b1 = np.zeros((hidden,), dtype=np.float32)
    Wp = (rng.standard_normal((hidden, out_size)) * scale).astype(np.float32)
    bp = np.zeros((out_size,), dtype=np.float32)
    Wv = (rng.standard_normal((hidden, 1)) * scale).astype(np.float32)
    bv = np.zeros((1,), dtype=np.float32)
    return [W1, b1, Wp, bp, Wv, bv]


def _forward(params, x: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
    W1, b1, Wp, bp, Wv, bv = params
    z1 = x @ W1 + b1
    h = np.tanh(z1)
    logits = h @ Wp + bp
    ex = np.exp(logits - np.max(logits))
    probs = ex / (np.sum(ex) + 1e-8)
    v = float(np.dot(h, Wv[:, 0]) + bv[0])
    cache = {"x": x, "z1": z1, "h": h, "probs": probs, "v": v}
    return probs, v, cache


def _backward_policy(params, cache, action: int, coeff: float):
    """Gradients for policy head with coefficient coeff.
    coeff encapsulates -adv * ratio when unclipped; otherwise 0.
    Returns gradients dW1, db1, dWp, dbp (value grads are separate).
    """
    W1, b1, Wp, bp, Wv, bv = params
    x = cache["x"]
    h = cache["h"]
    probs = cache["probs"].copy()
    dlogits = probs
    dlogits[action] -= 1.0
    dlogits *= coeff  # coeff already carries the sign

    dWp = np.outer(h, dlogits)
    dbp = dlogits
    dh = dlogits @ Wp.T

    dz1 = dh * (1 - np.tanh(cache["z1"]) ** 2)
    dW1 = np.outer(x, dz1)
    db1 = dz1
    return dW1, db1, dWp, dbp


def _backward_value(params, cache, target: float):
    W1, b1, Wp, bp, Wv, bv = params
    x = cache["x"]
    h = cache["h"]
    v = cache["v"]
    # 0.5*(target - v)^2 -> grad = -(target - v)
    dv = -(target - v)
    dWv = np.outer(h, np.array([dv], dtype=np.float32))
    dbv = np.array([dv], dtype=np.float32)
    dh = (Wv @ np.array([dv], dtype=np.float32)).reshape(-1)
    dz1 = dh * (1 - np.tanh(cache["z1"]) ** 2)
    dW1 = np.outer(x, dz1)
    db1 = dz1
    return dW1, db1, dWv, dbv


def _apply(params, grads, lr: float, clip: float = 1.0):
    W1, b1, Wp, bp, Wv, bv = params
    gW1, gb1, gWp, gbp, gWv, gbv = grads

    def gnorm(gs):
        tot = 0.0
        for g in gs:
            tot += float(np.sum(g * g))
        return np.sqrt(tot)

    gn = gnorm([gW1, gb1, gWp, gbp, gWv, gbv])
    scale = 1.0
    if clip and gn > clip:
        scale = clip / (gn + 1e-8)
    gW1 *= scale; gb1 *= scale; gWp *= scale; gbp *= scale; gWv *= scale; gbv *= scale

    W1 -= lr * gW1
    b1 -= lr * gb1
    Wp -= lr * gWp
    bp -= lr * gbp
    Wv -= lr * gWv
    bv -= lr * gbv
    return [W1, b1, Wp, bp, Wv, bv]


def ppo_train(cfg: Optional[MaCrossoverRlConfig] = None, model_path: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
    cfg = cfg or MaCrossoverRlConfig()
    env = MaCrossoverEnv(
        symbol=cfg.ticker,
        interval=cfg.interval,
        days=cfg.days,
        lot_size=cfg.lot_size,
        fee_rate=cfg.fee_rate,
        slippage_bps=cfg.slippage_bps,
        action_cost_open=cfg.action_cost_open,
        action_cost_close=cfg.action_cost_close,
        min_hold_bars=cfg.min_hold_bars,
        reopen_cooldown_bars=cfg.reopen_cooldown_bars,
        max_position_bars=cfg.max_position_bars,
        long_only=cfg.long_only,
        m2m_weight=cfg.m2m_weight,
        exec_at_next_open=cfg.exec_next_open,
        switch_penalty=cfg.switch_penalty,
        switch_window_bars=cfg.switch_window_bars,
        episode_len=cfg.episode_len,
        random_start=cfg.random_start,
        idle_penalty=cfg.idle_penalty,
        idle_grace_bars=cfg.idle_grace_bars,
        idle_ramp=cfg.idle_ramp,
        reward_atr_norm=cfg.reward_atr_norm,
        atr_period=cfg.atr_period,
        ma_short_window=cfg.ma_short_window,
        ma_mid_window=cfg.ma_mid_window,
        ma_long_window=cfg.ma_long_window,
        ma_type=cfg.ma_type,
        exit_only=cfg.exit_only,
        gate_on_heuristic=cfg.gate_on_heuristic,
    )

    obs = env.reset(seed=cfg.seed)
    obs_dim = env.observation_size
    act_dim = env.action_size

    hidden = cfg.hidden_size
    normalizer = RunningNorm(obs_dim)
    if model_path:
        try:
            data = np.load(model_path, allow_pickle=True)
            params = [
                data["W1"].astype(np.float32),
                data["b1"].astype(np.float32),
                data["W2"].astype(np.float32),
                data["b2"].astype(np.float32),
                (data["Wv"].astype(np.float32) if "Wv" in data else np.zeros((hidden, 1), np.float32)),
                (data["bv"].astype(np.float32) if "bv" in data else np.zeros((1,), np.float32)),
            ]
            if params[2].shape[1] != act_dim or params[0].shape[0] != obs_dim:
                params = _mlp_init(obs_dim, hidden, act_dim)
            if "norm_mean" in data and "norm_M2" in data and "norm_count" in data:
                normalizer.mean = data["norm_mean"].astype(np.float32)
                normalizer.M2 = data["norm_M2"].astype(np.float32)
                normalizer.count = float(data["norm_count"][0])
        except Exception:
            params = _mlp_init(obs_dim, hidden, act_dim)
    else:
        params = _mlp_init(obs_dim, hidden, act_dim)

    normalizer.update(obs)
    obs = normalizer.normalize(obs)

    # PPO hyperparams
    total_episodes = cfg.episodes
    rollout_len = min(cfg.episode_len or 2048, 4096)
    gamma = cfg.gamma
    gae_lambda = 0.95
    clip_range = 0.2
    lr = cfg.learning_rate
    vf_coef = 0.5
    ent_coef = 0.0
    grad_clip = cfg.grad_clip
    epochs = 4
    minibatch_size = 256

    history = []
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_dir = os.path.join("reports", "agents", "ma_crossover_rl")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"metrics_ppo_{cfg.ticker}_{cfg.interval}_{ts}.jsonl")

    for ep in range(total_episodes):
        obs_buf = np.zeros((rollout_len, obs_dim), dtype=np.float32)
        act_buf = np.zeros((rollout_len,), dtype=np.int32)
        logp_buf = np.zeros((rollout_len,), dtype=np.float32)
        val_buf = np.zeros((rollout_len,), dtype=np.float32)
        rew_buf = np.zeros((rollout_len,), dtype=np.float32)
        done_buf = np.zeros((rollout_len,), dtype=np.float32)

        trades = 0
        steps = 0
        for t in range(rollout_len):
            probs, v, cache = _forward(params, obs)
            # epsilon-greedy over policy probs for exploration
            eps = max(cfg.epsilon_end, cfg.epsilon_start * (1.0 - ep / max(total_episodes - 1, 1)))
            mix = (1.0 - eps) * probs + eps * (np.ones_like(probs) / len(probs))
            mix /= (np.sum(mix) + 1e-8)
            a = int(np.random.choice(len(probs), p=mix))
            logp = float(np.log(probs[a] + 1e-8))

            res = env.step(a)
            if "trade" in res.info or "trade_forced" in res.info:
                trades += 1

            obs_buf[t] = obs
            act_buf[t] = a
            logp_buf[t] = logp
            val_buf[t] = v
            rew_buf[t] = res.reward
            done_buf[t] = 1.0 if res.done else 0.0

            steps += 1
            obs = res.obs
            if res.done:
                obs = env.reset()

        # Bootstrap value
        next_v = _forward(params, obs)[1]

        # Compute GAE advantages
        adv = np.zeros_like(rew_buf)
        lastgaelam = 0.0
        for t in reversed(range(rollout_len)):
            nonterminal = 1.0 - done_buf[t]
            nextval = val_buf[t + 1] if t + 1 < rollout_len else next_v
            delta = rew_buf[t] + gamma * nextval * nonterminal - val_buf[t]
            lastgaelam = delta + gamma * gae_lambda * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + val_buf
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO update
        idx = np.arange(rollout_len)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, rollout_len, minibatch_size):
                mb = idx[start : start + minibatch_size]
                if len(mb) == 0:
                    continue
                gW1 = np.zeros_like(params[0]); gb1 = np.zeros_like(params[1])
                gWp = np.zeros_like(params[2]); gbp = np.zeros_like(params[3])
                gWv = np.zeros_like(params[4]); gbv = np.zeros_like(params[5])

                pol_loss = 0.0; val_loss = 0.0
                for i in mb:
                    ob = obs_buf[i]
                    a = int(act_buf[i])
                    old_logp = logp_buf[i]
                    v_targ = float(ret[i])
                    adv_i = float(adv[i])

                    probs, v_pred, cache = _forward(params, ob)
                    logp_new = float(np.log(probs[a] + 1e-8))
                    ratio = np.exp(logp_new - old_logp)
                    clipped_ratio = np.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)

                    # Choose unclipped vs clipped branch depending on advantage sign
                    use_unclipped = False
                    if adv_i >= 0.0:
                        use_unclipped = ratio <= clipped_ratio
                    else:
                        use_unclipped = ratio >= clipped_ratio

                    coeff = -adv_i * ratio if use_unclipped else 0.0
                    dW1p, db1p, dWp, dbp = _backward_policy(params, cache, a, coeff)
                    gW1 += dW1p; gb1 += db1p; gWp += dWp; gbp += dbp
                    # value grads
                    dW1v, db1v, dWv, dbv = _backward_value(params, cache, v_targ)
                    gW1 += vf_coef * dW1v; gb1 += vf_coef * db1v
                    gWv += vf_coef * dWv; gbv += vf_coef * dbv

                    # losses for logging
                    unclipped = ratio * adv_i
                    clipped = clipped_ratio * adv_i
                    obj = min(unclipped, clipped)
                    pol_loss += -obj
                    val_loss += 0.5 * (v_targ - v_pred) ** 2

                params = _apply(params, (gW1, gb1, gWp, gbp, gWv, gbv), lr, clip=grad_clip)

        ep_reward = float(np.sum(rew_buf))
        history.append({"episode": ep + 1, "steps": steps, "reward": ep_reward, "trades": trades})
        if (ep + 1) % 5 == 0:
            print(f"PPO Episode {ep+1}/{total_episodes} | steps={steps} reward={ep_reward:.2f} trades={trades}")
        try:
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(history[-1]) + "\n")
        except Exception:
            pass

    # Greedy eval
    eval_res = env.run_episode(lambda o: _forward(params, o)[0], max_steps=cfg.max_steps)
    print(f"PPO Greedy evaluation: reward={eval_res['reward']:.2f} steps={eval_res['steps']} trades={eval_res['trades']}")

    out_dir = os.path.join("reports", "agents", "ma_crossover_rl")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cfg.ticker}_{cfg.interval}.npz")
    np.savez(
        out_path,
        W1=params[0], b1=params[1], W2=params[2], b2=params[3], Wv=params[4], bv=params[5],
        config=np.array([str(asdict(cfg))], dtype=object),
    )
    print(f"Saved PPO agent to {out_path}")
    return {"history": history, "eval": eval_res, "model_path": out_path}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train PPO agent (NumPy) on MA crossover env")
    p.add_argument("--ticker", default="BTCUSDT")
    p.add_argument("--interval", default="15m")
    p.add_argument("--days", type=int, default=3650)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--episode_len", type=int, default=4096)
    p.add_argument("--long_only", action="store_true")
    # costs/dynamics
    p.add_argument("--fee_rate", type=float, default=0.0005)
    p.add_argument("--action_cost_open", type=float, default=0.0)
    p.add_argument("--action_cost_close", type=float, default=0.0)
    p.add_argument("--m2m_weight", type=float, default=0.2)
    p.add_argument("--min_hold_bars", type=int, default=3)
    p.add_argument("--reopen_cooldown_bars", type=int, default=1)
    p.add_argument("--exec_next_open", action="store_true")
    p.add_argument("--switch_penalty", type=float, default=0.02)
    p.add_argument("--switch_window_bars", type=int, default=5)
    p.add_argument("--reward_atr_norm", action="store_true")
    p.add_argument("--atr_period", type=int, default=14)
    p.add_argument("--ma_short_window", type=int, default=7)
    p.add_argument("--ma_mid_window", type=int, default=40)
    p.add_argument("--ma_long_window", type=int, default=120)
    p.add_argument("--ma_type", choices=["sma", "ema"], default="sma")
    p.add_argument("--gate_on_heuristic", action="store_true")
    # exploration
    p.add_argument("--epsilon_start", type=float, default=0.2)
    p.add_argument("--epsilon_end", type=float, default=0.02)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()

    cfg = MaCrossoverRlConfig(
        ticker=args.ticker,
        interval=args.interval,
        days=args.days,
        long_only=bool(args.long_only),
        episodes=args.episodes,
        hidden_size=args.hidden,
        learning_rate=args.lr,
        episode_len=args.episode_len,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        fee_rate=args.fee_rate,
        action_cost_open=args.action_cost_open,
        action_cost_close=args.action_cost_close,
        m2m_weight=args.m2m_weight,
        min_hold_bars=args.min_hold_bars,
        reopen_cooldown_bars=args.reopen_cooldown_bars,
        exec_next_open=args.exec_next_open,
        switch_penalty=args.switch_penalty,
        switch_window_bars=args.switch_window_bars,
        reward_atr_norm=args.reward_atr_norm,
        atr_period=args.atr_period,
        ma_short_window=args.ma_short_window,
        ma_mid_window=args.ma_mid_window,
        ma_long_window=args.ma_long_window,
        ma_type=args.ma_type,
        gate_on_heuristic=args.gate_on_heuristic,
    )
    ppo_train(cfg, model_path=args.model)
