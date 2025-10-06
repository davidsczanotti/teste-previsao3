from __future__ import annotations

import os
from dataclasses import asdict
from typing import Tuple, Dict, Any, Optional
import numpy as np

from .config import DeepTripleRsiConfig
from .env import TripleRsiEnv


def _mlp_init(in_size: int, hidden: int, out_size: int, scale: float = 0.02):
    rng = np.random.default_rng(42)
    W1 = (rng.standard_normal((in_size, hidden)) * scale).astype(np.float32)
    b1 = np.zeros((hidden,), dtype=np.float32)
    W2 = (rng.standard_normal((hidden, out_size)) * scale).astype(np.float32)
    b2 = np.zeros((out_size,), dtype=np.float32)
    # value head
    Wv = (rng.standard_normal((hidden, 1)) * scale).astype(np.float32)
    bv = np.zeros((1,), dtype=np.float32)
    return W1, b1, W2, b2, Wv, bv


def _mlp_forward(params, x: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
    W1, b1, W2, b2, Wv, bv = params
    z1 = x @ W1 + b1
    h1 = np.tanh(z1)
    logits = h1 @ W2 + b2
    ex = np.exp(logits - np.max(logits))
    probs = ex / np.sum(ex)
    v = float(h1 @ Wv + bv)
    cache = {"x": x, "z1": z1, "h1": h1, "probs": probs, "v": v}
    return probs, v, cache


def _mlp_backward(params, cache, action: int, advantage: float, value_error: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1, b1, W2, b2, Wv, bv = params
    x = cache["x"]
    h1 = cache["h1"]
    probs = cache["probs"].copy()

    # policy gradient for softmax: dL/dlogits = probs - onehot(action)
    dlogits = probs
    dlogits[action] -= 1.0
    dlogits *= -advantage  # ascent: negate to compute gradient of negative objective

    dW2 = np.outer(h1, dlogits)
    db2 = dlogits

    dh1 = dlogits @ W2.T
    # value head grads
    dv = value_error  # derivative of 0.5*(ret - v)^2 -> -(ret - v)
    dWv = np.outer(h1, np.array([dv], dtype=np.float32))
    dbv = np.array([dv], dtype=np.float32)
    dh1 += (Wv @ np.array([dv], dtype=np.float32)).reshape(-1)

    dz1 = dh1 * (1 - np.tanh(cache["z1"]) ** 2)
    dW1 = np.outer(x, dz1)
    db1 = dz1

    return dW1, db1, dW2, db2, dWv, dbv


def _apply_grads(params, grads, lr: float, clip: float = 1.0):
    W1, b1, W2, b2, Wv, bv = params
    gW1, gb1, gW2, gb2, gWv, gbv = grads
    # clip by global norm
    def _gnorm(gs):
        tot = 0.0
        for g in gs:
            tot += float(np.sum(g * g))
        return np.sqrt(tot)
    gnorm = _gnorm([gW1, gb1, gW2, gb2, gWv, gbv])
    scale = 1.0
    if clip and gnorm > clip:
        scale = clip / (gnorm + 1e-8)
    gW1 *= scale; gb1 *= scale; gW2 *= scale; gb2 *= scale; gWv *= scale; gbv *= scale
    W1 -= lr * gW1
    b1 -= lr * gb1
    W2 -= lr * gW2
    b2 -= lr * gb2
    Wv -= lr * gWv
    bv -= lr * gbv
    return W1, b1, W2, b2, Wv, bv


def train(config: Optional[DeepTripleRsiConfig] = None, model_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = config or DeepTripleRsiConfig()
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
        slippage_bps=cfg.slippage_bps,
        action_cost_open=cfg.action_cost_open,
        action_cost_close=cfg.action_cost_close,
        invalid_action_penalty=cfg.invalid_action_penalty,
        min_hold_bars=cfg.min_hold_bars,
        reopen_cooldown_bars=cfg.reopen_cooldown_bars,
        max_position_bars=cfg.max_position_bars,
        negative_close_boost=0.2,
        long_only=cfg.long_only,
        gate_enabled=cfg.gate_enabled,
        gate_margin=cfg.gate_margin,
        gate_recent_k=cfg.gate_recent_k,
        realized_weight=cfg.realized_weight,
        m2m_weight=cfg.m2m_weight,
        midrange_penalty=cfg.midrange_penalty,
        close_bonus_factor=cfg.close_bonus_factor,
        episode_len=cfg.episode_len,
        random_start=cfg.random_start,
    )

    obs = env.reset(seed=cfg.seed)
    in_size = env.observation_size
    out_size = env.action_size
    hidden = cfg.hidden_size
    # Initialize or warm-start
    if model_path:
        try:
            data = np.load(model_path, allow_pickle=True)
            W1 = data["W1"].astype(np.float32)
            b1 = data["b1"].astype(np.float32)
            W2 = data["W2"].astype(np.float32)
            b2 = data["b2"].astype(np.float32)
            if "Wv" in data:
                Wv = data["Wv"].astype(np.float32)
                bv = data["bv"].astype(np.float32)
            else:
                _, _, _, _, Wv, bv = _mlp_init(in_size, hidden, out_size)
            # handle output-size mismatch (e.g., switching to long-only)
            if W2.shape[1] != out_size:
                _, _, W2, b2, Wv, bv = _mlp_init(in_size, hidden, out_size)
        except Exception:
            W1, b1, W2, b2, Wv, bv = _mlp_init(in_size, hidden, out_size)
    else:
        W1, b1, W2, b2, Wv, bv = _mlp_init(in_size, hidden, out_size)
    params = [W1, b1, W2, b2, Wv, bv]

    gamma = cfg.gamma
    lr = cfg.learning_rate
    entropy_beta_start = cfg.entropy_beta
    entropy_beta_end = cfg.entropy_beta_end
    grad_clip = cfg.grad_clip

    def policy(obs_arr: np.ndarray) -> np.ndarray:
        probs, _, _ = _mlp_forward(params, obs_arr)
        return probs

    history = []
    for ep in range(cfg.episodes):
        obs = env.reset()
        episode_obs = []
        episode_act = []
        episode_rew = []
        episode_cache = []

        steps = 0
        while True:
            probs, v_pred, cache = _mlp_forward(params, obs)
            # Epsilon-greedy sampling (decay over episodes)
            frac_left = 1.0 - (ep / max(cfg.episodes - 1, 1))
            eps = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * frac_left
            K = len(probs)
            mix = (1.0 - eps) * probs + eps * (np.ones_like(probs) / K)
            mix = mix / (np.sum(mix) + 1e-8)
            action = int(np.random.choice(K, p=mix))
            res = env.step(action)

            episode_obs.append(obs)
            episode_act.append(action)
            episode_rew.append(res.reward)
            episode_cache.append(cache)
            # store value prediction
            # we will recompute cache['v'] already contains v

            steps += 1
            if res.done or (cfg.max_steps is not None and steps >= cfg.max_steps):
                break
            obs = res.obs

        # Compute returns and advantages (REINFORCE)
        R = 0.0
        returns = []
        for r in reversed(episode_rew):
            R = r + gamma * R
            returns.append(R)
        returns.reverse()
        returns = np.array(returns, dtype=np.float32)
        values = np.array([c["v"] for c in episode_cache], dtype=np.float32)
        advantages = returns - values
        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Accumulate grads over the episode
        W1, b1, W2, b2, Wv, bv = params
        gW1 = np.zeros_like(W1)
        gb1 = np.zeros_like(b1)
        gW2 = np.zeros_like(W2)
        gb2 = np.zeros_like(b2)
        gWv = np.zeros_like(Wv)
        gbv = np.zeros_like(bv)
        entropy = 0.0
        for obs_t, act_t, adv_t, ret_t, cache_t in zip(episode_obs, episode_act, advantages, returns, episode_cache):
            value_err = -(ret_t - cache_t["v"])  # d/dv of 0.5*(ret-v)^2
            dW1, db1, dW2, db2, dWv, dbv = _mlp_backward(params, cache_t, act_t, float(adv_t), float(value_err))
            gW1 += dW1
            gb1 += db1
            gW2 += dW2
            gb2 += db2
            gWv += dWv
            gbv += dbv
            # entropy bonus (encourage exploration)
            p = cache_t["probs"]
            entropy -= np.sum(p * np.log(p + 1e-8))

        # Auxiliary imitation (behavior cloning) toward heuristic rules around stochastic thresholds
        if cfg.bc_weight > 0.0:
            # stronger earlier, decays to 0
            bc_scale = float(cfg.bc_weight) * frac_left
            for cache_t, obs_t in zip(episode_cache, episode_obs):
                # obs: [rsi_s, rsi_m, rsi_l, st, dist_up, dist_low, cross_down_upper, cross_up_lower, pos_long, pos_short]
                st = float(obs_t[3])
                cross_down_upper = bool(obs_t[6] > 0.5)
                cross_up_lower = bool(obs_t[7] > 0.5)
                pos_long = bool(obs_t[-2] > 0.5)
                pos_short = bool(obs_t[-1] > 0.5)
                # heuristic labels
                ex_action = 0
                if not pos_long and not pos_short:
                    if (st <= (cfg.stoch_lower + cfg.gate_margin)) or cross_up_lower:
                        ex_action = 1  # open long
                    elif (st >= (cfg.stoch_upper - cfg.gate_margin)) or cross_down_upper:
                        ex_action = 0 if cfg.long_only else 3  # open short or hold if long-only
                    else:
                        ex_action = 0
                elif pos_long:
                    if st >= cfg.stoch_upper:
                        ex_action = 2  # close long
                    else:
                        ex_action = 0
                elif pos_short:
                    if cfg.long_only:
                        ex_action = 0
                    else:
                        if st <= cfg.stoch_lower:
                            ex_action = 4  # close short
                        else:
                            ex_action = 0
                dW1, db1, dW2, db2, dWv, dbv = _mlp_backward(params, cache_t, int(ex_action), float(bc_scale), 0.0)
                gW1 += dW1; gb1 += db1; gW2 += dW2; gb2 += db2

        # Optional: small L2 regularization and entropy encouragement by scaling grads
        if entropy_beta_start > 0:
            K = out_size
            uniform = 1.0 / K
            mean_p = np.mean([c["probs"] for c in episode_cache], axis=0)
            frac = 1.0 - (ep / max(cfg.episodes - 1, 1))
            entropy_beta = entropy_beta_end + (entropy_beta_start - entropy_beta_end) * frac
            gb2 += -entropy_beta * (uniform - mean_p)

        # apply grads
        params = _apply_grads(params, (gW1, gb1, gW2, gb2, gWv, gbv), lr, clip=grad_clip)

        ep_reward = float(np.sum(episode_rew))
        history.append({"episode": ep + 1, "steps": steps, "reward": ep_reward, "entropy": float(entropy)})
        if (ep + 1) % 5 == 0:
            print(f"Episode {ep+1}/{cfg.episodes} | steps={steps} reward={ep_reward:.2f} ent={entropy:.3f}")

    # Evaluate greedy performance
    eval_res = env.run_episode(lambda o: _mlp_forward(params, o)[0], max_steps=cfg.max_steps)
    print(f"Greedy evaluation: reward={eval_res['reward']:.2f} steps={eval_res['steps']} trades={eval_res['trades']}")

    # Save model
    out_dir = os.path.join("reports", "agents", "triple_rsi_deep")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cfg.ticker}_{cfg.interval}.npz")
    np.savez(
        out_path,
        W1=params[0],
        b1=params[1],
        W2=params[2],
        b2=params[3],
        Wv=params[4],
        bv=params[5],
        config=np.array([str(asdict(cfg))], dtype=object),
    )
    print(f"Saved agent to {out_path}")

    return {"history": history, "eval": eval_res, "model_path": out_path}


if __name__ == "__main__":
    # Default training run for 15m BTCUSDT
    train()
