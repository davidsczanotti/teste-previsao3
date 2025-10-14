from __future__ import annotations

import os
import json
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np

from .config import Candle7RlConfig
from .env import Candle7Env


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
    W2 = (rng.standard_normal((hidden, hidden)) * scale).astype(np.float32)
    b2 = np.zeros((hidden,), dtype=np.float32)
    Wp = (rng.standard_normal((hidden, out_size)) * scale).astype(np.float32)
    bp = np.zeros((out_size,), dtype=np.float32)
    Wv = (rng.standard_normal((hidden, 1)) * scale).astype(np.float32)
    bv = np.zeros((1,), dtype=np.float32)
    return [W1, b1, W2, b2, Wp, bp, Wv, bv]


def _forward(params, x: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
    W1, b1, W2, b2, Wp, bp, Wv, bv = params
    z1 = x @ W1 + b1
    h1 = np.tanh(z1)
    z2 = h1 @ W2 + b2
    h2 = np.tanh(z2)
    logits = h2 @ Wp + bp
    ex = np.exp(logits - np.max(logits))
    probs = ex / (np.sum(ex) + 1e-8)
    v = float(np.dot(h2, Wv[:, 0]) + bv[0])
    cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "probs": probs, "v": v}
    return probs, v, cache


def _backward_entropy(params, cache, ent_coef: float):
    W1, b1, W2, b2, Wp, bp, Wv, bv = params
    x, h1, z2, h2, probs = cache["x"], cache["h1"], cache["z2"], cache["h2"], cache["probs"]
    log_probs = np.log(probs + 1e-8)
    dlogits = ent_coef * probs * (log_probs + 1.0)
    dW1, db1, dW2, db2, dWp, dbp = _propagate_grads(params, cache, dlogits)
    return dW1, db1, dW2, db2, dWp, dbp


def _backward_policy(params, cache, action: int, coeff: float):
    W1, b1, W2, b2, Wp, bp, Wv, bv = params
    probs = cache["probs"].copy()
    dlogits = probs
    dlogits[action] -= 1.0
    dlogits *= coeff

    dW1, db1, dW2, db2, dWp, dbp = _propagate_grads(params, cache, dlogits)
    return dW1, db1, dW2, db2, dWp, dbp


def _backward_value(params, cache, target: float):
    W1, b1, W2, b2, Wp, bp, Wv, bv = params
    v = cache["v"]
    dv = -(target - v)
    dWv = np.outer(cache["h2"], np.array([dv], dtype=np.float32))
    dbv = np.array([dv], dtype=np.float32)
    dlogits_from_value = (Wv @ np.array([dv], dtype=np.float32)).reshape(-1)

    dW1, db1, dW2, db2, _, _ = _propagate_grads(params, cache, dlogits_from_value, from_value_head=True)
    return dW1, db1, dW2, db2, dWv, dbv


def _propagate_grads(params, cache, dlogits, from_value_head=False):
    W1, b1, W2, b2, Wp, bp, Wv, bv = params
    x, h1, z2, h2 = cache["x"], cache["h1"], cache["z2"], cache["h2"]

    if from_value_head:
        dh2 = dlogits
        dWp = np.zeros_like(Wp)
        dbp = np.zeros_like(bp)
    else:
        dWp = np.outer(h2, dlogits)
        dbp = dlogits
        dh2 = dlogits @ Wp.T

    dz2 = dh2 * (1 - np.tanh(z2) ** 2)
    dW2 = np.outer(h1, dz2)
    db2 = dz2
    dh1 = dz2 @ W2.T
    dz1 = dh1 * (1 - np.tanh(cache["z1"]) ** 2)
    dW1 = np.outer(x, dz1)
    db1 = dz1
    return dW1, db1, dW2, db2, dWp, dbp


def _apply(params, grads, lr: float, clip: float = 1.0):
    W1, b1, W2, b2, Wp, bp, Wv, bv = params
    gW1, gb1, gW2, gb2, gWp, gbp, gWv, gbv = grads

    def gnorm(gs):
        tot = 0.0
        for g in gs:
            tot += float(np.sum(g * g))
        return np.sqrt(tot)

    gn = gnorm([gW1, gb1, gW2, gb2, gWp, gbp, gWv, gbv])
    scale = 1.0
    if clip and gn > clip:
        scale = clip / (gn + 1e-8)
    gW1 *= scale
    gb1 *= scale
    gW2 *= scale
    gb2 *= scale
    gWp *= scale
    gbp *= scale
    gWv *= scale
    gbv *= scale

    W1 -= lr * gW1
    b1 -= lr * gb1
    W2 -= lr * gW2
    b2 -= lr * gb2
    Wp -= lr * gWp
    bp -= lr * gbp
    Wv -= lr * gWv
    bv -= lr * gbv
    return [W1, b1, W2, b2, Wp, bp, Wv, bv]


def ppo_train(
    cfg: Optional[Candle7RlConfig] = None, model_path: Optional[str] = None, save: bool = True
) -> Dict[str, Any]:
    cfg = cfg or Candle7RlConfig()
    env = Candle7Env(
        symbol=cfg.ticker,
        interval=cfg.interval,
        days=cfg.days,
        lot_size=cfg.lot_size,
        fee_rate=cfg.fee_rate,
        slippage_bps=cfg.slippage_bps,
        action_cost_open=cfg.action_cost_open,
        action_cost_close=cfg.action_cost_close,
        invalid_action_penalty=cfg.invalid_action_penalty,
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
        gate_on_heuristic=cfg.gate_on_heuristic,
        # bc_weight é usado diretamente no loop de treino, não no env
    )

    obs = env.reset(seed=cfg.seed)
    obs_dim = env.observation_size
    act_dim = env.action_size

    hidden = cfg.hidden_size
    normalizer = RunningNorm(obs_dim)
    if model_path:
        try:
            data = np.load(model_path, allow_pickle=True)
            W1 = data["W1"].astype(np.float32)
            b1 = data["b1"].astype(np.float32)
            W2 = data.get("W2", np.zeros((hidden, hidden), dtype=np.float32)).astype(np.float32)
            b2 = data.get("b2", np.zeros((hidden,), dtype=np.float32)).astype(np.float32)
            Wp = data.get("Wp", data.get("W2", np.zeros((hidden, act_dim), dtype=np.float32))).astype(np.float32)
            bp = data.get("bp", data.get("b2", np.zeros((act_dim,), dtype=np.float32))).astype(np.float32)
            Wv = data.get("Wv", np.zeros((hidden, 1), dtype=np.float32)).astype(np.float32)
            bv = data.get("bv", np.zeros((1,), dtype=np.float32)).astype(np.float32)
            params = [W1, b1, W2, b2, Wp, bp, Wv, bv]
            if params[0].shape[0] != obs_dim or params[4].shape[1] != act_dim:
                print("Model shape mismatch, re-initializing.")
                params = _mlp_init(obs_dim, hidden, act_dim)
            if "norm_mean" in data and "norm_M2" in data and "norm_count" in data:
                normalizer.mean = data["norm_mean"].astype(np.float32)
                normalizer.M2 = data["norm_M2"].astype(np.float32)
                normalizer.count = float(data["norm_count"][0])
        except Exception as e:
            print(f"Could not load model, starting from scratch. Error: {e}")
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
    base_lr = cfg.learning_rate
    bc_weight_start = cfg.bc_weight
    vf_coef = 0.5
    ent_coef_start = 0.02
    ent_coef_end = 0.005
    grad_clip = cfg.grad_clip
    epochs = 4
    minibatch_size = 256

    history = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = os.path.join("reports", "agents", "candle_pattern7_rl")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"metrics_ppo_{cfg.ticker}_{cfg.interval}_{ts}.jsonl")

    for ep in range(total_episodes):
        obs_buf = np.zeros((rollout_len, obs_dim), dtype=np.float32)
        act_buf = np.zeros((rollout_len,), dtype=np.int32)
        logp_buf = np.zeros((rollout_len,), dtype=np.float32)
        val_buf = np.zeros((rollout_len,), dtype=np.float32)
        heur_act_buf = np.zeros((rollout_len,), dtype=np.int32)
        rew_buf = np.zeros((rollout_len,), dtype=np.float32)
        done_buf = np.zeros((rollout_len,), dtype=np.float32)

        trades = 0
        steps = 0
        for t in range(rollout_len):
            probs, v, cache = _forward(params, obs)
            a = int(np.random.choice(len(probs), p=probs))
            logp = float(np.log(probs[a] + 1e-8))

            res = env.step(a)
            if "trade" in res.info or "trade_forced" in res.info:
                trades += 1

            obs_buf[t] = obs
            act_buf[t] = a
            logp_buf[t] = logp
            val_buf[t] = v
            heur_act_buf[t] = res.info.get("heuristic_action", -1)
            rew_buf[t] = res.reward
            done_buf[t] = 1.0 if res.done else 0.0

            steps += 1
            obs = res.obs
            normalizer.update(obs)
            obs = normalizer.normalize(obs)
            if res.done:
                obs = env.reset()
                normalizer.update(obs)
                obs = normalizer.normalize(obs)

        next_v = _forward(params, obs)[1]

        adv = np.zeros_like(rew_buf)
        lastgaelam = 0.0
        for t in reversed(range(rollout_len)):
            nonterminal = 1.0 - done_buf[t]
            nextval = val_buf[t + 1] if t + 1 < rollout_len else next_v
            delta = rew_buf[t] + gamma * nextval * nonterminal - val_buf[t]
            lastgaelam = delta + gamma * gae_lambda * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + val_buf
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Scheduling
        current_lr = base_lr * max(0.1, 1.0 - ep / total_episodes)
        current_ent_coef = ent_coef_start * (1 - ep / total_episodes) + ent_coef_end * (ep / total_episodes)

        idx = np.arange(rollout_len)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, rollout_len, minibatch_size):
                mb = idx[start : start + minibatch_size]
                if len(mb) == 0:
                    continue

                gW1, gb1, gW2, gb2, gWp, gbp, gWv, gbv = [np.zeros_like(p) for p in params]

                # Decaimento do peso do BC
                frac_left = 1.0 - (ep / max(total_episodes - 1, 1))
                current_bc_weight = bc_weight_start * frac_left

                for i in mb:
                    ob, a, old_logp, v_targ, adv_i = (
                        obs_buf[i],
                        int(act_buf[i]),
                        logp_buf[i],
                        float(ret[i]),
                        float(adv[i]),
                    )

                    probs, v_pred, cache = _forward(params, ob)
                    logp_new = float(np.log(probs[a] + 1e-8))
                    ratio = np.exp(logp_new - old_logp)
                    clipped_ratio = np.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)

                    use_unclipped = (adv_i >= 0.0 and ratio <= clipped_ratio) or (
                        adv_i < 0.0 and ratio >= clipped_ratio  # This logic is for deciding *if* we update
                    )

                    coeff = (
                        -adv_i if use_unclipped else 0.0
                    )  # The gradient is scaled by advantage, not advantage * ratio
                    dW1p, db1p, dW2p, db2p, dWp, dbp = _backward_policy(params, cache, a, coeff)
                    gW1 += dW1p
                    gb1 += db1p
                    gW2 += dW2p
                    gb2 += db2p
                    gWp += dWp
                    gbp += dbp

                    # Perda auxiliar de Behavioral Cloning
                    if current_bc_weight > 0:
                        heur_a = heur_act_buf[i]
                        if heur_a != -1:
                            # Usa o mesmo backward, mas com 'advantage' fixo (o peso do BC)
                            dW1_bc, db1_bc, dW2_bc, db2_bc, dWp_bc, dbp_bc = _backward_policy(
                                params, cache, int(heur_a), -current_bc_weight
                            )
                            gW1 += dW1_bc
                            gb1 += db1_bc
                            gW2 += dW2_bc
                            gb2 += db2_bc
                            gWp += dWp_bc
                            gbp += dbp_bc

                    dW1v, db1v, dW2v, db2v, dWv, dbv = _backward_value(params, cache, v_targ)
                    gW1 += vf_coef * dW1v
                    gb1 += vf_coef * db1v
                    gW2 += vf_coef * dW2v
                    gb2 += vf_coef * db2v
                    gWv += vf_coef * dWv
                    gbv += vf_coef * dbv

                    # Entropy bonus for exploration
                    dW1e, db1e, dW2e, db2e, dWpe, dbpe = _backward_entropy(params, cache, current_ent_coef)
                    gW1 += dW1e
                    gb1 += db1e
                    gW2 += dW2e
                    gb2 += db2e
                    gWp += dWpe
                    gbp += dbpe

                params = _apply(params, (gW1, gb1, gW2, gb2, gWp, gbp, gWv, gbv), current_lr, clip=grad_clip)

        ep_reward = float(np.sum(rew_buf))
        history.append({"episode": ep + 1, "steps": steps, "reward": ep_reward, "trades": trades})
        if (ep + 1) % 5 == 0:
            print(f"PPO Episode {ep+1}/{total_episodes} | steps={steps} reward={ep_reward:.2f} trades={trades}")
        try:
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(history[-1]) + "\n")
        except Exception:
            pass

    def greedy_policy(o):
        o_norm = normalizer.normalize(o)
        return _forward(params, o_norm)[0]

    eval_res = env.run_episode(greedy_policy, max_steps=cfg.max_steps)
    print(
        f"PPO Greedy evaluation: reward={eval_res['reward']:.2f} steps={eval_res['steps']} trades={eval_res['trades']}"
    )

    if save:
        out_dir = os.path.join("reports", "agents", "candle_pattern7_rl")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"ppo_{cfg.ticker}_{cfg.interval}.npz")
        np.savez(
            out_path,
            W1=params[0],
            b1=params[1],
            W2=params[2],
            b2=params[3],
            Wp=params[4],
            bp=params[5],
            Wv=params[6],
            bv=params[7],
            norm_mean=normalizer.mean,
            norm_M2=normalizer.M2,
            norm_count=np.array([normalizer.count]),
            config=np.array([str(asdict(cfg))], dtype=object),
        )
        print(f"Saved PPO agent to {out_path}")
    return {"history": history, "eval": eval_res, "model_path": out_path if save else ""}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train PPO agent (NumPy) on Candle7Env")
    p.add_argument("--ticker", default="BTCUSDT")
    p.add_argument("--interval", default="15m")
    p.add_argument("--days", type=int, default=3650)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--episode_len", type=int, default=4096)
    p.add_argument("--long_only", action="store_true")
    # costs/dynamics
    p.add_argument("--fee_rate", type=float, default=0.0005)
    p.add_argument("--action_cost_open", type=float, default=0.01)
    p.add_argument("--action_cost_close", type=float, default=0.01)
    p.add_argument("--invalid_action_penalty", type=float, default=0.05)
    p.add_argument("--m2m_weight", type=float, default=0.15)
    p.add_argument("--min_hold_bars", type=int, default=5)
    p.add_argument("--reopen_cooldown_bars", type=int, default=3)
    p.add_argument("--exec_next_open", action="store_true", default=True)
    p.add_argument("--switch_penalty", type=float, default=0.01)
    p.add_argument("--switch_window_bars", type=int, default=5)
    p.add_argument("--reward_atr_norm", action="store_true")
    p.add_argument("--atr_period", type=int, default=14)
    p.add_argument("--gate_on_heuristic", action="store_true")
    p.add_argument("--bc_weight", type=float, default=0.2, help="Peso da perda auxiliar de imitação (decai)")
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()

    cfg = Candle7RlConfig(
        ticker=args.ticker,
        interval=args.interval,
        days=args.days,
        long_only=bool(args.long_only),
        episodes=args.episodes,
        hidden_size=args.hidden,
        learning_rate=args.lr,
        episode_len=args.episode_len,
        fee_rate=args.fee_rate,
        action_cost_open=args.action_cost_open,
        action_cost_close=args.action_cost_close,
        invalid_action_penalty=args.invalid_action_penalty,
        m2m_weight=args.m2m_weight,
        min_hold_bars=args.min_hold_bars,
        reopen_cooldown_bars=args.reopen_cooldown_bars,
        exec_next_open=args.exec_next_open,
        switch_penalty=args.switch_penalty,
        switch_window_bars=args.switch_window_bars,
        reward_atr_norm=args.reward_atr_norm,
        atr_period=args.atr_period,
        gate_on_heuristic=args.gate_on_heuristic,
        bc_weight=args.bc_weight,
    )
    ppo_train(cfg, model_path=args.model)
