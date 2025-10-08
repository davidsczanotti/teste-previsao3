from __future__ import annotations

import os
from dataclasses import asdict
from typing import Tuple, Dict, Any, Optional
import json
from datetime import datetime
import numpy as np

from .config import Candle7RlConfig
from .env import Candle7Env


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
    # ensure scalar extraction without deprecated ndarray-to-scalar casts
    v = float(h1 @ Wv[:, 0] + bv[0])
    cache = {"x": x, "z1": z1, "h1": h1, "probs": probs, "v": v}
    return probs, v, cache


def _mlp_backward(params, cache, action: int, advantage: float, value_error: float):
    W1, b1, W2, b2, Wv, bv = params
    x = cache["x"]
    h1 = cache["h1"]
    probs = cache["probs"].copy()

    dlogits = probs
    dlogits[action] -= 1.0
    dlogits *= -advantage

    dW2 = np.outer(h1, dlogits)
    db2 = dlogits

    dh1 = dlogits @ W2.T
    # value head grads
    dv = value_error
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

    def _gnorm(gs):
        tot = 0.0
        for g in gs:
            tot += float(np.sum(g * g))
        return np.sqrt(tot)

    gnorm = _gnorm([gW1, gb1, gW2, gb2, gWv, gbv])
    scale = 1.0
    if clip and gnorm > clip:
        scale = clip / (gnorm + 1e-8)
    gW1 *= scale
    gb1 *= scale
    gW2 *= scale
    gb2 *= scale
    gWv *= scale
    gbv *= scale

    W1 -= lr * gW1
    b1 -= lr * gb1
    W2 -= lr * gW2
    b2 -= lr * gb2
    Wv -= lr * gWv
    bv -= lr * gbv
    return W1, b1, W2, b2, Wv, bv


def train(config: Optional[Candle7RlConfig] = None, model_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = config or Candle7RlConfig()
    env = Candle7Env(
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
        gate_on_heuristic=cfg.gate_on_heuristic,
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
            if W2.shape[1] != out_size:
                W1, b1, W2, b2, Wv, bv = _mlp_init(in_size, hidden, out_size)
            # Se o tamanho da entrada mudou, reinicialize tudo
            if W1.shape[0] != in_size:
                print("WARN: Input size mismatch. Re-initializing model from scratch.")
                W1, b1, W2, b2, Wv, bv = _mlp_init(in_size, hidden, out_size)
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

    history = []
    # metrics logging
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = os.path.join("reports", "agents", "candle_pattern7_rl")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"metrics_{cfg.ticker}_{cfg.interval}_{ts}.jsonl")
    for ep in range(cfg.episodes):
        obs = env.reset()
        episode_obs = []
        episode_act = []
        episode_rew = []
        episode_lbl = []
        episode_cache = []
        # counters
        trades = 0
        invalids = 0
        idles = 0
        action_hist = np.zeros(env.action_size, dtype=int)
        hold_durations: list[int] = []

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
            episode_lbl.append(int(res.info.get("heuristic_action", -1)))
            action_hist[action] += 1
            if "trade" in res.info or "trade_forced" in res.info:
                trades += 1
            if "invalid" in res.info:
                invalids += 1
            if res.info.get("idle", False):
                idles += 1
            if "hold_bars" in res.info:
                try:
                    hold_durations.append(int(res.info["hold_bars"]))
                except Exception:
                    pass

            steps += 1
            if res.done or (cfg.max_steps is not None and steps >= cfg.max_steps):
                break
            obs = res.obs

        # Returns and advantages
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

        # Accumulate grads over episode
        W1, b1, W2, b2, Wv, bv = params
        gW1 = np.zeros_like(W1)
        gb1 = np.zeros_like(b1)
        gW2 = np.zeros_like(W2)
        gb2 = np.zeros_like(b2)
        gWv = np.zeros_like(Wv)
        gbv = np.zeros_like(bv)
        entropy = 0.0
        for idx, (obs_t, act_t, adv_t, ret_t, cache_t) in enumerate(
            zip(episode_obs, episode_act, advantages, returns, episode_cache)
        ):
            value_err = -(ret_t - cache_t["v"])  # d/dv of 0.5*(ret-v)^2
            dW1, db1, dW2, db2, dWv, dbv = _mlp_backward(params, cache_t, int(act_t), float(adv_t), float(value_err))
            gW1 += dW1
            gb1 += db1
            gW2 += dW2
            gb2 += db2
            gWv += dWv
            gbv += dbv
            p = cache_t["probs"]
            entropy -= np.sum(p * np.log(p + 1e-8))

            # Behavior cloning auxiliar com rótulo heurístico
            if cfg.bc_weight and cfg.bc_weight > 0.0:
                frac_left = 1.0 - (ep / max(cfg.episodes - 1, 1))
                bc_scale = float(cfg.bc_weight) * frac_left
                lbl = episode_lbl[idx]
                if lbl >= 0:
                    dW1, db1, dW2, db2, dWv, dbv = _mlp_backward(params, cache_t, int(lbl), float(bc_scale), 0.0)
                    gW1 += dW1
                    gb1 += db1
                    gW2 += dW2
                    gb2 += db2

        # Optional entropy shaping (disabled by default)
        if entropy_beta_start > 0:
            K = out_size
            uniform = 1.0 / K
            mean_p = np.mean([c["probs"] for c in episode_cache], axis=0)
            frac = 1.0 - (ep / max(cfg.episodes - 1, 1))
            entropy_beta = entropy_beta_end + (entropy_beta_start - entropy_beta_end) * frac
            gb2 += -entropy_beta * (uniform - mean_p)

        params = _apply_grads(params, (gW1, gb1, gW2, gb2, gWv, gbv), lr, clip=grad_clip)

        ep_reward = float(np.sum(episode_rew))
        history.append({"episode": ep + 1, "steps": steps, "reward": ep_reward, "entropy": float(entropy)})
        if (ep + 1) % 5 == 0:
            print(f"Episode {ep+1}/{cfg.episodes} | steps={steps} reward={ep_reward:.2f} ent={entropy:.3f}")
        # write metrics line
        rec = {
            "episode": ep + 1,
            "steps": steps,
            "reward": ep_reward,
            "trades": trades,
            "invalids": invalids,
            "idles": idles,
            "action_hist": action_hist.tolist(),
            "mean_hold_bars": (float(np.mean(hold_durations)) if hold_durations else 0.0),
        }
        try:
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    # Evaluate greedy
    eval_res = env.run_episode(lambda o: _mlp_forward(params, o)[0], max_steps=cfg.max_steps)
    print(f"Greedy evaluation: reward={eval_res['reward']:.2f} steps={eval_res['steps']} trades={eval_res['trades']}")

    out_dir = os.path.join("reports", "agents", "candle_pattern7_rl")
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
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agent on Candle7Env")
    parser.add_argument("--ticker", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--days", type=int, default=3650)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--long_only", action="store_true")
    parser.add_argument("--episode_len", type=int, default=4096)
    # exploration & shaping
    parser.add_argument("--epsilon_start", type=float, default=0.3)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--entropy_beta", type=float, default=0.0)
    parser.add_argument(
        "--bc_weight", type=float, default=0.1, help="Peso da perda auxiliar de imitação (decai ao longo dos episódios)"
    )
    # dynamics/costs
    parser.add_argument("--min_hold_bars", type=int, default=0)
    parser.add_argument("--reopen_cooldown_bars", type=int, default=0)
    parser.add_argument("--action_cost_open", type=float, default=0.0)
    parser.add_argument("--action_cost_close", type=float, default=0.0)
    parser.add_argument("--m2m_weight", type=float, default=0.05)
    # idle penalty
    parser.add_argument(
        "--idle_penalty", type=float, default=0.0, help="Penalidade (USD) por manter Hold flat após carência"
    )
    parser.add_argument("--idle_grace", type=int, default=0, help="Barras de carência sem penalidade")
    parser.add_argument("--idle_ramp", type=float, default=0.0, help="Rampa linear adicional por barra após a carência")
    # execution+anti-churn
    parser.add_argument(
        "--exec_next_open", action="store_true", help="Executa ordens na Abertura da próxima barra (padrão)"
    )
    parser.add_argument("--no_exec_next_open", dest="exec_next_open", action="store_false")
    parser.set_defaults(exec_next_open=True)
    parser.add_argument(
        "--switch_penalty", type=float, default=0.0, help="Penalidade por reabrir/virar lado em janela curta"
    )
    parser.add_argument("--switch_window_bars", type=int, default=5, help="Janela (barras) para switch_penalty")
    # reward normalization by ATR
    parser.add_argument("--reward_atr_norm", action="store_true", help="Normaliza recompensa por ATR da barra")
    parser.add_argument("--atr_period", type=int, default=14)
    parser.add_argument(
        "--gate_on_heuristic",
        action="store_true",
        help="Permite abrir posição apenas quando a heurística 7-candles concordar",
    )
    # resume/warm-start
    parser.add_argument("--model", type=str, default=None, help="Caminho para .npz salvo para continuar o treinamento")
    args = parser.parse_args()

    cfg = Candle7RlConfig(
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
        entropy_beta=args.entropy_beta,
        bc_weight=args.bc_weight,
        idle_penalty=args.idle_penalty,
        idle_grace_bars=args.idle_grace,
        idle_ramp=args.idle_ramp,
        min_hold_bars=args.min_hold_bars,
        reopen_cooldown_bars=args.reopen_cooldown_bars,
        action_cost_open=args.action_cost_open,
        action_cost_close=args.action_cost_close,
        m2m_weight=args.m2m_weight,
        exec_next_open=args.exec_next_open,
        switch_penalty=args.switch_penalty,
        switch_window_bars=args.switch_window_bars,
        reward_atr_norm=args.reward_atr_norm,
        atr_period=args.atr_period,
        gate_on_heuristic=args.gate_on_heuristic,
    )
    # pass idle shaping to env via train()
    train(cfg, model_path=args.model)
