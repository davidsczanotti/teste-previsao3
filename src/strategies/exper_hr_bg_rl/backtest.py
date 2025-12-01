from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from .data import build_dataset
from .env import EnvConfig, RangeVolEnv
from .models import PolicyValueNet


CFG_PATH = Path("src/strategies/exper_hr_bg_rl/config.json")


def run_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    env_cfg_dict = config.get("env", {})
    back_cfg = config.get("backtest", {}) or {}

    price_df, feat_df, timestamps = build_dataset(config)

    env_cfg = EnvConfig(**env_cfg_dict)
    # avaliação determinística: sem random_start, janela final (eval_days)
    env_cfg.random_start = False
    eval_days = int(back_cfg.get("eval_days", 180))
    timeframe = str(data_cfg.get("timeframe", "1h"))
    # 24 candles por dia aproximado para 1h
    eval_bars = max(2, eval_days * 24)
    if eval_bars < len(price_df):
        price_df = price_df.tail(eval_bars).reset_index(drop=True)
        feat_df = feat_df.tail(eval_bars).reset_index(drop=True)
        timestamps = timestamps[-eval_bars:]
    env_cfg.window_bars = len(price_df)

    env = RangeVolEnv(price_df, feat_df, env_cfg, timestamps=timestamps)

    input_dim = feat_df.shape[1] + 1
    num_actions = env.n_actions
    hidden_sizes = config.get("model", {}).get("hidden_sizes", [128, 64])

    device = torch.device("cpu")
    policy = PolicyValueNet(input_dim=input_dim, hidden_sizes=hidden_sizes, num_actions=num_actions).to(device)

    # carregar melhor modelo, se existir; senão, final
    train_cfg = config.get("train", {})
    outdir = Path(train_cfg.get("outdir", "src/strategies/exper_hr_bg_rl/reports/train"))
    best_path = outdir / "policy_best.pt"
    final_path = outdir / "policy_final.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
    elif final_path.exists():
        state = torch.load(final_path, map_location=device)
    else:
        raise FileNotFoundError("Nenhum modelo treinado encontrado (policy_best.pt / policy_final.pt).")
    policy.load_state_dict(state)
    policy.eval()

    obs_np = env.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    equity = env_cfg.init_equity
    done = False
    equity_curve = []

    while not done:
        with torch.no_grad():
            dist, _ = policy(obs.unsqueeze(0))
            action = torch.argmax(dist.probs, dim=-1)
        next_obs_np, reward, done, info = env.step(int(action.item()))
        equity = float(info.get("equity", equity))
        equity_curve.append(equity)
        obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)

    if not equity_curve:
        raise RuntimeError("Backtest não gerou passos.")

    equity_arr = np.asarray(equity_curve, dtype=np.float64)
    total_pnl = float(equity_arr[-1] - env_cfg.init_equity)
    ret = np.diff(equity_arr) / equity_arr[:-1]
    avg_ret = float(ret.mean()) if ret.size > 0 else 0.0
    std_ret = float(ret.std()) if ret.size > 0 else 0.0

    result = {
        "strategy": "exper_hr_bg_rl",
        "symbol": data_cfg.get("base_symbol", "BTCUSDT"),
        "interval": timeframe,
        "eval_days": eval_days,
        "steps": int(len(equity_curve)),
        "final_equity": float(equity_arr[-1]),
        "total_pnl": total_pnl,
        "avg_return_per_step": avg_ret,
        "std_return_per_step": std_ret,
    }
    return result


def main() -> None:
    cfg = json.loads(CFG_PATH.read_text())
    outdir = Path(cfg.get("backtest", {}).get("outdir", "src/strategies/exper_hr_bg_rl/reports/backtest"))
    outdir.mkdir(parents=True, exist_ok=True)
    res = run_backtest(cfg)
    out_json = outdir / f"exper_hr_bg_rl_{res['symbol']}_{res['interval']}.json"
    out_json.write_text(json.dumps(res, indent=2))
    print(f"[exper_hr_bg_rl.backtest] Resultado salvo em {out_json}")


if __name__ == "__main__":
    main()

