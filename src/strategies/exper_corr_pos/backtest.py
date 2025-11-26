from __future__ import annotations

"""
Greedy backtest (one‑shot) CLI for exper_corr_pos.

Usage (clean, config‑driven):

  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.backtest

Reads src/strategies/exper_corr_pos/config.json, loads dataset from local cache,
loads the preferred checkpoint (best|final|latest) if available, runs a greedy
evaluation over a deterministic window, renders a compact chart, and writes a
JSON with minimal results for quick comparisons.
"""

import json
import sys
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from .data import load_primary_series, load_confirm_series, prepare_dataset
from .env import BTCMixtureEnv, EnvConfig
from .models import MoEPolicy
from .utils_cfg import build_policy, bars_for_days, hours_per_bar, merged_env_config, num_actions_from_env
from ...utils.metrics import calculate_metrics


CFG_PATH = Path("src/strategies/exper_corr_pos/config.json")
OUTDIR = Path("src/strategies/exper_corr_pos/reports/backtest")


def _collect_run_env() -> Dict[str, Any]:
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    # Poetry version (best effort)
    try:
        import subprocess

        out = subprocess.check_output(["poetry", "--version"], stderr=subprocess.DEVNULL).decode().strip()
        m = re.search(r"(\d+\.\d+\.\d+)", out)
        poetry_ver = m.group(1) if m else out
    except Exception:
        poetry_ver = None

    libs = ["pandas", "numpy", "matplotlib", "torch", "optuna"]
    versions: Dict[str, Optional[str]] = {}
    for name in libs:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", None)
        except Exception:
            versions[name] = None
    return {"python": py_ver, "poetry": poetry_ver, "lib_versions": versions}


def _load_policy_for_eval(input_dim: int, cfg: Dict[str, Any]) -> Tuple[MoEPolicy, Optional[Path]]:
    policy = build_policy(input_dim, cfg)
    # Prefer checkpoint using visualize's selection
    ckpt_path: Optional[Path] = None
    try:
        from .visualize import _find_checkpoint

        prefer = str(cfg.get("visualize", {}).get("prefer", "best")).lower()
        ckpt = _find_checkpoint(prefer)
        state = torch.load(ckpt, map_location="cpu")
        try:
            policy.load_state_dict(state)
        except Exception:
            # partial/shape‑tolerant load
            model_state = policy.state_dict()
            filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
            policy.load_state_dict(filtered, strict=False)
        ckpt_path = ckpt
    except FileNotFoundError:
        ckpt_path = None
    except Exception:
        ckpt_path = None
    policy.eval()
    return policy, ckpt_path


@dataclass
class EvalResult:
    equity_end: float
    pnl: float
    trades: int
    trade_list: List[Dict[str, Any]]
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    ruined: bool


def _eval_env_greedy(policy: MoEPolicy, env: BTCMixtureEnv, *, device: torch.device, bar_hours: float) -> EvalResult:
    obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    rewards: List[float] = []
    trades: List[Dict[str, Any]] = []
    done = False
    ruined = False
    while not done:
        with torch.no_grad():
            dist, _, _ = policy(obs.unsqueeze(0))
            action = torch.argmax(dist.probs, dim=-1).item()
        next_obs, reward, done, info = env.step(action)
        rewards.append(float(reward))
        if info.get("trade_closed"):
            trades.append({
                "pnl": float(info.get("trade_pnl", 0.0)),
                "duration_bars": int(info.get("trade_bars", 0)),
            })
        if info.get("ruined"):
            ruined = True
        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    metrics = calculate_metrics(trades)
    return EvalResult(
        equity_end=float(info.get("equity", 0.0)),
        pnl=float(np.sum(rewards)),
        trades=int(len(trades)),
        trade_list=trades,
        win_rate=float(metrics.get("win_rate", 0.0)),
        profit_factor=float(metrics.get("profit_factor", 0.0)),
        avg_win=float(metrics.get("avg_win", 0.0)),
        avg_loss=float(metrics.get("avg_loss", 0.0)),
        ruined=bool(ruined),
    )


def _render_chart(price_df, actions: np.ndarray, equity: np.ndarray, names: List[str], *, out: Path) -> None:
    closes = price_df["close"].to_numpy()[: len(actions)]
    idx = np.arange(len(actions))
    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(idx, closes, color="#4da6ff", linewidth=1.0, label="Preço")
    longs = actions == 1
    shorts = actions == -1
    ax.scatter(idx[longs], closes[longs], color="#6cff6c", marker="^", s=36, label="Long")
    ax.scatter(idx[shorts], closes[shorts], color="#ff6666", marker="v", s=36, label="Short")
    ax2 = ax.twinx()
    ax2.plot(idx, equity, color="#f0c674", linewidth=1.0, label="Equity")
    ax.set_xlabel("Índice do candle")
    ax.set_ylabel("Preço")
    ax2.set_ylabel("Equity")
    ax.legend(loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def run_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    symbol = str(data_cfg.get("base_symbol", "BTCUSDT"))
    timeframe = str(data_cfg.get("timeframe", "1d"))

    primary_df = load_primary_series(config)
    confirm_df = load_confirm_series(config)
    dataset = prepare_dataset(primary_df, config=config, confirm_df=confirm_df)
    if dataset.empty:
        raise RuntimeError("Dataset vazio para backtest. Atualize o cache local.")

    price_cols = ["open", "high", "low", "close", "volume"]
    timestamps = dataset.index.to_list()
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    # Deterministic evaluation window
    eval_days = int(config.get("train", {}).get("eval_days", 90) or 0)
    eval_bars = bars_for_days(timeframe, eval_days) if eval_days > 0 else 0
    if eval_bars > 0:
        price_df = price_df.tail(eval_bars).reset_index(drop=True)
        feat_df = feat_df.tail(eval_bars).reset_index(drop=True)
        timestamps = timestamps[-len(price_df) :]

    env_cfg_dict = merged_env_config(config)
    env_cfg = EnvConfig(**env_cfg_dict)
    env_cfg.random_start = False  # manter avaliação determinística
    env_cfg.window_bars = 0
    env = BTCMixtureEnv(price_df, feat_df, env_cfg, timestamps=timestamps)

    # Garante que o head da policy casa com o espaço de ações do env (suporta long-only)
    num_actions = env.action_space.n if hasattr(env, "action_space") else None
    policy, ckpt_path = _load_policy_for_eval(feat_df.shape[1], config)
    if num_actions is not None and getattr(policy, "num_actions", num_actions) != num_actions:
        policy = build_policy(feat_df.shape[1], config, num_actions_override=num_actions)

    # Run greedy and also capture actions/equity for chart
    device = torch.device(config.get("train", {}).get("device", "cpu"))
    bar_hours = hours_per_bar(timeframe)
    obs = torch.tensor(env.reset(), dtype=torch.float32)
    actions: List[int] = []
    equity: List[float] = []
    done = False
    allow_short = bool(getattr(env_cfg, "allow_short", True))
    while not done:
        with torch.no_grad():
            dist, _, _ = policy(obs.unsqueeze(0))
            action = torch.argmax(dist.probs, dim=-1).item()
        next_obs, _, done, info = env.step(action)
        pos = (action - 1) if allow_short else action  # map para {-1,0,1} ou {0,1}
        actions.append(pos)
        equity.append(float(info.get("equity", 0.0)))
        obs = torch.tensor(next_obs, dtype=torch.float32)

    # Evaluate metrics using the dedicated function
    # Recreate eval env to avoid double counting any internal state from the previous loop
    eval_env = BTCMixtureEnv(price_df, feat_df, env_cfg, timestamps=timestamps)
    res = _eval_env_greedy(policy, eval_env, device=device, bar_hours=bar_hours)

    # Chart path
    charts_dir = OUTDIR / "charts"
    chart_path = charts_dir / f"exper_corr_pos_backtest_{symbol}_{timeframe}.png"
    _render_chart(price_df, np.asarray(actions), np.asarray(equity), [], out=chart_path)

    # Period boundaries
    period_start = str(timestamps[0]) if timestamps else ""
    period_end = str(timestamps[-1]) if timestamps else ""

    payload = {
        "strategy": "exper_corr_pos",
        "symbol": symbol,
        "interval": timeframe,
        "period": {"start": period_start, "end": period_end},
        "trades": res.trades,
        "total_pnl": round(res.pnl, 6),
        "win_rate": res.win_rate,
        "profit_factor": res.profit_factor,
        "avg_win": res.avg_win,
        "avg_loss": res.avg_loss,
        "chart_path": str(chart_path),
        "config_path": str(CFG_PATH),
        "seed": int(config.get("train", {}).get("seed", 42)),
        "run_env": _collect_run_env(),
        "model_path": str(ckpt_path) if ckpt_path else None,
    }
    return payload


def main() -> None:
    cfg = json.loads(CFG_PATH.read_text())
    OUTDIR.mkdir(parents=True, exist_ok=True)
    result = run_backtest(cfg)
    out_json = OUTDIR / f"exper_corr_pos_{result['symbol']}_{result['interval']}.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[backtest] Resultado salvo em {out_json}")


if __name__ == "__main__":
    main()
