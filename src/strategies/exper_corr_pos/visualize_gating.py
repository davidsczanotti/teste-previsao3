from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import load_primary_series, load_confirm_series, prepare_dataset
from .env import BTCMixtureEnv, EnvConfig
from .models import MoEPolicy
from .utils_cfg import build_policy
from .utils_cfg import enabled_expert_names

CFG_PATH = Path("src/strategies/exper_corr_pos/config.json")
FINAL_MODEL_PATH = Path("src/strategies/exper_corr_pos/reports/train/moe_policy_final.pt")
BEST_MODEL_PATH = Path("src/strategies/exper_corr_pos/reports/train/moe_policy_best_eval.pt")
TRAIN_DIR = Path("src/strategies/exper_corr_pos/reports/train")
OUTDIR = Path("src/strategies/exper_corr_pos/reports/train")


def _expert_names(cfg: dict, num_experts: int) -> list[str]:
    names = enabled_expert_names(cfg)
    if len(names) == num_experts:
        return names
    defaults = ["TrendML", "MultiFrame", "Spread", "Pattern"]
    return (names or defaults)[:num_experts]


def _find_checkpoint() -> Path:
    if BEST_MODEL_PATH.exists():
        return BEST_MODEL_PATH
    if FINAL_MODEL_PATH.exists():
        return FINAL_MODEL_PATH
    if TRAIN_DIR.exists():
        eps = sorted(TRAIN_DIR.glob("moe_policy_ep*.pt"))
        if eps:
            return eps[-1]
    raise FileNotFoundError(
        f"Nenhum modelo encontrado. Esperado {BEST_MODEL_PATH} ou {FINAL_MODEL_PATH} ou 'moe_policy_ep*.pt'. Aguarde salvar um checkpoint (10,20,...) ou reduza 'episodes'."
    )


def _load_policy(input_dim: int, cfg: dict) -> MoEPolicy:
    policy = build_policy(input_dim, cfg)
    chosen_path = _find_checkpoint()
    # load parcial tolerante a mudanças de arquitetura
    state_dict = torch.load(chosen_path, map_location="cpu")
    model_state = policy.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    policy.load_state_dict(filtered, strict=False)
    policy.eval()
    print(f"[gating] Carregado (parcial={len(filtered)}/{len(model_state)}) de {chosen_path}")
    return policy


def main() -> None:
    cfg = json.loads(CFG_PATH.read_text())
    env_cfg = EnvConfig(**cfg.get("env", {}))
    train_cfg = cfg.get("train", {})
    eval_days = int(train_cfg.get("eval_days", 90))

    # dados
    primary_df = load_primary_series(cfg)
    confirm_df = load_confirm_series(cfg)
    lookback_bars = max(eval_days, 120) * 24
    primary_df = primary_df.tail(lookback_bars)
    dataset = prepare_dataset(primary_df, config=cfg, confirm_df=confirm_df)
    price_cols = ["open", "high", "low", "close", "volume"]
    timestamps = dataset.index.to_list()
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    policy = _load_policy(feat_df.shape[1], cfg)

    # janela de avaliação
    hours = eval_days * 24
    prices = price_df.tail(hours).reset_index(drop=True)
    feats = feat_df.tail(hours).reset_index(drop=True)
    tail_ts = timestamps[-len(prices) :] if len(prices) > 0 else []
    env = BTCMixtureEnv(prices, feats, env_cfg, timestamps=tail_ts)

    obs = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)

    weights_trace: List[np.ndarray] = []
    masks_trace: List[np.ndarray] = []
    actions: List[int] = []
    equities: List[float] = []
    timestamps_step: List[str] = []

    done = False
    drawdowns: List[float] = []
    ruined_step: int | None = None
    while not done:
        with torch.no_grad():
            dist, _, _ = policy(obs)
            action = torch.argmax(dist.probs, dim=-1).item()
            weights, mask = policy.gating(obs, top_k=policy.top_k)

        next_obs, _, done, info = env.step(action)
        weights_trace.append(weights.squeeze(0).cpu().numpy())
        masks_trace.append(mask.squeeze(0).cpu().numpy())
        actions.append(action)
        equities.append(float(info.get("equity", 0.0)))
        timestamps_step.append(str(info.get("timestamp", "")))
        dd = float(info.get("drawdown", 0.0))
        drawdowns.append(dd)
        if info.get("ruined") and ruined_step is None:
            ruined_step = len(weights_trace) - 1
        obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

    weights_arr = np.vstack(weights_trace)  # [T, E]
    masks_arr = np.vstack(masks_trace)  # [T, E]

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # CSV de traço
    trace_path = OUTDIR / "gating_trace.csv"
    header = ["timestamp"] + [f"w_e{i}" for i in range(weights_arr.shape[1])] + [f"m_e{i}" for i in range(masks_arr.shape[1])] + ["action", "equity"]
    import csv

    with trace_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in range(weights_arr.shape[0]):
            stamp = timestamps_step[t] if t < len(timestamps_step) else ""
            writer.writerow([stamp, *weights_arr[t].tolist(), *masks_arr[t].tolist(), actions[t], equities[t]])
    print(f"[gating] Traço salvo em {trace_path}")

    # Heatmap de pesos
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(weights_arr.T, aspect="auto", interpolation="nearest", cmap="viridis")
    # nomes
    names = _expert_names(cfg, weights_arr.shape[1])
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_ylabel("experts")
    ax.set_xlabel("t")
    ax.set_title("Gating weights (heatmap)")
    # marcadores de drawdown
    dd = np.asarray(drawdowns)
    thr = float(env_cfg.max_drawdown_pct)
    if thr < 1.0:
        dd_idx = np.where(dd >= thr)[0]
        for t in dd_idx:
            ax.axvline(t, color="#f0e68c", alpha=0.35, lw=0.6)  # amarelo claro
    # ruína (linha vermelha)
    if ruined_step is not None:
        ax.axvline(ruined_step, color="#ff6666", ls="--", lw=1.0, label="ruína")
        ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    heatmap_path = OUTDIR / "gating_heatmap.png"
    fig.savefig(heatmap_path, dpi=140)
    plt.close(fig)
    print(f"[gating] Heatmap salvo em {heatmap_path}")

    # Uso (contagem de top-k) médio
    usage = masks_arr.mean(axis=0)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(len(usage)), usage, color="#4da6ff")
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(usage)))
    ax.set_xticklabels(names)
    ax.set_title("Expert usage (média do top-k)")
    fig.tight_layout()
    usage_path = OUTDIR / "gating_usage.png"
    fig.savefig(usage_path, dpi=140)
    plt.close(fig)
    print(f"[gating] Uso salvo em {usage_path}")


if __name__ == "__main__":
    main()
