from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import load_btc_1h, prepare_dataset
from .env import BTCMixtureEnv, EnvConfig
from .models import MoEPolicy

CFG_PATH = Path("src/strategies/exper_corr_neg/config.json")
FINAL_MODEL_PATH = Path("src/strategies/exper_corr_neg/reports/train/moe_policy_final.pt")
BEST_MODEL_PATH = Path("src/strategies/exper_corr_neg/reports/train/moe_policy_best_eval.pt")
OUTDIR = Path("src/strategies/exper_corr_neg/reports/train")


def _expert_names(cfg: dict, num_experts: int) -> list[str]:
    names = cfg.get("model", {}).get("expert_names")
    if isinstance(names, list) and len(names) == num_experts:
        return [str(n) for n in names]
    # defaults (curto e didático)
    defaults = [
        "Trend",
        "MeanRev",
        "Volatility",
        "VolumeFlow",
        "SqueezeBreakout",
        "Pattern",
    ]
    if num_experts <= len(defaults):
        return defaults[:num_experts]
    return [f"e{i}" for i in range(num_experts)]


def _load_policy(input_dim: int, cfg: dict) -> MoEPolicy:
    model_cfg = cfg.get("model", {})
    policy = MoEPolicy(
        input_dim=input_dim,
        num_actions=3,
        expert_hidden=model_cfg.get("expert_hidden", [64, 32]),
        gating_hidden=model_cfg.get("gating_hidden", [64, 32]),
        num_experts=model_cfg.get("num_experts", 5),
        temperature=model_cfg.get("temperature", 0.7),
        top_k=model_cfg.get("top_k", 2),
    )
    chosen_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else FINAL_MODEL_PATH
    if not chosen_path.exists():
        raise FileNotFoundError(
            f"Nenhum modelo encontrado. Esperado {BEST_MODEL_PATH} ou {FINAL_MODEL_PATH}. Rode o treino antes."
        )
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
    df = load_btc_1h(days=max(eval_days, 120))
    dataset = prepare_dataset(df)
    price_cols = ["open", "high", "low", "close", "volume"]
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    policy = _load_policy(feat_df.shape[1], cfg)

    # janela de avaliação
    hours = eval_days * 24
    prices = price_df.tail(hours).reset_index(drop=True)
    feats = feat_df.tail(hours).reset_index(drop=True)
    env = BTCMixtureEnv(prices, feats, env_cfg)

    obs = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)

    weights_trace: List[np.ndarray] = []
    masks_trace: List[np.ndarray] = []
    actions: List[int] = []
    equities: List[float] = []

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
    header = [f"w_e{i}" for i in range(weights_arr.shape[1])] + [f"m_e{i}" for i in range(masks_arr.shape[1])] + ["action", "equity"]
    import csv

    with trace_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in range(weights_arr.shape[0]):
            writer.writerow([*weights_arr[t].tolist(), *masks_arr[t].tolist(), actions[t], equities[t]])
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
