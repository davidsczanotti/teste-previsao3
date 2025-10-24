from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import load_btc_1h, prepare_dataset
from .env import BTCMixtureEnv, EnvConfig
from .models import MoEPolicy

CFG_PATH = Path("src/strategies/exper_corr_neg/config.json")
FINAL_MODEL_PATH = Path("src/strategies/exper_corr_neg/reports/train/moe_policy_final.pt")
BEST_MODEL_PATH = Path("src/strategies/exper_corr_neg/reports/train/moe_policy_best_eval.pt")
TRAIN_DIR = Path("src/strategies/exper_corr_neg/reports/train")
OUT_PATH = Path("src/strategies/exper_corr_neg/reports/train/visual_backtest.png")


def _find_checkpoint() -> Path:
    if BEST_MODEL_PATH.exists():
        return BEST_MODEL_PATH
    if FINAL_MODEL_PATH.exists():
        return FINAL_MODEL_PATH
    # fallback: último checkpoint epXX
    if TRAIN_DIR.exists():
        eps = sorted(TRAIN_DIR.glob("moe_policy_ep*.pt"))
        if eps:
            return eps[-1]
    raise FileNotFoundError(
        f"Nenhum modelo encontrado. Esperado {BEST_MODEL_PATH} ou {FINAL_MODEL_PATH} ou 'moe_policy_ep*.pt'. Aguarde salvar um checkpoint (10,20,...) ou reduza 'episodes'."
    )


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
    # Preferir best_eval, depois final; fallback para último epXX
    chosen_path = _find_checkpoint()
    # Carregamento tolerante: se o checkpoint for de outra arquitetura, faz load parcial
    state_dict = torch.load(chosen_path, map_location="cpu")
    model_state = policy.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    missing = [k for k in model_state.keys() if k not in filtered]
    policy.load_state_dict(filtered, strict=False)
    loaded_frac = f"{len(filtered)}/{len(model_state)}"
    print(f"[visualize] Carregado (parcial={loaded_frac}) de {chosen_path}")
    if missing:
        print("[visualize] Aviso: pesos ignorados por incompatibilidade (arquitetura mudou). Recomenda-se treinar novamente.")
    policy.eval()
    return policy


def _expert_names(cfg: dict, num_experts: int) -> list[str]:
    names = cfg.get("model", {}).get("expert_names")
    if isinstance(names, list) and len(names) == num_experts:
        return [str(n) for n in names]
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


def _run_policy(env: BTCMixtureEnv, policy: MoEPolicy) -> Tuple[np.ndarray, np.ndarray]:
    """Executa a política greedy e retorna ações (−1/0/+1) e equity."""
    obs = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
    actions = []
    equity = []
    done = False
    while not done:
        with torch.no_grad():
            dist, _, _ = policy(obs)
            action = torch.argmax(dist.probs, dim=-1).item()
        next_obs, _, done, info = env.step(action)
        actions.append(action - 1)  # mapeia {0,1,2} -> {-1,0,+1}
        equity.append(info.get("equity", 0.0))
        obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
    return np.asarray(actions), np.asarray(equity)


def main() -> None:
    cfg = json.loads(CFG_PATH.read_text())

    # usa janela curta por padrão para ficar didático/rápido
    df = load_btc_1h(days=90)
    dataset = prepare_dataset(df)
    price_cols = ["open", "high", "low", "close", "volume"]
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    env_cfg = EnvConfig(**cfg.get("env", {}))
    env = BTCMixtureEnv(price_df, feat_df, env_cfg)

    policy = _load_policy(feat_df.shape[1], cfg)
    actions, equity = _run_policy(env, policy)

    closes = price_df["close"].to_numpy()[: len(actions)]
    idx = np.arange(len(actions))

    plt.style.use("dark_background")
    fig, ax_price = plt.subplots(figsize=(12, 6))
    ax_price.plot(idx, closes, color="#4da6ff", linewidth=1.2, label="Preço")

    # marca entradas long/short para visualização rápida
    longs = actions == 1
    shorts = actions == -1
    ax_price.scatter(idx[longs], closes[longs], color="#6cff6c", marker="^", s=40, label="Long")
    ax_price.scatter(idx[shorts], closes[shorts], color="#ff6666", marker="v", s=40, label="Short")

    ax_price.set_xlabel("Índice do candle")
    ax_price.set_ylabel("Preço BTCUSDT (1h)")
    ax_price.legend(loc="upper left")

    ax_equity = ax_price.twinx()
    ax_equity.plot(idx, equity, color="#f0c674", linewidth=1.0, label="Equity")
    ax_equity.set_ylabel("Equity lógica")
    ax_equity.legend(loc="upper right")

    fig.suptitle("Agente MoE — Visualização das ações (BTCUSDT 1h)")
    # legenda didática com nomes dos experts
    names = _expert_names(cfg, cfg.get("model", {}).get("num_experts", 5))
    legend_text = " | ".join([f"e{i}:{n}" for i, n in enumerate(names)])
    fig.text(0.01, 0.01, legend_text, fontsize=8, color="#bdbdbd")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Gráfico salvo em {OUT_PATH}")


if __name__ == "__main__":
    main()
