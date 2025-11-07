from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import load_primary_series, load_confirm_series, prepare_dataset
from .env import BTCMixtureEnv, EnvConfig
from .models import MoEPolicy
from .utils_cfg import build_policy, enabled_expert_names, bars_for_days

CFG_PATH = Path("src/strategies/exper_corr_pos/config.json")
FINAL_MODEL_PATH = Path("src/strategies/exper_corr_pos/reports/train/moe_policy_final.pt")
BEST_MODEL_PATH = Path("src/strategies/exper_corr_pos/reports/train/moe_policy_best_eval.pt")
TRAIN_DIR = Path("src/strategies/exper_corr_pos/reports/train")
OUT_PATH = Path("src/strategies/exper_corr_pos/reports/train/visual_backtest.png")


def _compute_donchian(high: np.ndarray, low: np.ndarray, period: int = 55) -> Tuple[np.ndarray, np.ndarray]:
    if len(high) == 0:
        return np.array([]), np.array([])
    upper = np.full_like(high, np.nan, dtype=np.float64)
    lower = np.full_like(low, np.nan, dtype=np.float64)
    for i in range(period - 1, len(high)):
        window_high = high[i - period + 1 : i + 1]
        window_low = low[i - period + 1 : i + 1]
        upper[i] = np.max(window_high)
        lower[i] = np.min(window_low)
    return upper, lower


def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    if len(close) < period + 1:
        return np.zeros_like(close)
    delta = np.diff(close)
    gains = np.clip(delta, 0.0, None)
    losses = -np.clip(delta, None, 0.0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[period] = gains[:period].mean()
    avg_loss[period] = losses[:period].mean()
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = np.divide(avg_gain, avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[: period + 1] = 50.0
    return rsi


def _find_checkpoint(prefer: str = "latest") -> Path:
    """Escolhe o checkpoint a carregar.

    prefer:
      - "best": sempre usa moe_policy_best_eval.pt se existir
      - "final": sempre usa moe_policy_final.pt se existir
      - "latest": escolhe o arquivo mais recente entre best/final/epXX
    """
    prefer = str(prefer or "latest").lower()

    candidates = []
    if BEST_MODEL_PATH.exists():
        candidates.append(BEST_MODEL_PATH)
    if FINAL_MODEL_PATH.exists():
        candidates.append(FINAL_MODEL_PATH)
    if TRAIN_DIR.exists():
        candidates.extend(sorted(TRAIN_DIR.glob("moe_policy_ep*.pt")))

    if not candidates:
        raise FileNotFoundError(
            f"Nenhum modelo encontrado. Esperado {BEST_MODEL_PATH} ou {FINAL_MODEL_PATH} ou 'moe_policy_ep*.pt'."
        )

    if prefer == "best" and BEST_MODEL_PATH.exists():
        return BEST_MODEL_PATH
    if prefer == "final" and FINAL_MODEL_PATH.exists():
        return FINAL_MODEL_PATH

    # latest por mtime
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _load_policy(input_dim: int, cfg: dict) -> Tuple[MoEPolicy, Path]:
    policy = build_policy(input_dim, cfg)
    # Preferência pode ser definida em config.visualize.prefer (best|final|latest)
    prefer = str(cfg.get("visualize", {}).get("prefer", "latest")).lower()
    chosen_path = _find_checkpoint(prefer)
    # Carregamento tolerante: se o checkpoint for de outra arquitetura, faz load parcial
    state_dict = torch.load(chosen_path, map_location="cpu")
    model_state = policy.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    missing = [k for k in model_state.keys() if k not in filtered]
    policy.load_state_dict(filtered, strict=False)
    loaded_frac = f"{len(filtered)}/{len(model_state)}"
    try:
        mtime = datetime.fromtimestamp(chosen_path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00","Z")
    except Exception:
        mtime = "?"
    print(f"[visualize] Carregado (parcial={loaded_frac}) de {chosen_path} (mtime={mtime})")
    if missing:
        print("[visualize] Aviso: pesos ignorados por incompatibilidade (arquitetura mudou). Recomenda-se treinar novamente.")
    policy.eval()
    return policy, chosen_path


def _expert_names(cfg: dict, num_experts: int) -> list[str]:
    names = enabled_expert_names(cfg)
    if len(names) == num_experts:
        return names
    # fallback se houver inconsistência com o checkpoint
    defaults = ["TrendML", "MultiFrame", "Spread", "Pattern"]
    return (names or defaults)[:num_experts]


def _run_policy(env: BTCMixtureEnv, policy: MoEPolicy) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executa a política greedy e retorna ações (−1/0/+1), equity e pesos do gate."""
    obs = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
    actions = []
    equity = []
    weights_hist = []
    done = False
    while not done:
        with torch.no_grad():
            dist, _, _ = policy(obs)
            action = torch.argmax(dist.probs, dim=-1).item()
            gate_weights, _ = policy.gating(obs, top_k=policy.top_k)
        next_obs, _, done, info = env.step(action)
        actions.append(action - 1)  # mapeia {0,1,2} -> {-1,0,+1}
        equity.append(info.get("equity", 0.0))
        weights_hist.append(gate_weights.squeeze(0).cpu().numpy())
        obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
    return np.asarray(actions), np.asarray(equity), np.asarray(weights_hist)


def main() -> None:
    cfg = json.loads(CFG_PATH.read_text())

    # Janela de dados para visualização
    data_cfg = cfg.get("data", {})
    timeframe = str(data_cfg.get("timeframe") or "").strip()
    if not timeframe:
        raise ValueError("Parâmetro obrigatório ausente: data.timeframe no config.json")
    vis_days = int(cfg.get("visualize", {}).get("days", 90))
    primary_df = load_primary_series(cfg)
    confirm_df = load_confirm_series(cfg)
    spread_window = int(data_cfg.get("spread_window", 240))
    min_bars = max(spread_window + 20, bars_for_days(timeframe, vis_days) if vis_days > 0 else 0)
    if min_bars > 0:
        primary_trimmed = primary_df.tail(min_bars)
    else:
        primary_trimmed = primary_df
    dataset = prepare_dataset(primary_trimmed, config=cfg, confirm_df=confirm_df)
    if dataset.empty:
        # fallback para série completa
        dataset = prepare_dataset(primary_df, config=cfg, confirm_df=confirm_df)
        if dataset.empty:
            raise RuntimeError(
                "Dataset vazio após aplicar filtros para visualização. Aumente visualize.days ou ajuste spread_window no config."
            )

    price_cols = ["open", "high", "low", "close", "volume"]
    timestamps = dataset.index.to_list()
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    env_cfg = EnvConfig(**cfg.get("env", {}))
    # Garante que random_start funcione com janelas menores (window_bars < len)
    if env_cfg.random_start:
        total = len(price_df)
        if env_cfg.window_bars <= 0 or env_cfg.window_bars >= total:
            env_cfg.window_bars = max(1, total - 1)
    env = BTCMixtureEnv(price_df, feat_df, env_cfg, timestamps=timestamps)

    policy, chosen_ckpt = _load_policy(feat_df.shape[1], cfg)
    names = _expert_names(cfg, policy.num_experts)
    actions, equity, gate_weights = _run_policy(env, policy)

    closes = price_df["close"].to_numpy()[: len(actions)]
    idx = np.arange(len(actions))

    # Extrai indicadores de tendência usados pelos experts para contexto visual
    ema_fast = dataset.get("ema_fast")
    ema_slow = dataset.get("ema_slow")
    htf_trend = dataset.get("htf_trend_state")
    if ema_fast is not None and ema_slow is not None:
        ema_fast = ema_fast.to_numpy()[: len(actions)]
        ema_slow = ema_slow.to_numpy()[: len(actions)]
    else:
        ema_fast = ema_slow = None
    if htf_trend is not None:
        htf_trend = htf_trend.to_numpy()[: len(actions)]

    plt.style.use("dark_background")
    fig, (ax_price, ax_rsi, ax_weights) = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.1, 1.1]},
    )
    fig.subplots_adjust(hspace=0.08)
    ax_price.plot(idx, closes, color="#4da6ff", linewidth=1.2, label="Preço")
    if ema_fast is not None and ema_slow is not None:
        ax_price.plot(idx, ema_fast, color="#63ff85", linewidth=1.0, label="EMA rápida")
        ax_price.plot(idx, ema_slow, color="#ff5f5f", linewidth=1.0, label="EMA lenta")

    upper, lower = _compute_donchian(
        price_df["high"].to_numpy(dtype=np.float64)[: len(actions)],
        price_df["low"].to_numpy(dtype=np.float64)[: len(actions)],
        period=55,
    )
    if not np.isnan(upper).all():
        ax_price.plot(idx, upper, color="#cccccc", linewidth=0.8, linestyle="--", label="Donchian sup")
        ax_price.plot(idx, lower, color="#999999", linewidth=0.8, linestyle="--", label="Donchian inf")

    ax_price.legend(loc="upper left")

    # marca entradas long/short para visualização rápida
    longs = actions == 1
    shorts = actions == -1
    ax_price.scatter(idx[longs], closes[longs], color="#6cff6c", marker="^", s=40, label="Long")
    ax_price.scatter(idx[shorts], closes[shorts], color="#ff6666", marker="v", s=40, label="Short")

    ax_price.set_xlabel("Índice do candle")
    ax_price.set_ylabel(f"Preço {data_cfg.get('base_symbol', 'BTCUSDT')} ({timeframe})")

    ax_equity = ax_price.twinx()
    ax_equity.plot(idx, equity, color="#f0c674", linewidth=1.0, label="Equity")
    ax_equity.set_ylabel("Equity lógica")
    ax_equity.legend(loc="upper right")

    if htf_trend is not None:
        ax_trend = ax_price.twinx()
        ax_trend.step(idx, htf_trend, where="post", color="#8ecae6", alpha=0.3, linewidth=1.0, label="Trend HTF")
        ax_trend.set_yticks([-1, 0, 1])
        ax_trend.set_ylim(-1.5, 1.5)
        ax_trend.spines["right"].set_position(("axes", 1.08))
        ax_trend.tick_params(axis="y", colors="#8ecae6")
        ax_trend.set_ylabel("HTF trend", color="#8ecae6")
        ax_trend.legend(loc="lower right")

    # RSI subpainel
    rsi = _compute_rsi(closes, period=14)
    ax_rsi.plot(idx, rsi, color="#ffb703", linewidth=1.0, label="RSI 14")
    ax_rsi.axhline(70, color="#ff5f5f", linestyle="--", linewidth=0.8)
    ax_rsi.axhline(30, color="#63ff85", linestyle="--", linewidth=0.8)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc="upper left")

    # Pesos do gate
    if gate_weights.size > 0:
        for i, name in enumerate(names):
            if i >= gate_weights.shape[1]:
                break
            ax_weights.plot(idx[: gate_weights.shape[0]], gate_weights[:, i], linewidth=0.9, label=name)
    ax_weights.set_ylabel("Peso gate")
    ax_weights.set_ylim(0, 1)
    ax_weights.legend(loc="upper left", ncol=max(1, len(names) // 2))
    ax_weights.set_xlabel("Índice do candle")

    fig.suptitle(f"Agente MoE — Confluência de Correlação Positiva ({data_cfg.get('base_symbol', 'BTCUSDT')} {timeframe})")
    # legenda didática com nomes dos experts
    legend_text = " | ".join([f"e{i}:{n}" for i, n in enumerate(names)])
    fig.text(0.01, 0.01, legend_text, fontsize=8, color="#bdbdbd")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Salva arquivo padrão e também um snapshot identificado pelo checkpoint
    fig.savefig(OUT_PATH, dpi=150)
    ckpt_tag = chosen_ckpt.stem
    snapshot_path = OUT_PATH.parent / f"visual_backtest-{ckpt_tag}.png"
    try:
        fig.savefig(snapshot_path, dpi=150)
    except Exception:
        snapshot_path = None
    print(f"Gráfico salvo em {OUT_PATH}")
    if snapshot_path:
        print(f"Snapshot salvo em {snapshot_path}")


if __name__ == "__main__":
    main()
