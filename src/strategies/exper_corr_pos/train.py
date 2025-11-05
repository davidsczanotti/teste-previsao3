from __future__ import annotations

import argparse
import json
from pathlib import Path
import csv
import hashlib
import random
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch

from .data import load_primary_series, load_confirm_series, prepare_dataset
from .env import BTCMixtureEnv, EnvConfig
from .models import PPOConfig
from .utils_cfg import build_policy
from .trainer import PPOTrainer


DEFAULT_CONFIG = Path("src/strategies/exper_corr_pos/config.json")
MANIFEST_PATH = Path("src/strategies/exper_corr_pos/reports/train/run_manifest.json")


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depende de GPU
        torch.cuda.manual_seed_all(seed)


def _record_manifest(config: dict, cfg_path: Path, seed: int) -> None:
    env_cfg = config.get("env", {})
    train_cfg = config.get("train", {})
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config_path": str(cfg_path),
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "seed": seed,
        "accounting_mode": env_cfg.get("accounting_mode", "mtm"),
        "eval": {
            "eval_every": train_cfg.get("eval_every"),
            "eval_days": train_cfg.get("eval_days"),
            "random_start": env_cfg.get("random_start"),
            "window_bars": env_cfg.get("window_bars"),
        },
        "risk": {
            "equity_floor_pct": env_cfg.get("equity_floor_pct"),
            "max_drawdown_pct": env_cfg.get("max_drawdown_pct"),
            "drawdown_kill_bars": env_cfg.get("drawdown_kill_bars"),
            "leverage": env_cfg.get("leverage"),
            "dynamic_position": env_cfg.get("dynamic_position"),
            "stop_atr_mult": env_cfg.get("stop_atr_mult"),
            "trail_atr_mult": env_cfg.get("trail_atr_mult"),
        },
    }

    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        if commit:
            entry["git_commit"] = commit
    except Exception:  # pragma: no cover - git pode não estar disponível
        pass

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text())
            if not isinstance(data, list):
                data = [data]
        except Exception:
            data = []
    else:
        data = []

    data.append(entry)
    MANIFEST_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MoE PPO agent (config-driven)")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to JSON configuration (default: src/strategies/exper_corr_pos/config.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    config = json.loads(cfg_path.read_text())

    train_cfg = config.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    _set_seeds(seed)
    _record_manifest(config, cfg_path, seed)

    primary_df = load_primary_series(config)
    confirm_df = load_confirm_series(config)
    dataset = prepare_dataset(primary_df, config=config, confirm_df=confirm_df)

    price_cols = ["open", "high", "low", "close", "volume"]
    timestamps = dataset.index.to_list()
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    env_cfg = EnvConfig(**config.get("env", {}))
    env = BTCMixtureEnv(price_df, feat_df, env_cfg, timestamps=timestamps)

    input_dim = feat_df.shape[1]
    policy = build_policy(input_dim, config)
    ppo_cfg = PPOConfig(**config.get("ppo", {}))
    trainer = PPOTrainer(
        policy,
        ppo_cfg,
        device=torch.device(train_cfg.get("device", "cpu")),
        lb_coef=float(train_cfg.get("lb_coef", 0.01)),
    )

    outdir = Path(train_cfg.get("outdir", "src/strategies/exper_corr_pos/reports/train"))
    outdir.mkdir(parents=True, exist_ok=True)

    # Resume logic: if enabled and checkpoint exists, warm-start the policy
    resume = bool(train_cfg.get("resume", False))
    resume_path_str = train_cfg.get("resume_path")
    resume_path = Path(resume_path_str) if resume_path_str else (outdir / "moe_policy_final.pt")
    if resume:
        try:
            if resume_path.exists():
                state = torch.load(resume_path, map_location="cpu")
                policy.load_state_dict(state)
                print(f"[resume] Carregado checkpoint de {resume_path}")
            else:
                print(f"[resume] Arquivo não encontrado em {resume_path}; iniciando do zero.")
        except Exception as e:
            print(f"[resume] Falha ao carregar {resume_path}: {e}. Iniciando do zero.")

    episodes = int(train_cfg.get("episodes", 500))
    rollout_steps = int(train_cfg.get("rollout_steps", 2048))
    log_every = int(train_cfg.get("log_every", 5))
    plot_every = int(train_cfg.get("plot_every", 50))
    eval_every = int(train_cfg.get("eval_every", 50))
    eval_days = int(train_cfg.get("eval_days", 90))
    ckpt_every = int(train_cfg.get("ckpt_every", 10))
    final_every = int(train_cfg.get("final_every", 0))  # 0 = só no fim

    metrics_path = outdir / "metrics.csv"
    usage_cols = [f"usage_e{i}" for i in range(policy.num_experts)]
    # Se já existir um CSV antigo sem as colunas novas, rotaciona para *_legacy.csv
    if metrics_path.exists():
        try:
            with metrics_path.open("r") as f:
                header = f.readline().strip()
            expected_cols = [
                "episode",
                "policy_loss",
                "value_loss",
                "entropy",
                "entropy_coef",
                "load_balance",
                "avg_reward",
                "sum_reward",
            ] + usage_cols + ["greedy_equity", "greedy_ruined"]

            header_ok = header.startswith("episode,")
            for col in ("entropy_coef", "greedy_ruined"):
                header_ok &= col in header
            if usage_cols:
                header_ok &= usage_cols[0] in header

            if not header_ok:
                legacy_path = outdir / "metrics_legacy.csv"
                metrics_path.rename(legacy_path)
                print(f"[metrics] CSV antigo movido para {legacy_path} e um novo será criado.")
        except Exception as e:
            print(f"[metrics] Falha ao inspecionar CSV existente: {e}")

    if not metrics_path.exists():
        with metrics_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "policy_loss",
                "value_loss",
                "entropy",
                "entropy_coef",
                "load_balance",
                "avg_reward",
                "sum_reward",
                *usage_cols,
                "greedy_equity",
                "greedy_ruined",
            ])

    def _append_metrics(
        ep: int,
        m: dict,
        greedy_equity: float | None = None,
        greedy_ruined: bool | None = None,
        entropy_coef_val: float | None = None,
    ) -> None:
        with metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            row = [
                ep,
                m.get("policy_loss"),
                m.get("value_loss"),
                m.get("entropy"),
                entropy_coef_val if entropy_coef_val is not None else ppo_cfg.entropy_coef,
                m.get("load_balance"),
                m.get("avg_reward"),
                m.get("sum_reward"),
            ]
            usage = m.get("expert_usage") or []
            # garante número fixo de colunas
            usage = list(usage) + [""] * max(0, len(usage_cols) - len(usage))
            row.extend(usage[: len(usage_cols)])
            row.append(greedy_equity if greedy_equity is not None else "")
            row.append(int(greedy_ruined) if greedy_ruined is not None else "")
            writer.writerow(row)

    def _plot_metrics() -> None:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            dfm = pd.read_csv(metrics_path)
            plt.style.use("dark_background")
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            ax = axes[0, 0]
            ax.plot(dfm["episode"], dfm["policy_loss"], label="policy_loss")
            ax.plot(dfm["episode"], dfm["value_loss"], label="value_loss")
            ax.legend(); ax.set_title("Losses")

            ax = axes[0, 1]
            ax.plot(dfm["episode"], dfm["entropy"], label="entropy")
            if "entropy_coef" in dfm.columns:
                ax.plot(dfm["episode"], dfm["entropy_coef"], label="entropy_coef")
            ax.plot(dfm["episode"], dfm["load_balance"], label="load_balance")
            ax.legend(); ax.set_title("Entropy / Balance")

            ax = axes[1, 0]
            ax.plot(dfm["episode"], dfm["avg_reward"], label="avg_reward")
            ax.plot(dfm["episode"], dfm["sum_reward"], label="sum_reward")
            ax.legend(); ax.set_title("Rewards")

            ax = axes[1, 1]
            if "greedy_equity" in dfm.columns:
                ge = pd.to_numeric(dfm["greedy_equity"], errors="coerce")
                mask = ge.notna()
                if mask.any():
                    ax.plot(dfm["episode"][mask], ge[mask], label="greedy_equity", marker="o", markersize=3)
                    y_min, y_max = float(ge[mask].min()), float(ge[mask].max())
                    margin = max(1.0, (y_max - y_min) * 0.1)
                    ax.set_ylim(y_min - margin, y_max + margin)
                    if "greedy_ruined" in dfm.columns:
                        ruined_mask = pd.to_numeric(dfm["greedy_ruined"], errors="coerce").fillna(0).astype(bool)
                        ruined_mask &= mask
                        if ruined_mask.any():
                            ax.scatter(
                                dfm["episode"][ruined_mask],
                                ge[ruined_mask],
                                color="#ff6666",
                                label="ruína",
                                marker="x",
                                s=30,
                            )
                else:
                    ax.text(0.5, 0.5, "sem avaliações ainda", transform=ax.transAxes, ha="center", va="center")
            ax.legend(); ax.set_title("Greedy equity (eval)")

            fig.tight_layout()
            fig.savefig(outdir / "metrics.png", dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"[plot] Falha ao gerar metrics.png: {e}")

    def _plot_usage(window: int = 100) -> None:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            if not usage_cols:
                return
            dfm = pd.read_csv(metrics_path)
            cols = [c for c in usage_cols if c in dfm.columns]
            if not cols:
                return
            tail = dfm.tail(window)
            means = tail[cols].mean(numeric_only=True)
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(cols)), means.values, color="#4da6ff")
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45)
            ax.set_ylim(0, 1)
            last_episode = int(tail["episode"].iloc[-1]) if "episode" in tail.columns and not tail.empty else 0
            ax.set_title(
                f"Expert usage (média das últimas {len(tail)} execuções — até ep {last_episode})"
            )
            fig.tight_layout()
            fig.savefig(outdir / "expert_usage.png", dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"[plot] Falha ao gerar expert_usage.png: {e}")

    def _greedy_eval() -> tuple[float, bool]:
        try:
            # pequena avaliação em uma janela do fim dos dados
            hours = eval_days * 24
            tail_prices = price_df.tail(hours).reset_index(drop=True)
            tail_feats = feat_df.tail(hours).reset_index(drop=True)
            # Avaliação determinística: força random_start=False e janela completa
            eval_cfg = EnvConfig(**env_cfg.__dict__)
            eval_cfg.random_start = False
            eval_cfg.window_bars = 0
            tail_ts = timestamps[-len(tail_prices) :] if len(tail_prices) > 0 else []
            eval_env = BTCMixtureEnv(tail_prices, tail_feats, eval_cfg, timestamps=tail_ts)
            obs = torch.tensor(eval_env.reset(), dtype=torch.float32)
            done = False
            info = {"equity": float(env_cfg.init_equity)}
            ruined = False
            while not done:
                with torch.no_grad():
                    dist, _, _ = policy(obs.unsqueeze(0))
                    action = torch.argmax(dist.probs, dim=-1).item()
                next_obs, _, done, info = eval_env.step(action)
                if info.get("ruined"):
                    ruined = True
                obs = torch.tensor(next_obs, dtype=torch.float32)
            return float(info.get("equity", 0.0)), ruined
        except Exception as e:
            print(f"[eval] Falha na avaliação greedy: {e}")
            return float("nan"), True

    episode_offset = 0
    best_greedy = float("-inf")
    if metrics_path.exists():
        try:
            import pandas as pd

            df_prev = pd.read_csv(metrics_path)
            if not df_prev.empty:
                if "episode" in df_prev.columns:
                    episode_offset = int(pd.to_numeric(df_prev["episode"], errors="coerce").max())
                if "greedy_equity" in df_prev.columns:
                    prev_greedy = pd.to_numeric(df_prev["greedy_equity"], errors="coerce")
                    if pd.notna(prev_greedy).any():
                        best_greedy = float(prev_greedy.max())
        except Exception as e:
            print(f"[metrics] Não foi possível ler métricas anteriores: {e}")

    best_greedy = float(best_greedy)
    best_path = outdir / "moe_policy_best_eval.pt"

    # Entropy schedule (opcional): linear start->end em 'entropy_decay_episodes'
    ent_start = float(train_cfg.get("entropy_coef_start", ppo_cfg.entropy_coef))
    ent_end = float(train_cfg.get("entropy_coef_end", ent_start))
    ent_decay_episodes = int(train_cfg.get("entropy_decay_episodes", episodes))

    for episode in range(1, episodes + 1):
        # atualiza coeficiente de entropia conforme agenda
        if ent_decay_episodes > 0:
            progress = min(1.0, episode / float(ent_decay_episodes))
            current_entropy_coef = ent_start + (ent_end - ent_start) * progress
            trainer.cfg.entropy_coef = float(current_entropy_coef)
        else:
            current_entropy_coef = trainer.cfg.entropy_coef
        metrics = trainer.train_step(env, rollout_steps)
        actual_episode = episode_offset + episode

        greedy_equity = None
        greedy_ruined = None
        if eval_every > 0 and episode % eval_every == 0:
            greedy_equity, greedy_ruined = _greedy_eval()
            if (
                greedy_equity == greedy_equity
                and not greedy_ruined
                and greedy_equity > best_greedy
            ):
                best_greedy = greedy_equity
                torch.save(policy.state_dict(), best_path)
                print(f"[eval] Novo melhor greedy_equity={best_greedy:.2f} salvo em {best_path}")

        _append_metrics(
            actual_episode,
            metrics,
            greedy_equity,
            greedy_ruined,
            entropy_coef_val=current_entropy_coef,
        )

        if plot_every > 0 and episode % plot_every == 0:
            _plot_metrics()
            _plot_usage(window=int(train_cfg.get("usage_window", train_cfg.get("plot_every", 100))))

        if ckpt_every > 0 and episode % ckpt_every == 0:
            ckpt_path = outdir / f"moe_policy_ep{episode}.pt"
            torch.save(policy.state_dict(), ckpt_path)
        if final_every > 0 and episode % final_every == 0:
            # alias de conveniência para scripts que buscam 'final'
            torch.save(policy.state_dict(), outdir / "moe_policy_final.pt")
        if episode % log_every == 0:
            print(f"Episode {actual_episode} (run {episode}): {metrics}")

    final_path = outdir / "moe_policy_final.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"Treinamento finalizado. Modelo salvo em {final_path}")


if __name__ == "__main__":
    main()
