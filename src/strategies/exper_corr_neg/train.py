from __future__ import annotations

import argparse
import json
from pathlib import Path
import csv

import torch

from .data import load_btc_1h, prepare_dataset
from .env import BTCMixtureEnv, EnvConfig
from .models import MoEPolicy, PPOConfig
from .trainer import PPOTrainer


DEFAULT_CONFIG = Path("src/strategies/exper_corr_neg/config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MoE PPO agent (config-driven)")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to JSON configuration (default: src/strategies/exper_corr_neg/config.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    config = json.loads(cfg_path.read_text())

    df = load_btc_1h(days=3650)
    dataset = prepare_dataset(df)

    price_cols = ["open", "high", "low", "close", "volume"]
    price_df = dataset[price_cols]
    feat_df = dataset.drop(columns=price_cols)

    env_cfg = EnvConfig(**config.get("env", {}))
    env = BTCMixtureEnv(price_df, feat_df, env_cfg)

    input_dim = feat_df.shape[1]
    model_cfg = config.get("model", {})
    policy = MoEPolicy(
        input_dim=input_dim,
        num_actions=3,
        expert_hidden=model_cfg.get("expert_hidden", [64, 32]),
        gating_hidden=model_cfg.get("gating_hidden", [64, 32]),
        num_experts=model_cfg.get("num_experts", 5),
        temperature=model_cfg.get("temperature", 0.7),
        top_k=model_cfg.get("top_k", 2),
    )
    ppo_cfg = PPOConfig(**config.get("ppo", {}))
    train_cfg = config.get("train", {})
    trainer = PPOTrainer(
        policy,
        ppo_cfg,
        device=torch.device(train_cfg.get("device", "cpu")),
        lb_coef=float(train_cfg.get("lb_coef", 0.01)),
    )

    outdir = Path(train_cfg.get("outdir", "reports/exper_corr_neg/train"))
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

    metrics_path = outdir / "metrics.csv"
    usage_cols = [f"usage_e{i}" for i in range(policy.num_experts)]
    # Se já existir um CSV antigo sem as colunas novas, rotaciona para *_legacy.csv
    if metrics_path.exists():
        try:
            with metrics_path.open("r") as f:
                header = f.readline()
            if (
                "entropy_coef" not in header
                or (usage_cols and usage_cols[0] not in header)
                or "greedy_ruined" not in header
            ):
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
            eval_env = BTCMixtureEnv(tail_prices, tail_feats, env_cfg)
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

        if episode % 10 == 0:
            ckpt_path = outdir / f"moe_policy_ep{episode}.pt"
            torch.save(policy.state_dict(), ckpt_path)
        if episode % log_every == 0:
            print(f"Episode {actual_episode} (run {episode}): {metrics}")

    final_path = outdir / "moe_policy_final.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"Treinamento finalizado. Modelo salvo em {final_path}")


if __name__ == "__main__":
    main()
