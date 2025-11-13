from __future__ import annotations

import argparse
import json
from pathlib import Path
import csv
import hashlib
import random
import subprocess
from datetime import datetime, timezone
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import sys
import importlib
import re

import numpy as np
import torch

from .data import load_primary_series, load_confirm_series, prepare_dataset
from .env import BTCMixtureEnv, EnvConfig
from .models import PPOConfig
from .utils_cfg import build_policy, bars_for_days
from .trainer import PPOTrainer

try:  # Optional Weights & Biases integration
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None  # type: ignore


DEFAULT_CONFIG = Path("src/strategies/exper_corr_pos/config.json")
MANIFEST_PATH = Path("src/strategies/exper_corr_pos/reports/train/run_manifest.json")


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depende de GPU
        torch.cuda.manual_seed_all(seed)


def _record_manifest(config: dict, cfg_path: Path, seed: int) -> None:
    def _collect_run_env() -> Dict[str, Any]:
        # Python version
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Poetry version (best-effort)
        poetry_ver: Optional[str] = None
        try:
            out = subprocess.check_output(["poetry", "--version"], stderr=subprocess.DEVNULL).decode().strip()
            # Examples: "Poetry (version 1.8.3)" or "Poetry version 1.8.3"
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            poetry_ver = m.group(1) if m else out
        except Exception:
            poetry_ver = None

        # Library versions (best-effort)
        libs = [
            "pandas",
            "numpy",
            "matplotlib",
            "torch",
            "optuna",
        ]
        lib_versions: Dict[str, Optional[str]] = {}
        for name in libs:
            try:
                mod = importlib.import_module(name)
                ver = getattr(mod, "__version__", None)
                lib_versions[name] = str(ver) if ver is not None else None
            except Exception:
                lib_versions[name] = None

        return {"python": py_ver, "poetry": poetry_ver, "lib_versions": lib_versions}

    env_cfg = config.get("env", {})
    train_cfg = config.get("train", {})
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config_path": str(cfg_path),
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "seed": seed,
        "accounting_mode": env_cfg.get("accounting_mode", "mtm"),
        "run_env": _collect_run_env(),
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


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_curriculum_phase(
    curriculum_cfg: Optional[Dict[str, Any]],
    env: BTCMixtureEnv,
    episode: int,
    default_rollout: int,
) -> int:
    if not curriculum_cfg:
        return default_rollout

    phases = curriculum_cfg.get("phases") or []
    chosen: Optional[Dict[str, Any]] = None
    for phase in phases:
        until = int(phase.get("until_episode", 0))
        if until <= 0:
            continue
        if episode <= until:
            chosen = phase
            break
    if chosen is None:
        chosen = curriculum_cfg.get("final", {})

    env_overrides = (chosen or {}).get("env", {})
    for attr, value in env_overrides.items():
        if hasattr(env.cfg, attr):
            setattr(env.cfg, attr, value)

    train_overrides = (chosen or {}).get("train", {})
    if "rollout_steps" in train_overrides:
        try:
            return int(train_overrides["rollout_steps"])
        except Exception:
            pass
    return default_rollout


def train_agent(
    config: Dict[str, Any],
    *,
    cfg_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    record_manifest: bool = True,
    enable_plots: bool = True,
    trial_id: Optional[str] = None,
    disable_wandb: bool = False,
) -> Dict[str, Any]:
    cfg = deepcopy(config)
    if overrides:
        _deep_update(cfg, overrides)

    train_cfg = cfg.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    _set_seeds(seed)

    logging_cfg = cfg.get("logging", {})
    wandb_cfg = logging_cfg.get("wandb", {}) or {}
    wandb_enabled = bool(wandb_cfg.get("enabled", False)) and not disable_wandb
    wandb_run = None
    if wandb_enabled and wandb is None:
        print("[wandb] habilitado no config, mas pacote não encontrado. Desativando.")
        wandb_enabled = False
    artifact_prefix = wandb_cfg.get("artifact_prefix", "exper_corr_pos")
    best_artifact_logged = False

    if record_manifest and cfg_path is not None:
        _record_manifest(cfg, cfg_path, seed)

    primary_df = load_primary_series(cfg)
    confirm_df = load_confirm_series(cfg)
    dataset = prepare_dataset(primary_df, config=cfg, confirm_df=confirm_df)

    data_cfg = cfg.get("data", {})
    timeframe = str(data_cfg.get("timeframe") or "").strip()
    if not timeframe:
        raise ValueError("Parâmetro obrigatório ausente: data.timeframe no config.json")

    price_cols = ["open", "high", "low", "close", "volume"]
    timestamps = dataset.index.to_list()
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    env_cfg = EnvConfig(**cfg.get("env", {}))
    env = BTCMixtureEnv(price_df, feat_df, env_cfg, timestamps=timestamps)

    input_dim = feat_df.shape[1]
    policy = build_policy(input_dim, cfg)
    ppo_cfg = PPOConfig(**cfg.get("ppo", {}))
    trainer = PPOTrainer(
        policy,
        ppo_cfg,
        device=torch.device(train_cfg.get("device", "cpu")),
        lb_coef=float(train_cfg.get("lb_coef", 0.01)),
    )

    if wandb_enabled:
        init_kwargs: Dict[str, Any] = {}
        for key in ("project", "entity", "name", "group", "job_type"):
            value = wandb_cfg.get(key)
            if value:
                init_kwargs[key] = value
        tags = wandb_cfg.get("tags")
        if tags:
            init_kwargs["tags"] = tags
        try:
            logged_config = json.loads(json.dumps(cfg))
        except TypeError:
            logged_config = {}
        wandb_run = wandb.init(config=logged_config, reinit=True, **init_kwargs)  # type: ignore[arg-type]
        wandb_run.define_metric("episode")
        wandb_run.define_metric("greedy_equity", step="episode")
        wandb_run.define_metric("avg_reward", step="episode")
        wandb_run.define_metric("sum_reward", step="episode")
        if wandb_cfg.get("watch", False):
            wandb.watch(  # type: ignore[call-arg]
                policy,
                log=wandb_cfg.get("watch_log", "gradients"),
                log_freq=int(wandb_cfg.get("watch_freq", 100)),
            )

    base_outdir = Path(train_cfg.get("outdir", "src/strategies/exper_corr_pos/reports/train"))
    outdir = base_outdir / trial_id if trial_id else base_outdir
    outdir.mkdir(parents=True, exist_ok=True)

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
    base_rollout_steps = int(train_cfg.get("rollout_steps", 2048))
    log_every_cfg = train_cfg.get("log_every")
    if log_every_cfg is None:
        log_every = max(1, episodes // 20)
    else:
        log_every = max(1, int(log_every_cfg))
    plot_every = int(train_cfg.get("plot_every", 50))
    eval_every = int(train_cfg.get("eval_every", 50))
    eval_days = int(train_cfg.get("eval_days", 90))
    ckpt_every = int(train_cfg.get("ckpt_every", 10))
    final_every = int(train_cfg.get("final_every", 0))
    usage_window = int(train_cfg.get("usage_window", train_cfg.get("plot_every", 100)))
    curriculum_cfg = train_cfg.get("curriculum")

    metrics_path = outdir / "metrics.csv"
    usage_cols = [f"usage_e{i}" for i in range(policy.num_experts)]
    if metrics_path.exists():
        try:
            with metrics_path.open("r") as f:
                header = f.readline().strip()
            header_ok = (
                header.startswith("episode,")
                and ("entropy_coef" in header)
                and ("lb_coef" in header)
                and ("greedy_ruined" in header)
            )
            try:
                cols = [c.strip() for c in header.split(",") if c]
                usage_in_header = [c for c in cols if c.startswith("usage_e")]
                if len(usage_in_header) != len(usage_cols):
                    header_ok = False
            except Exception:
                header_ok = False

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
                "lb_coef",
                "load_balance",
                "avg_reward",
                "sum_reward",
                *usage_cols,
                "greedy_equity",
                "greedy_ruined",
            ])

    def _append_metrics(
        ep: int,
        m: Dict[str, Any],
        greedy_equity: Optional[float] = None,
        greedy_ruined: Optional[bool] = None,
        entropy_coef_val: Optional[float] = None,
        lb_coef_val: Optional[float] = None,
    ) -> Dict[str, Any]:
        usage = list(m.get("expert_usage") or [])
        row_map: Dict[str, Any] = {
            "episode": ep,
            "policy_loss": m.get("policy_loss"),
            "value_loss": m.get("value_loss"),
            "entropy": m.get("entropy"),
            "entropy_coef": entropy_coef_val if entropy_coef_val is not None else ppo_cfg.entropy_coef,
            "lb_coef": lb_coef_val if lb_coef_val is not None else trainer.lb_coef,
            "load_balance": m.get("load_balance"),
            "avg_reward": m.get("avg_reward"),
            "sum_reward": m.get("sum_reward"),
            "greedy_equity": greedy_equity,
            "greedy_ruined": int(greedy_ruined) if greedy_ruined is not None else None,
        }
        for idx, col in enumerate(usage_cols):
            row_map[col] = usage[idx] if idx < len(usage) else None

        with metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            row = [
                row_map["episode"],
                row_map["policy_loss"],
                row_map["value_loss"],
                row_map["entropy"],
                row_map["entropy_coef"],
                row_map["lb_coef"],
                row_map["load_balance"],
                row_map["avg_reward"],
                row_map["sum_reward"],
            ]
            for col in usage_cols:
                value = row_map[col]
                row.append("" if value is None else value)
            row.append("" if greedy_equity is None else greedy_equity)
            row.append("" if row_map["greedy_ruined"] is None else row_map["greedy_ruined"])
            writer.writerow(row)

        return row_map

    def _plot_metrics() -> None:
        if not enable_plots or plot_every <= 0:
            return
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            dfm = pd.read_csv(metrics_path)
            plt.style.use("dark_background")
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            ax = axes[0, 0]
            ax.plot(dfm["episode"], dfm["policy_loss"], label="policy_loss")
            ax.plot(dfm["episode"], dfm["value_loss"], label="value_loss")
            ax.legend()
            ax.set_title("Losses")

            ax = axes[0, 1]
            ax.plot(dfm["episode"], dfm["entropy"], label="entropy")
            if "entropy_coef" in dfm.columns:
                ax.plot(dfm["episode"], dfm["entropy_coef"], label="entropy_coef")
            ax.plot(dfm["episode"], dfm["load_balance"], label="load_balance")
            ax.legend()
            ax.set_title("Entropy / Balance")

            ax = axes[1, 0]
            ax.plot(dfm["episode"], dfm["avg_reward"], label="avg_reward")
            ax.plot(dfm["episode"], dfm["sum_reward"], label="sum_reward")
            ax.legend()
            ax.set_title("Rewards")

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
            ax.legend()
            ax.set_title("Greedy equity (eval)")

            fig.tight_layout()
            fig.savefig(outdir / "metrics.png", dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"[plot] Falha ao gerar metrics.png: {e}")

    def _plot_usage() -> None:
        if not enable_plots or plot_every <= 0:
            return
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            if not usage_cols:
                return
            dfm = pd.read_csv(metrics_path)
            cols = [c for c in usage_cols if c in dfm.columns]
            if not cols:
                return
            tail = dfm.tail(usage_window)
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

    def _greedy_eval() -> Tuple[float, bool]:
        try:
            eval_bars = bars_for_days(timeframe, eval_days)
            tail_prices = price_df.tail(eval_bars).reset_index(drop=True)
            tail_feats = feat_df.tail(eval_bars).reset_index(drop=True)
            eval_cfg = EnvConfig(**env_cfg.__dict__)
            eval_cfg.random_start = False
            eval_cfg.window_bars = 0
            tail_ts = timestamps[-len(tail_prices) :] if len(tail_prices) > 0 else []
            eval_env = BTCMixtureEnv(tail_prices, tail_feats, eval_cfg, timestamps=tail_ts)
            obs = torch.tensor(eval_env.reset(), dtype=torch.float32)
            done = False
            info: Dict[str, Any] = {"equity": float(env_cfg.init_equity)}
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

    ent_start = float(train_cfg.get("entropy_coef_start", ppo_cfg.entropy_coef))
    ent_end = float(train_cfg.get("entropy_coef_end", ent_start))
    ent_decay_episodes = int(train_cfg.get("entropy_decay_episodes", episodes))

    # Schedule para lb_coef (balanceamento do gate)
    lb_start = float(train_cfg.get("lb_coef_start", train_cfg.get("lb_coef", 0.01)))
    lb_end = float(train_cfg.get("lb_coef_end", lb_start))
    lb_decay_episodes = int(train_cfg.get("lb_decay_episodes", ent_decay_episodes))
    trainer.lb_coef = lb_start

    last_metrics: Optional[Dict[str, Any]] = None
    for episode in range(1, episodes + 1):
        # Compute absolute episode index (respects resumed runs)
        actual_episode = episode_offset + episode
        if ent_decay_episodes > 0:
            ent_progress = min(1.0, episode / float(ent_decay_episodes))
            current_entropy_coef = ent_start + (ent_end - ent_start) * ent_progress
            trainer.cfg.entropy_coef = float(current_entropy_coef)
        else:
            current_entropy_coef = trainer.cfg.entropy_coef

        if lb_decay_episodes > 0:
            lb_progress = min(1.0, episode / float(lb_decay_episodes))
            current_lb_coef = lb_start + (lb_end - lb_start) * lb_progress
            trainer.lb_coef = float(current_lb_coef)
        else:
            current_lb_coef = trainer.lb_coef

        # Use absolute index to select curriculum phase when resuming
        rollout_steps = _apply_curriculum_phase(curriculum_cfg, env, actual_episode, base_rollout_steps)
        last_metrics = trainer.train_step(env, rollout_steps)

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
                if wandb_run:
                    wandb_run.log(
                        {"best_greedy": best_greedy, "best_episode": actual_episode},
                        step=actual_episode,
                    )
                    if not best_artifact_logged and best_path.exists():
                        artifact = wandb.Artifact(f"{artifact_prefix}-best", type="model")
                        artifact.add_file(str(best_path))
                        wandb_run.log_artifact(artifact)
                        best_artifact_logged = True

        row_data = _append_metrics(
            actual_episode,
            last_metrics or {},
            greedy_equity,
            greedy_ruined,
            entropy_coef_val=current_entropy_coef,
            lb_coef_val=current_lb_coef,
        )
        if wandb_run:
            log_payload = {k: v for k, v in row_data.items() if v is not None}
            log_payload["episode"] = actual_episode
            if greedy_ruined is not None:
                log_payload["greedy_ruined"] = bool(greedy_ruined)
            if greedy_equity is not None:
                log_payload["greedy_equity"] = greedy_equity
            wandb_run.log(log_payload, step=actual_episode)

        if plot_every > 0 and episode % plot_every == 0:
            _plot_metrics()
            _plot_usage()

        if ckpt_every > 0 and episode % ckpt_every == 0:
            ckpt_path = outdir / f"moe_policy_ep{episode}.pt"
            torch.save(policy.state_dict(), ckpt_path)
        if final_every > 0 and episode % final_every == 0:
            torch.save(policy.state_dict(), outdir / "moe_policy_final.pt")
        if log_every > 0 and episode % log_every == 0:
            print(f"Episode {actual_episode} (run {episode}): {last_metrics}")

    final_greedy, final_ruined = _greedy_eval()
    if (
        final_greedy == final_greedy
        and not final_ruined
        and final_greedy > best_greedy
    ):
        best_greedy = final_greedy
        torch.save(policy.state_dict(), best_path)
        print(f"[eval] Final greedy melhorado={best_greedy:.2f} salvo em {best_path}")
        if wandb_run:
            wandb_run.log(
                {"best_greedy": best_greedy, "best_episode": episode_offset + episodes},
                step=episode_offset + episodes,
            )
            if not best_artifact_logged and best_path.exists():
                artifact = wandb.Artifact(f"{artifact_prefix}-best", type="model")
                artifact.add_file(str(best_path))
                wandb_run.log_artifact(artifact)
                best_artifact_logged = True

    final_path = outdir / "moe_policy_final.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"Treinamento finalizado. Modelo salvo em {final_path}")
    if wandb_run:
        artifact = wandb.Artifact(f"{artifact_prefix}-final", type="model")
        artifact.add_file(str(final_path))
        wandb_run.log_artifact(artifact)
        wandb_run.log(
            {
                "final_greedy": final_greedy,
                "final_ruined": bool(final_ruined),
                "best_greedy": best_greedy,
            },
            step=episode_offset + episodes,
        )

    best_metric = best_greedy if best_greedy != float("-inf") else final_greedy
    result = {
        "best_greedy": float(best_metric),
        "final_greedy": float(final_greedy),
        "final_ruined": bool(final_ruined),
        "outdir": str(outdir),
        "metrics_path": str(metrics_path),
        "episodes": episodes,
        "rollout_steps": base_rollout_steps,
        "last_metrics": last_metrics or {},
    }
    if wandb_run:
        wandb_run.finish()
    return result


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
    train_agent(config, cfg_path=cfg_path)


if __name__ == "__main__":
    main()
