from __future__ import annotations

"""
PBT‑lite (população de runs) para o experimento exper_corr_pos.

Executa N treinamentos em paralelo (WSL/CPU friendly), cada um com pequenas
variações de hiperparâmetros. Ao final de cada round, escolhe o campeão
por greedy_equity (sem ruína), atualiza os demais para retomar do campeão e
inicia o próximo round.

Uso típico (na raiz do projeto):

  BINANCE_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_NUM_THREADS=2 \
    poetry run python -m src.strategies.exper_corr_pos.scripts.pop_runner \
      --base src/strategies/exper_corr_pos/config.json \
      --pop 2 --rounds 3 --episodes 400 --concurrency 2

Os artefatos ficam em: src/strategies/exper_corr_pos/reports/train/pop/
  - configs/run_{i}_round_{r}.json
  - run_{i}/round_{r}/ (outdir por round)
  - scoreboard.json (histórico de campeões)
"""

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


ROOT = Path("src/strategies/exper_corr_pos")
POP_ROOT = ROOT / "reports" / "train" / "pop"
CONF_ROOT = POP_ROOT / "configs"


def jload(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def jdump(obj: Dict[str, Any], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def mutate_cfg(base: Dict[str, Any], seed: int) -> Dict[str, Any]:
    rnd = random.Random(seed)
    cfg = json.loads(json.dumps(base))  # deep copy

    # Mutação leve de hiperparâmetros mais sensíveis
    model = cfg.setdefault("model", {})
    train = cfg.setdefault("train", {})
    ppo = cfg.setdefault("ppo", {})
    train["seed"] = int(seed)

    # temperature ±0.2 (limitando entre 0.8 e 2.0)
    temp = float(model.get("temperature", 1.2)) + rnd.uniform(-0.2, 0.2)
    model["temperature"] = float(max(0.6, min(2.0, temp)))

    # top_k ∈ {1,2}
    model["top_k"] = int(rnd.choice([1, 2]))

    # lb_coef em [0.02, 0.06]
    lb = float(train.get("lb_coef", 0.02))
    lb = max(0.01, min(0.08, lb + rnd.uniform(-0.01, 0.02)))
    train["lb_coef"] = float(lb)

    # learning_rate × {0.7, 1.0, 1.3}
    lr = float(ppo.get("learning_rate", 3e-4))
    lr *= rnd.choice([0.7, 1.0, 1.3])
    ppo["learning_rate"] = float(lr)

    return cfg


def last_valid_eval(metrics_csv: Path, eval_every: int) -> Optional[float]:
    if not metrics_csv.exists():
        return None
    try:
        df = pd.read_csv(metrics_csv)
        if "episode" not in df.columns:
            return None
        if eval_every > 0:
            df = df[df["episode"] % eval_every == 0]
        if "greedy_equity" not in df.columns:
            return None
        df = df[pd.to_numeric(df["greedy_equity"], errors="coerce").notna()]
        if "greedy_ruined" in df.columns:
            df = df[df["greedy_ruined"].fillna(1) == 0]
        if df.empty:
            return None
        return float(df["greedy_equity"].iloc[-1])
    except Exception:
        return None


def best_eval(metrics_csvs: List[Path], eval_every: int) -> Optional[Path]:
    best_path: Optional[Path] = None
    best_val = -math.inf
    for p in metrics_csvs:
        val = last_valid_eval(p, eval_every)
        if val is not None and val > best_val:
            best_val, best_path = val, p
    return best_path


def run_process(cfg_path: Path, env_extra: Dict[str, str]) -> subprocess.Popen:
    cmd = [
        "poetry",
        "run",
        "python",
        "-m",
        "src.strategies.exper_corr_pos.train",
        "--config",
        str(cfg_path),
    ]
    env = os.environ.copy()
    env.update(env_extra)
    env.setdefault("BINANCE_OFFLINE", "1")
    return subprocess.Popen(cmd, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="PBT‑lite runner (população de runs)")
    ap.add_argument("--base", default=str(ROOT / "config.json"))
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None, help="threads por processo")
    ap.add_argument(
        "--seed_checkpoint",
        type=str,
        default=None,
        help="checkpoint inicial para todos os runs (resume=True no round 0)",
    )
    ap.add_argument(
        "--promote_to_root",
        action="store_true",
        help="Copia o checkpoint do campeão para reports/train/moe_policy_final.pt",
    )
    args = ap.parse_args()

    base = jload(Path(args.base))
    eval_every = int(base.get("train", {}).get("eval_every", 50))
    pbt_cfg = base.get("pbt", {})

    pop_size = args.pop if args.pop is not None else int(pbt_cfg.get("pop", 2))
    rounds = args.rounds if args.rounds is not None else int(pbt_cfg.get("rounds", 1))
    episodes = args.episodes if args.episodes is not None else int(pbt_cfg.get("episodes", 400))
    concurrency = args.concurrency if args.concurrency is not None else int(pbt_cfg.get("concurrency", 1))
    threads = args.threads if args.threads is not None else int(pbt_cfg.get("threads", 1))
    seed_checkpoint = (
        args.seed_checkpoint
        if args.seed_checkpoint is not None
        else pbt_cfg.get("seed_checkpoint", "")
    )
    promote_to_root = bool(args.promote_to_root or pbt_cfg.get("promote_to_root", False))

    POP_ROOT.mkdir(parents=True, exist_ok=True)
    CONF_ROOT.mkdir(parents=True, exist_ok=True)
    scoreboard_path = POP_ROOT / "scoreboard.json"
    scoreboard: List[Dict[str, Any]] = []
    if scoreboard_path.exists():
        try:
            scoreboard = json.loads(scoreboard_path.read_text())
        except Exception:
            scoreboard = []

    base_seed = int(time.time()) % 10_000

    run_cfgs: List[Dict[str, Any]] = []
    for i in range(pop_size):
        cfg = mutate_cfg(base, base_seed + i)
        cfg.setdefault("train", {})
        cfg["train"]["episodes"] = int(episodes)
        if seed_checkpoint:
            cfg["train"]["resume"] = True
            cfg["train"]["resume_path"] = seed_checkpoint
        else:
            cfg["train"]["resume"] = False
        cfg["train"]["outdir"] = str(POP_ROOT / f"run_{i}" / "round_0")
        run_cfgs.append(cfg)

    env_threads = {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "PYTORCH_NUM_THREADS": str(threads),
    }

    for r in range(rounds):
        procs: List[subprocess.Popen] = []
        cfg_paths: List[Path] = []
        for i, cfg in enumerate(run_cfgs):
            outdir = POP_ROOT / f"run_{i}" / f"round_{r}"
            cfg["train"]["outdir"] = str(outdir)
            cfg_path = CONF_ROOT / f"run_{i}_round_{r}.json"
            jdump(cfg, cfg_path)
            cfg_paths.append(cfg_path)

        for i, cfg_path in enumerate(cfg_paths):
            while len(procs) >= concurrency:
                procs[0].wait()
                procs.pop(0)
            procs.append(run_process(cfg_path, env_threads))

        for p in procs:
            p.wait()

        metrics_list = [
            (POP_ROOT / f"run_{i}" / f"round_{r}" / "metrics.csv") for i in range(pop_size)
        ]
        best_metrics = best_eval(metrics_list, eval_every)
        if best_metrics is None:
            print(f"[round {r}] Nenhum candidato válido (greedy sem ruína). Pulando promoção.")
            continue

        champion_dir = best_metrics.parent
        champion_ckpt = champion_dir / "moe_policy_final.pt"
        if not champion_ckpt.exists():
            eps = sorted(champion_dir.glob("moe_policy_ep*.pt"))
            champion_ckpt = eps[-1] if eps else champion_ckpt
        print(f"[round {r}] Campeão: {champion_dir} -> {champion_ckpt}")

        scoreboard.append({
            "round": r,
            "champion": str(champion_dir),
            "checkpoint": str(champion_ckpt),
            "metrics": str(best_metrics),
        })
        jdump(scoreboard, scoreboard_path)

        if promote_to_root and champion_ckpt.exists():
            root_final = ROOT / "reports" / "train" / "moe_policy_final.pt"
            root_final.write_bytes(champion_ckpt.read_bytes())
            print(f"  -> promovido para {root_final}")

        if r + 1 < rounds:
            for i in range(pop_size):
                new_cfg = mutate_cfg(base, base_seed + (r + 1) * 100 + i)
                new_cfg.setdefault("train", {})
                new_cfg["train"]["episodes"] = int(episodes)
                new_cfg["train"]["resume"] = True
                new_cfg["train"]["resume_path"] = str(champion_ckpt)
                new_cfg["train"]["outdir"] = str(POP_ROOT / f"run_{i}" / f"round_{r+1}")
                run_cfgs[i] = new_cfg

    print(f"Concluído. Scoreboard em {scoreboard_path}")


if __name__ == "__main__":
    main()
