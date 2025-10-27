from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path("src/strategies/exper_corr_neg")
POP_ROOT = ROOT / "reports" / "train" / "pop"
SCOREBOARD = POP_ROOT / "scoreboard.json"


def jload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def scoreboard_len() -> int:
    if not SCOREBOARD.exists():
        return 0
    try:
        data: List[Dict[str, Any]] = json.loads(SCOREBOARD.read_text())
        return len(data)
    except Exception:
        return 0


def run_cmd(cmd: List[str], env: Dict[str, str], title: str) -> None:
    print("->", title, " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="Executa PBT e, se houver campeão, continua treino principal")
    ap.add_argument("--base", default=str(ROOT / "config.json"), help="Config base")
    ap.add_argument("--skip-train", action="store_true", help="Não roda train.py mesmo havendo campeão")
    args = ap.parse_args()

    cfg = jload(Path(args.base))
    pbt_cfg = cfg.get("pbt", {})

    # Ambiente compartilhado (threads/Offline)
    env = os.environ.copy()
    env.setdefault("BINANCE_OFFLINE", "1")
    threads = str(pbt_cfg.get("threads", 1))
    env.setdefault("OMP_NUM_THREADS", threads)
    env.setdefault("MKL_NUM_THREADS", threads)
    env.setdefault("PYTORCH_NUM_THREADS", threads)

    prev_len = scoreboard_len()

    # 1) roda pop_runner com config
    pop_cmd = [
        "poetry",
        "run",
        "python",
        "-m",
        "src.strategies.exper_corr_neg.scripts.pop_runner",
        "--base",
        str(args.base),
    ]
    run_cmd(pop_cmd, env, "PBT")

    new_len = scoreboard_len()
    if new_len <= prev_len:
        print("Nenhum campeão promovido; nada a retomar no train.")
        return

    print(f"Campeão promovido (scoreboard: {prev_len} -> {new_len}).")
    if args.skip_train:
        print("--skip-train definido; treino principal não será executado.")
        return

    # 2) continua treino principal a partir do campeão
    train_cmd = [
        "poetry",
        "run",
        "python",
        "-m",
        "src.strategies.exper_corr_neg.train",
        "--config",
        str(args.base),
    ]
    run_cmd(train_cmd, env, "Train")


if __name__ == "__main__":
    main()

