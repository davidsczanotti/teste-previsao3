from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path


DEFAULT_RUNS_ROOT = Path("src/strategies/exper_corr_pos/reports/runs")
CONFIG_PATH = Path("src/strategies/exper_corr_pos/config.json")
SCOREBOARD_PATH = Path("src/strategies/exper_corr_pos/reports/train/pop/scoreboard.json")
CHECKPOINT_PATH = Path("src/strategies/exper_corr_pos/reports/train/moe_policy_final.pt")


def default_run_dir(prefix: str | None = None) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}" if not prefix else f"{prefix}_{timestamp}"
    return DEFAULT_RUNS_ROOT / base_name


def copy_if_exists(src: Path, dst: Path, label: str) -> None:
    if not src.exists():
        print(f"[aviso] {label} não encontrado em {src} — pulando cópia.")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copiado {label}: {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arquiva config.json, scoreboard.json e moe_policy_final.pt para rastreabilidade."
    )
    parser.add_argument(
        "--prefix",
        default="run",
        help="Prefixo opcional para o diretório de saída (default: 'run'). "
        "O timestamp é acrescentado automaticamente.",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Diretório de destino. Quando não especificado, usa reports/runs/<prefix>_<timestamp>.",
    )
    parser.add_argument(
        "--include-checkpoint",
        action="store_true",
        help="Também copia reports/train/moe_policy_final.pt para o diretório.",
    )
    args = parser.parse_args()

    out_dir = Path(args.dest) if args.dest else default_run_dir(args.prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    copy_if_exists(CONFIG_PATH, out_dir / "config.json", "config.json")
    copy_if_exists(SCOREBOARD_PATH, out_dir / "scoreboard.json", "scoreboard")
    if args.include_checkpoint:
        copy_if_exists(CHECKPOINT_PATH, out_dir / "moe_policy_final.pt", "checkpoint final")

    print(f"[ok] arquivos arquivados em {out_dir}")


if __name__ == "__main__":
    main()

