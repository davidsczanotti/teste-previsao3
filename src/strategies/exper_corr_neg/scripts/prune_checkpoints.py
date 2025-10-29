from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List


CHECKPOINT_PATTERN = re.compile(r"moe_policy_ep(\d+)\.pt$")


def find_checkpoint_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return []
    # includes root itself if it has checkpoints plus all subdirectories
    dirs: List[Path] = []
    for path in root.rglob("*"):
        if path.is_dir():
            dirs.append(path)
    # make sure root is processed first
    return [root, *dirs]


def prune_dir(directory: Path, keep: int, dry_run: bool) -> None:
    checkpoints = []
    for file in directory.iterdir():
        match = CHECKPOINT_PATTERN.match(file.name)
        if match:
            episode = int(match.group(1))
            checkpoints.append((episode, file))
    if not checkpoints:
        return

    checkpoints.sort(key=lambda x: x[0])
    to_keep = checkpoints[-keep:] if keep > 0 else []
    keep_paths = {path for _, path in to_keep}

    for _, path in checkpoints:
        if path not in keep_paths:
            if dry_run:
                print(f"[dry-run] removeria {path}")
            else:
                try:
                    path.unlink()
                    print(f"removido {path}")
                except Exception as exc:
                    print(f"falha ao remover {path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove checkpoints moe_policy_ep*.pt antigos mantendo apenas os mais recentes."
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=[],
        help="Diretório base para limpeza (pode ser informado múltiplas vezes). "
        "Default: train/ e train/pop/",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Quantidade de checkpoints mo_policy_ep*.pt a manter por diretório (default: 3)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Apenas lista o que seria removido.")
    args = parser.parse_args()

    default_dirs = [
        Path("src/strategies/exper_corr_neg/reports/train"),
        Path("src/strategies/exper_corr_neg/reports/train/pop"),
    ]
    target_dirs = [Path(d) for d in args.dir] if args.dir else default_dirs

    for base_dir in target_dirs:
        if not base_dir.exists():
            print(f"[aviso] diretório não encontrado: {base_dir}")
            continue
        for folder in find_checkpoint_dirs(base_dir):
            prune_dir(folder, args.keep, args.dry_run)


if __name__ == "__main__":
    main()
