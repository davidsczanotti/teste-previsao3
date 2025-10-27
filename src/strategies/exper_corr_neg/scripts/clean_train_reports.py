from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


DEFAULT_DIR = Path("src/strategies/exper_corr_neg/reports/train")


def keep_files(train_dir: Path, keep_ep: int) -> List[Path]:
    patterns_keep = [
        "metrics.csv",
        "metrics.png",
        "expert_usage.png",
        "gating_heatmap.png",
        "gating_usage.png",
        "moe_policy_best_eval.pt",
        "moe_policy_final.pt",
    ]
    keep: List[Path] = []
    for name in patterns_keep:
        p = train_dir / name
        if p.exists():
            keep.append(p)

    eps = sorted(train_dir.glob("moe_policy_ep*.pt"))
    if eps:
        keep.extend(eps[-keep_ep:])
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description="Limpa reports/train mantendo arquivos essenciais")
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--keep-ep", type=int, default=3, help="quantidade de checkpoints epXX a manter")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train_dir = Path(args.dir)
    if not train_dir.exists():
        print(f"Diretório não existe: {train_dir}")
        return

    keep = set(p.resolve() for p in keep_files(train_dir, args.keep_ep))
    removed = []
    for p in train_dir.iterdir():
        if p.resolve() in keep:
            continue
        if p.is_dir():
            continue
        if p.suffix == ".json":
            continue
        removed.append(p)

    if args.dry_run:
        print("DRY-RUN: Arquivos que seriam removidos:")
        for p in removed:
            print("  ", p)
        return

    for p in removed:
        try:
            p.unlink()
            print("removido:", p)
        except Exception as e:
            print("falha ao remover:", p, e)

    print("Concluído. Mantidos:")
    for p in keep:
        print("  ", p)


if __name__ == "__main__":
    main()

