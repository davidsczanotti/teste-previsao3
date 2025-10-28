from __future__ import annotations

"""
Limpa todos os artefatos da estratégia exper_corr_neg para começar do zero.

Uso:
    poetry run python -m scripts.reset_exper_corr_neg_reports

O script pergunta confirmação antes de apagar:
- src/strategies/exper_corr_neg/reports/train/
- src/strategies/exper_corr_neg/reports/walk_forward/
- src/strategies/exper_corr_neg/reports/train/pop/

Após a limpeza, executa:
    poetry run python -m src.strategies.exper_corr_neg.scripts.clean_train_reports --keep-ep 0
para recriar a estrutura básica e garantir consistência.
"""

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path("src/strategies/exper_corr_neg/reports")
TRAIN_DIR = ROOT / "train"
POP_DIR = TRAIN_DIR / "pop"
WF_DIR = ROOT / "walk_forward"


def remove_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"[reset] Removido: {path}")
    else:
        print(f"[reset] Ignorado (não existe): {path}")


def prompt_yes_no(message: str) -> bool:
    answer = input(message).strip().lower()
    return answer in {"y", "yes", "s", "sim"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset exper_corr_neg reports directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="não pergunta confirmação (USE COM CUIDADO)",
    )
    args = parser.parse_args()

    print("Este script vai apagar TODOS os artefatos em:")
    print(f"  - {TRAIN_DIR}")
    print(f"  - {POP_DIR}")
    print(f"  - {WF_DIR}")
    print("Use somente se quiser realmente começar do zero.")
    if not args.force and not prompt_yes_no("Confirma (y/N)? "):
        print("Abortado.")
        return

    remove_path(TRAIN_DIR)
    remove_path(WF_DIR)

    ROOT.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "poetry",
            "run",
            "python",
            "-m",
            "src.strategies.exper_corr_neg.scripts.clean_train_reports",
            "--keep-ep",
            "0",
        ],
        check=False,
    )
    print("[reset] Concluído. Pasta limpa e pronta para novo treino.")


if __name__ == "__main__":
    main()
