from __future__ import annotations

"""
Reset completo dos artefatos de exper_corr_neg.

Uso:
    poetry run python -m src.strategies.exper_corr_neg.scripts.reset_reports
    poetry run python -m src.strategies.exper_corr_neg.scripts.reset_reports --force

O script remove:
  - src/strategies/exper_corr_neg/reports/train/
  - src/strategies/exper_corr_neg/reports/train/pop/
  - src/strategies/exper_corr_neg/reports/walk_forward/
Recria a pasta `reports/train` vazia e executa `clean_train_reports --keep-ep 0`
para garantir consistência (mesmo sem arquivos).
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
    parser = argparse.ArgumentParser(description="Reseta os artefatos da estratégia exper_corr_neg")
    parser.add_argument("--force", action="store_true", help="não pede confirmação")
    args = parser.parse_args()

    print("Este script vai APAGAR todos os artefatos em:")
    print(f"  - {TRAIN_DIR}")
    print(f"  - {POP_DIR}")
    print(f"  - {WF_DIR}")
    print("Use apenas se quiser começar o experimento do zero.")

    if not args.force and not prompt_yes_no("Confirma (y/N)? "):
        print("[reset] Abortado pelo usuário.")
        return

    remove_path(TRAIN_DIR)
    remove_path(WF_DIR)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "poetry",
            "run",
            "python",
            "-m",
            "src.strategies.exper_corr_neg.scripts.clean_train_reports",
            "--dir",
            str(TRAIN_DIR),
            "--keep-ep",
            "0",
        ],
        check=False,
    )

    print("[reset] Concluído. Diretório pronto para novo treinamento.")


if __name__ == "__main__":
    main()

