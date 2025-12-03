from __future__ import annotations

"""
Entrada de treino RL para ema_only.

Uso recomendado (offline, cache já populado):

  BINANCE_OFFLINE=1 poetry run python -m src.strategies.ema_only.train
"""

from .rl_train import main as rl_main


def main() -> None:
    rl_main()


if __name__ == "__main__":
    main()

