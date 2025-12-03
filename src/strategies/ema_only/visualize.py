from __future__ import annotations

"""
Gera gráficos simples de métricas e ações para o agente RL ema_only.

Comandos:

  BINANCE_OFFLINE=1 poetry run python -m src.strategies.ema_only.visualize
"""

from .rl_visualize import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()

