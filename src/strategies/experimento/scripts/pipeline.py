from __future__ import annotations

"""
Pipeline 1-clique: update (implícito) -> backtest -> monte_carlo -> report.
Respeita o JSON (sem flags) e encadeia os passos principais.
"""

from importlib import import_module


def main() -> None:
    # backtest já atualiza o cache automaticamente se data.update_cache=true
    backtest = import_module("src.strategies.experimento.scripts.backtest")
    backtest.main()

    # Monte Carlo no último run
    monte = import_module("src.strategies.experimento.scripts.monte_carlo")
    monte.main()

    # Gera relatório do último run
    report = import_module("src.strategies.experimento.scripts.report")
    report.main()


if __name__ == "__main__":
    main()

