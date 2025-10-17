from __future__ import annotations

"""
Pipeline WFO 1-clique: update (implícito) -> walk_forward -> report_wfo.
"""

from importlib import import_module


def main() -> None:
    # walk_forward atualiza cache se data.update_cache=true
    wfo = import_module("src.strategies.experimento.scripts.walk_forward")
    wfo.main()

    # Relatórios agregados do WFO mais recente (ou reconstrói do DB se artifacts desativados)
    wreport = import_module("src.strategies.experimento.scripts.report_wfo")
    wreport.main()


if __name__ == "__main__":
    main()

