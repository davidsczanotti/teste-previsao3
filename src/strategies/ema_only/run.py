#!/usr/bin/env python3
"""
CLI limpa para executar backtest EMA-only baseado em config.json
"""

import json
import sys
from pathlib import Path

try:
    # Execução como módulo: `python -m src.strategies.ema_only.run`
    from .backtest import run_backtest
except ImportError:  # pragma: no cover
    # Execução como script: `python src/strategies/ema_only/run.py`
    from backtest import run_backtest

def main():
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        print(f"Erro: {config_path} não encontrado.")
        sys.exit(1)

    run_backtest(str(config_path))

if __name__ == "__main__":
    main()
